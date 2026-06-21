"""SQLite persistence layer for users, appointments, and call summaries."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "voice_agent.db"

SLOT_TIMES = [
    "09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
    "14:00", "14:30", "15:00", "15:30", "16:00", "16:30",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    phone TEXT PRIMARY KEY,
    name TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phone TEXT NOT NULL,
    name TEXT,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'booked',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Prevents double booking: only one *active* (booked) appointment per slot.
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_active_slot
    ON appointments(date, time)
    WHERE status = 'booked';

CREATE TABLE IF NOT EXISTS call_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phone TEXT,
    room_name TEXT,
    summary TEXT,
    appointments_json TEXT,
    preferences TEXT,
    intent TEXT,
    created_at TEXT NOT NULL
);
"""


class DoubleBookingError(Exception):
    pass


class AppointmentNotFoundError(Exception):
    pass


@dataclass
class Appointment:
    id: int
    phone: str
    name: str | None
    date: str
    time: str
    status: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "phone": self.phone,
            "name": self.name,
            "date": self.date,
            "time": self.time,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class Database:
    """Thread-safe SQLite wrapper. Call methods via asyncio.to_thread from async code."""

    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(path)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get_or_create_user(self, phone: str, name: str | None = None) -> dict:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE phone = ?", (phone,)).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO users (phone, name, created_at) VALUES (?, ?, ?)",
                    (phone, name, datetime.utcnow().isoformat()),
                )
                return {"phone": phone, "name": name, "is_new": True}
            if name and not row["name"]:
                conn.execute("UPDATE users SET name = ? WHERE phone = ?", (name, phone))
            return {"phone": row["phone"], "name": name or row["name"], "is_new": False}

    def get_booked_times(self, date: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT time FROM appointments WHERE date = ? AND status = 'booked'",
                (date,),
            ).fetchall()
            return [r["time"] for r in rows]

    def available_slots(self, date: str | None = None, days_ahead: int = 7) -> dict[str, list[str]]:
        today = datetime.utcnow().date()
        if date:
            dates = [date]
        else:
            dates = [(today + timedelta(days=i)).isoformat() for i in range(1, days_ahead + 1)]
        result = {}
        for d in dates:
            booked = set(self.get_booked_times(d))
            result[d] = [t for t in SLOT_TIMES if t not in booked]
        return result

    def book_appointment(self, phone: str, name: str | None, date: str, time: str) -> Appointment:
        if time not in SLOT_TIMES:
            raise ValueError(f"'{time}' is not a valid slot time. Valid times: {SLOT_TIMES}")
        now = datetime.utcnow().isoformat()
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM appointments WHERE date = ? AND time = ? AND status = 'booked'",
                (date, time),
            ).fetchone()
            if existing is not None:
                raise DoubleBookingError(f"Slot {date} {time} is already booked.")
            cur = conn.execute(
                "INSERT INTO appointments (phone, name, date, time, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'booked', ?, ?)",
                (phone, name, date, time, now, now),
            )
            row = conn.execute("SELECT * FROM appointments WHERE id = ?", (cur.lastrowid,)).fetchone()
            return Appointment(**dict(row))

    def list_appointments(self, phone: str) -> list[Appointment]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM appointments WHERE phone = ? ORDER BY date, time", (phone,)
            ).fetchall()
            return [Appointment(**dict(r)) for r in rows]

    def find_active_appointment(self, phone: str, date: str, time: str) -> Appointment | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM appointments WHERE phone = ? AND date = ? AND time = ? AND status = 'booked'",
                (phone, date, time),
            ).fetchone()
            return Appointment(**dict(row)) if row else None

    def cancel_appointment(self, phone: str, date: str, time: str) -> Appointment:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM appointments WHERE phone = ? AND date = ? AND time = ? AND status = 'booked'",
                (phone, date, time),
            ).fetchone()
            if row is None:
                raise AppointmentNotFoundError(f"No active appointment found for {date} {time}.")
            now = datetime.utcnow().isoformat()
            conn.execute(
                "UPDATE appointments SET status = 'cancelled', updated_at = ? WHERE id = ?",
                (now, row["id"]),
            )
            updated = dict(row)
            updated["status"] = "cancelled"
            updated["updated_at"] = now
            return Appointment(**updated)

    def modify_appointment(
        self, phone: str, old_date: str, old_time: str, new_date: str, new_time: str
    ) -> Appointment:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM appointments WHERE phone = ? AND date = ? AND time = ? AND status = 'booked'",
                (phone, old_date, old_time),
            ).fetchone()
            if row is None:
                raise AppointmentNotFoundError(f"No active appointment found for {old_date} {old_time}.")
            conflict = conn.execute(
                "SELECT id FROM appointments WHERE date = ? AND time = ? AND status = 'booked' AND id != ?",
                (new_date, new_time, row["id"]),
            ).fetchone()
            if conflict is not None:
                raise DoubleBookingError(f"Slot {new_date} {new_time} is already booked.")
            if new_time not in SLOT_TIMES:
                raise ValueError(f"'{new_time}' is not a valid slot time. Valid times: {SLOT_TIMES}")
            now = datetime.utcnow().isoformat()
            conn.execute(
                "UPDATE appointments SET date = ?, time = ?, updated_at = ? WHERE id = ?",
                (new_date, new_time, now, row["id"]),
            )
            updated = dict(row)
            updated["date"] = new_date
            updated["time"] = new_time
            updated["updated_at"] = now
            return Appointment(**updated)

    def save_call_summary(
        self,
        phone: str | None,
        room_name: str,
        summary: str,
        appointments: list[dict],
        preferences: str,
        intent: str | None,
    ) -> dict:
        import json

        now = datetime.utcnow().isoformat()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO call_summaries (phone, room_name, summary, appointments_json, preferences, intent, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (phone, room_name, summary, json.dumps(appointments), preferences, intent, now),
            )
            return {
                "id": cur.lastrowid,
                "phone": phone,
                "room_name": room_name,
                "summary": summary,
                "appointments": appointments,
                "preferences": preferences,
                "intent": intent,
                "created_at": now,
            }

    def get_call_summary_by_room(self, room_name: str) -> dict | None:
        import json

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM call_summaries WHERE room_name = ? ORDER BY id DESC LIMIT 1",
                (room_name,),
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["appointments"] = json.loads(d.pop("appointments_json") or "[]")
            return d
