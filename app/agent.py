"""LiveKit Agent worker: front-desk healthcare voice assistant.

Pipeline: Deepgram STT -> Groq LLM (tool calling) -> Cartesia TTS,
with a Beyond Presence avatar video track and a data-channel feed that
mirrors every tool call to the frontend in real time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, UTC

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import bey, cartesia, deepgram, groq

from app.db import (
    AppointmentNotFoundError,
    Database,
    DoubleBookingError,
)
from app.summary import generate_call_summary

load_dotenv()

logger = logging.getLogger("front-desk-agent")
logger.setLevel(logging.INFO)

db = Database()

INSTRUCTIONS = """
You are Aria, a friendly front-desk voice assistant for a healthcare clinic.
You speak naturally and concisely, like a real receptionist on the phone — short sentences, no markdown, no lists read aloud.

Conversation flow:
1. Open warmly: welcome them to the clinic, ask how they're doing today, then ask how you can help.
   Keep it to 1-2 short sentences — warm, not gushing.
2. Before doing anything account-specific (booking, retrieving, cancelling, modifying), you MUST identify the caller:
   ask for their phone number, WAIT for them to actually say it, and only then call `identify_user` with the
   exact digits they spoke. Also ask their name if you don't have it yet, and pass it in.
   NEVER call `identify_user` with a guessed, example, or placeholder phone number (e.g. "123456789",
   "unknown", "0000000000"). If the caller hasn't said their number yet, ask for it and wait — do not call
   any tool until you have their real spoken number.
3. To book an appointment: find out what date/timeframe they want, call `fetch_slots` to see real availability,
   offer 2-3 concrete options out loud, then call `book_appointment` once they pick one. Always confirm the
   final date and time back to the caller clearly after booking.
4. To check existing bookings, call `retrieve_appointments`.
5. To cancel or reschedule, call `cancel_appointment` or `modify_appointment`. Always confirm details before
   calling, and always state the final outcome clearly afterward.
6. If a slot is already taken, apologize briefly and offer the nearest alternatives from `fetch_slots`.
7. When the caller is done (says bye, thanks, that's all, etc.), call `end_conversation`.

Always extract and remember: caller's name, phone number, requested date/time, and their intent (why they're calling).
Never invent appointment data — only state what tools actually returned. Keep responses short; this is a phone call, not a chat window.

Today's date is {today} ({weekday}). Resolve relative dates the caller says ("tomorrow", "next Monday",
"this Friday") against this before calling any tool — tools only accept absolute YYYY-MM-DD dates.
"""

PHONE_RE = re.compile(r"\D")


def normalize_phone(raw: str) -> str:
    digits = PHONE_RE.sub("", raw)
    return digits[-10:] if len(digits) >= 10 else digits


def looks_like_placeholder_phone(digits: str) -> bool:
    """Catches LLM-hallucinated numbers (sequential, repeated-digit) that would
    otherwise pass basic length validation, e.g. "123456789" or "0000000000"."""
    if len(set(digits)) <= 2:
        return True
    deltas = {(int(b) - int(a)) % 10 for a, b in zip(digits, digits[1:])}
    return deltas in ({1}, {9})  # strictly ascending or strictly descending


class FrontDeskAgent(Agent):
    def __init__(self, job_ctx: JobContext):
        now = datetime.now(UTC)
        instructions = INSTRUCTIONS.format(today=now.date().isoformat(), weekday=now.strftime("%A"))
        super().__init__(instructions=instructions)
        self._job_ctx = job_ctx
        self._room = job_ctx.room
        self.phone: str | None = None
        self.name: str | None = None
        self.intent: str | None = None
        self.preferences: list[str] = []
        self.ended = False

    # ---- UI event helper -------------------------------------------------

    async def _notify(self, tool: str, status: str, message: str, data: dict | None = None) -> None:
        payload = json.dumps(
            {
                "type": "tool_call",
                "tool": tool,
                "status": status,
                "message": message,
                "data": data or {},
                "ts": datetime.now(UTC).isoformat(),
            }
        )
        try:
            await self._room.local_participant.publish_data(payload, topic="tool-status", reliable=True)
        except Exception:
            logger.exception("failed to publish tool-status event")

    # ---- Tools -------------------------------------------------------------

    @function_tool()
    async def identify_user(self, phone_number: str, caller_name: str | None = None) -> str:
        """Identify the caller by phone number. Always call this before any booking/retrieval/cancel/modify action.

        Args:
            phone_number: The caller's phone number, spoken digits normalized (e.g. "9876543210").
            caller_name: The caller's name, if they have given it.
        """
        await self._notify("identify_user", "running", "Looking up caller...")
        phone = normalize_phone(phone_number)
        if len(phone) < 7 or looks_like_placeholder_phone(phone):
            await self._notify("identify_user", "error", "No real phone number provided yet.")
            return (
                "You have not actually been told a phone number yet — that looks like a guessed or "
                "placeholder value. Ask the caller to say their phone number out loud, wait for their "
                "answer, then call identify_user again with exactly what they said."
            )
        info = await asyncio.to_thread(db.get_or_create_user, phone, caller_name)
        self.phone = phone
        self.name = info["name"]
        await self._notify(
            "identify_user", "done", f"Caller identified ({phone})", {"phone": phone, "name": self.name}
        )
        greeting = "Welcome back" if not info["is_new"] else "Got it, nice to meet you"
        return f"{greeting}. Caller identified with phone {phone}, name: {self.name or 'unknown'}."

    @function_tool()
    async def fetch_slots(self, date: str | None = None) -> str:
        """Fetch available appointment slots. Returns up to 7 upcoming days if no date given.

        Args:
            date: Specific date in YYYY-MM-DD format, or omit to see the next 7 days.
        """
        await self._notify("fetch_slots", "running", "Fetching slots...")
        slots = await asyncio.to_thread(db.available_slots, date)
        await self._notify("fetch_slots", "done", "Slots fetched", {"slots": slots})
        return f"Available slots: {json.dumps(slots)}"

    @function_tool()
    async def book_appointment(self, date: str, time: str, name: str | None = None) -> str:
        """Book an appointment for the already-identified caller. Call identify_user first.

        Args:
            date: Date in YYYY-MM-DD format.
            time: Time in 24h HH:MM format, must be one of the slots returned by fetch_slots.
            name: Caller's name, if known and not already captured.
        """
        if not self.phone:
            return "Caller is not identified yet. Ask for their phone number and call identify_user first."
        await self._notify("book_appointment", "running", f"Booking {date} {time}...")
        name = name or self.name
        try:
            appt = await asyncio.to_thread(db.book_appointment, self.phone, name, date, time)
        except DoubleBookingError:
            alternatives = await asyncio.to_thread(db.available_slots, date)
            await self._notify(
                "book_appointment", "error", f"Slot {date} {time} already booked", {"alternatives": alternatives}
            )
            return (
                f"That slot ({date} {time}) is already booked. Offer these alternatives: "
                f"{json.dumps(alternatives.get(date, []))}"
            )
        except ValueError as e:
            await self._notify("book_appointment", "error", str(e))
            return str(e)
        await self._notify(
            "book_appointment", "done", f"Booking confirmed for {date} {time} ✅", appt.to_dict()
        )
        return f"Booking confirmed for {date} at {time}. Confirm this clearly to the caller."

    @function_tool()
    async def retrieve_appointments(self, phone_number: str | None = None) -> str:
        """Retrieve the caller's past and upcoming appointments.

        Args:
            phone_number: Phone number to look up; omit to use the already-identified caller.
        """
        phone = normalize_phone(phone_number) if phone_number else self.phone
        if not phone:
            return "No phone number available. Ask for it and call identify_user first."
        await self._notify("retrieve_appointments", "running", "Fetching bookings...")
        appts = await asyncio.to_thread(db.list_appointments, phone)
        data = [a.to_dict() for a in appts]
        await self._notify("retrieve_appointments", "done", f"Found {len(data)} booking(s)", {"appointments": data})
        if not data:
            return "No appointments found for this caller."
        return f"Appointments: {json.dumps(data)}"

    @function_tool()
    async def cancel_appointment(self, date: str, time: str) -> str:
        """Cancel the caller's appointment at the given date/time.

        Args:
            date: Date in YYYY-MM-DD format.
            time: Time in 24h HH:MM format.
        """
        if not self.phone:
            return "Caller is not identified yet. Ask for their phone number and call identify_user first."
        await self._notify("cancel_appointment", "running", f"Cancelling {date} {time}...")
        try:
            appt = await asyncio.to_thread(db.cancel_appointment, self.phone, date, time)
        except AppointmentNotFoundError as e:
            await self._notify("cancel_appointment", "error", str(e))
            return str(e)
        await self._notify("cancel_appointment", "done", f"Cancelled {date} {time} ✅", appt.to_dict())
        return f"Appointment on {date} at {time} has been cancelled. Confirm this to the caller."

    @function_tool()
    async def modify_appointment(self, old_date: str, old_time: str, new_date: str, new_time: str) -> str:
        """Reschedule an existing appointment to a new date/time.

        Args:
            old_date: Current appointment date in YYYY-MM-DD format.
            old_time: Current appointment time in HH:MM format.
            new_date: New date in YYYY-MM-DD format.
            new_time: New time in HH:MM format, must be a valid slot.
        """
        if not self.phone:
            return "Caller is not identified yet. Ask for their phone number and call identify_user first."
        await self._notify("modify_appointment", "running", f"Rescheduling to {new_date} {new_time}...")
        try:
            appt = await asyncio.to_thread(
                db.modify_appointment, self.phone, old_date, old_time, new_date, new_time
            )
        except DoubleBookingError as e:
            await self._notify("modify_appointment", "error", str(e))
            return str(e)
        except AppointmentNotFoundError as e:
            await self._notify("modify_appointment", "error", str(e))
            return str(e)
        except ValueError as e:
            await self._notify("modify_appointment", "error", str(e))
            return str(e)
        await self._notify(
            "modify_appointment", "done", f"Rescheduled to {new_date} {new_time} ✅", appt.to_dict()
        )
        return f"Appointment moved to {new_date} at {new_time}. Confirm this clearly to the caller."

    @function_tool()
    async def end_conversation(self, intent_summary: str | None = None) -> str:
        """End the call gracefully. Call this once the caller is done and you have said goodbye.

        Args:
            intent_summary: A short phrase describing why the caller called (their intent), e.g. "book a checkup".
        """
        if self.ended:
            return "Conversation already ended."
        self.ended = True
        self.intent = intent_summary or self.intent
        await self._notify("end_conversation", "running", "Wrapping up the call...")

        async def _close():
            await asyncio.sleep(1.5)  # let the goodbye audio finish playing
            try:
                appts = await asyncio.to_thread(db.list_appointments, self.phone) if self.phone else []
                summary_text, preferences = await generate_call_summary(
                    self.session.history, self.name, self.phone, self.intent
                )
                record = await asyncio.to_thread(
                    db.save_call_summary,
                    self.phone,
                    self._room.name,
                    summary_text,
                    [a.to_dict() for a in appts],
                    preferences,
                    self.intent,
                )
                await self._notify("end_conversation", "done", "Call summary ready 📝", record)
            except Exception:
                logger.exception("failed to generate/save call summary")
                await self._notify("end_conversation", "error", "Failed to generate summary")
            await asyncio.sleep(1)
            self._job_ctx.shutdown()

        asyncio.create_task(_close())
        return "Say a brief, warm goodbye to the caller now. Do not call any other tool after this."


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = FrontDeskAgent(ctx)

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", smart_format=True, numerals=True),
        llm=groq.LLM(model="llama-3.3-70b-versatile", temperature=0.4),
        tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
        turn_handling={
            # Default max_delay=2.5s is how long the model can wait, when unsure the
            # caller has finished talking, before replying. That reads as dead air on
            # a phone call, so cap it much tighter.
            "endpointing": {"min_delay": 0.2, "max_delay": 0.8},
            # The default "adaptive" (ML-based) interruption detector ignores
            # min_duration/min_words entirely and uses its own probability threshold,
            # which was firing on sub-100ms noise bursts (breath, echo of the agent's own
            # voice through speakers) and cutting the agent off mid-sentence. "vad" mode
            # is deterministic and actually honors these thresholds. Thresholds are biased
            # high (vs. the usual 0.5s/0 words) because our audio output routes through the
            # avatar and can't pause/resume — a false interruption permanently drops the
            # rest of that sentence, so it's worth trading a little barge-in speed for
            # fewer false positives.
            "interruption": {
                "mode": "vad",
                "min_duration": 0.8,
                "min_words": 3,
                "resume_false_interruption": False,
            },
        },
    )

    avatar = bey.AvatarSession(avatar_id=os.environ.get("BEY_AVATAR_ID") or None)
    await avatar.start(session, room=ctx.room)

    await session.start(agent=agent, room=ctx.room)

    # A fixed greeting goes straight to TTS with no LLM round-trip, so there's no visible
    # "thinking" step between the avatar becoming ready and Aria actually speaking.
    session.say("Welcome to our clinic! How are you doing today? How can I help you?")


if __name__ == "__main__":
    # Keep one process warm so a call doesn't pay process cold-start latency on top of
    # the avatar provisioning time.
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, num_idle_processes=1))
