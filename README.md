---
title: Voice Agent Backend
emoji: 📞
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Voice AI Front Desk — Backend

A LiveKit Agents (Python) voice pipeline for a healthcare front-desk AI: Deepgram STT → Groq LLM
(tool calling) → Cartesia TTS, plus a Beyond Presence lip-synced avatar, backed by SQLite.

## Architecture

- `app/agent.py` — the LiveKit Agent worker. One `FrontDeskAgent` per call, holding the 7 required
  tools (`identify_user`, `fetch_slots`, `book_appointment`, `retrieve_appointments`,
  `cancel_appointment`, `modify_appointment`, `end_conversation`). Every tool call publishes a
  `tool-status` data-channel event so the frontend can show "Fetching slots…", "Booking confirmed ✅", etc.
  in real time.
- `app/db.py` — SQLite layer. Double booking is prevented at the DB level via a partial unique index
  on `(date, time) WHERE status='booked'`, not just application logic.
- `app/summary.py` — generates the end-of-call summary via a one-shot Groq call against the session
  transcript, with a deterministic fallback if the LLM call fails or is slow.
- `app/server.py` — thin FastAPI side-car: issues LiveKit room tokens for the frontend, and exposes
  `GET /api/appointments/{phone}` and `GET /api/summary/{room_name}` for REST reads.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`).

```bash
uv sync               # creates .venv and installs locked dependencies
cp .env.example .env   # fill in your keys
```

Required keys (see `.env.example`):
- `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` — from your LiveKit Cloud project
- `DEEPGRAM_API_KEY` — deepgram.com (free trial credit)
- `GROQ_API_KEY` — console.groq.com (free tier, used for both conversation + summary)
- `CARTESIA_API_KEY` — cartesia.ai (free trial credit)
- `BEY_API_KEY` — app.bey.chat/settings (Beyond Presence; the default stock avatar works with no `avatar_id`)

## Run locally

Two processes:

```bash
# Terminal 1 — REST API (token issuance, appointments/summary reads)
uv run uvicorn app.server:app --reload --port 8000

# Terminal 2 — the voice agent worker (connects out to LiveKit Cloud)
uv run python -m app.agent dev
```

The frontend talks to `:8000` for tokens/REST, and to LiveKit Cloud directly (via the token) for the
realtime room — including the avatar video track and the `tool-status` data events.

## Deployment (Railway)

**Single service (recommended unless `Database` is backed by a real shared DB):** deploy one
Railway service with start command `uv run python -m app.combined start`. This runs the FastAPI
side-car and the LiveKit agent worker in the same container, sharing one filesystem — needed
because SQLite is a local file with no built-in replication. Splitting them into two services
(below) gives each one its own disk, so the worker's writes (appointments, call summaries) are
invisible to the api service's reads and `/api/summary/...`/`/api/appointments/...` will 404
forever.

**Two services** (only if you've moved `Database` onto something shared, e.g. Postgres):
1. **api** — start command `uv run uvicorn app.server:app --host 0.0.0.0 --port $PORT`
2. **worker** — start command `uv run python -m app.agent start`

Set the same env vars (from `.env.example`) on whichever service(s) you use.

## Cost per call (rough)

| Component | Rate | ~5 min call |
|---|---|---|
| Deepgram STT | ~$0.0043/min | ~$0.02 |
| Groq LLM | free tier | $0.00 |
| Cartesia TTS | ~$0.02–0.04/min audio generated | ~$0.10 |
| Beyond Presence avatar | usage-based, free dev credits | $0.00 (within trial) |
| LiveKit Cloud | free tier covers low-volume usage | $0.00 |

Real-world cost is dominated by TTS + avatar minutes once free credits run out; STT and the Groq LLM
are negligible by comparison.

## Edge cases handled

- Double booking is rejected at the DB layer (unique index), not just in agent logic — safe even
  under concurrent calls.
- Invalid/garbled phone numbers from STT are normalized and rejected if too short.
- Booking/modifying into a slot outside the fixed slot grid is rejected with a clear error fed back
  to the LLM so it can recover instead of confirming a fake booking.
- If Groq summary generation fails or times out, a deterministic fallback summary is saved instead
  of leaving the call with no summary.
