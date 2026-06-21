"""FastAPI side-car: issues LiveKit room tokens and exposes appointment/summary reads
for the frontend's REST panels (the realtime tool-call feed itself goes over the
LiveKit data channel directly from the agent, not through this API)."""

from __future__ import annotations

import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from livekit import api
from pydantic import BaseModel

from app.db import Database

load_dotenv()

db = Database()

app = FastAPI(title="Voice AI Front Desk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TokenRequest(BaseModel):
    identity: str | None = None
    room: str | None = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/token")
def create_token(req: TokenRequest):
    livekit_url = os.environ.get("LIVEKIT_URL")
    api_key = os.environ.get("LIVEKIT_API_KEY")
    api_secret = os.environ.get("LIVEKIT_API_SECRET")
    if not all([livekit_url, api_key, api_secret]):
        raise HTTPException(500, "LiveKit credentials not configured on the server.")

    identity = req.identity or f"caller-{uuid.uuid4().hex[:8]}"
    room_name = req.room or f"front-desk-{uuid.uuid4().hex[:8]}"

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room_name, can_publish=True, can_subscribe=True))
        .to_jwt()
    )
    return {"token": token, "url": livekit_url, "room": room_name, "identity": identity}


@app.get("/api/appointments/{phone}")
def get_appointments(phone: str):
    appts = db.list_appointments(phone)
    return {"phone": phone, "appointments": [a.to_dict() for a in appts]}


@app.get("/api/summary/{room_name}")
def get_summary(room_name: str):
    record = db.get_call_summary_by_room(room_name)
    if record is None:
        raise HTTPException(404, "Summary not ready yet.")
    return record
