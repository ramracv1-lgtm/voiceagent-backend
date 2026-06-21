"""Generates the end-of-call summary via Groq, fast enough to land within the 10s budget."""

from __future__ import annotations

import json
import logging
import os

from groq import AsyncGroq
from livekit.agents.llm import ChatContext

logger = logging.getLogger("front-desk-agent.summary")

_SUMMARY_SYSTEM_PROMPT = """
You summarize a finished healthcare front-desk phone call for an internal dashboard.
Given the transcript, the caller's name/phone/intent if known, respond with strict JSON only:
{
  "summary": "<2-4 sentence plain-English summary of what happened on the call>",
  "preferences": "<any stated preferences, e.g. preferred days/times/doctor, or empty string if none>"
}
No markdown, no commentary, JSON only.
"""


def _transcript_text(history: ChatContext) -> str:
    lines = []
    for item in history.items:
        if getattr(item, "type", None) != "message":
            continue
        role = "Caller" if item.role == "user" else "Agent"
        text = item.text_content
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


async def generate_call_summary(
    history: ChatContext, name: str | None, phone: str | None, intent: str | None
) -> tuple[str, str]:
    """Returns (summary_text, preferences_text). Falls back to a deterministic summary on any failure."""
    transcript = _transcript_text(history)
    fallback = (
        f"Call with {name or 'an unidentified caller'} ({phone or 'no phone captured'}). "
        f"Intent: {intent or 'not captured'}."
    )
    try:
        client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
        resp = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=300,
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Name: {name}\nPhone: {phone}\nIntent: {intent}\n\nTranscript:\n{transcript}",
                },
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("summary", fallback), data.get("preferences", "")
    except Exception:
        logger.exception("summary generation failed, using fallback")
        return fallback, ""
