"""Demo script for the local_mic adapter.

Shows the full pipeline:
  WAV file -> on_message (VAD + trust + STT stub) -> ChannelMessage
  ChannelReply -> send (system_fallback TTS) -> speaker

Run:
    uv run python demo_local_mic.py

No API keys needed. Uses system_fallback TTS (macOS `say` command).
STT is stubbed to return a fixed transcript so no STT provider is needed.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from glc.channels.catalogue.local_mic.adapter import Adapter
from glc.channels.envelope import ChannelReply
from glc.security.pairing import get_pairing_store
from glc.voice.stt.base import STTProvider, TranscribeResult
from glc.voice.stt.router import register_test_provider

SPEAKER_ID = "owner"
TRANSCRIPT = "Hello, this is a local mic demo."


class _StubSTT(STTProvider):
    name = "groq_whisper"

    async def transcribe(self, audio: bytes, mime: str) -> TranscribeResult:
        return TranscribeResult(
            text=TRANSCRIPT,
            language="en",
            duration_ms=1000,
            provider="groq_whisper",
            cost_usd=0.0,
        )


def _make_wav(duration_s: float = 1.0, amplitude: float = 0.3) -> bytes:
    import io
    import math
    import struct
    import wave

    n = int(duration_s * 16000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        samples = bytearray()
        for i in range(n):
            v = int(amplitude * 32767 * math.sin(2 * math.pi * 180 * i / 16000))
            samples += struct.pack("<h", v)
        wf.writeframes(bytes(samples))
    return buf.getvalue()


async def main() -> None:
    # Register stub STT so on_message can transcribe without a real provider
    register_test_provider("groq_whisper", _StubSTT())

    # Pair the owner so trust_level == "owner_paired"
    store = get_pairing_store()
    store.force_pair_owner("local_mic", SPEAKER_ID, user_handle="owner")

    adapter = Adapter(config={"tts_prefer": "fallback"})

    # Build a fake inbound event (simulates what a recording loop would produce)
    raw = {
        "wav_bytes": _make_wav(amplitude=0.3),
        "sample_rate": 16000,
        "source": "mic",
        "speaker_id": SPEAKER_ID,
        "speaker_handle": "owner",
    }

    print("--- on_message ---")
    msg = await adapter.on_message(raw)
    if msg is None:
        print("VAD: silent — no envelope produced")
        return

    print(f"channel        : {msg.channel}")
    print(f"channel_user_id: {msg.channel_user_id}")
    print(f"trust_level    : {msg.trust_level}")
    print(f"text           : {msg.text}")
    print(f"voice_audio_ref: {msg.voice_audio_ref}")
    print(f"arrived_at     : {msg.arrived_at}")

    # Simulate what the gateway echo would reply
    reply_text = f"[echo] {msg.text}"
    reply = ChannelReply(
        channel="local_mic",
        channel_user_id=SPEAKER_ID,
        text=reply_text,
    )

    print(f"\n--- send (system_fallback TTS) ---")
    print(f"speaking: {reply_text!r}")
    result = await adapter.send(reply)
    print(f"result: {result}")

    # Clean up
    register_test_provider("groq_whisper", None)
    store.revoke("local_mic", SPEAKER_ID)


if __name__ == "__main__":
    asyncio.run(main())
