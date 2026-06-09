import base64
from datetime import datetime
from pathlib import Path

from openai import AzureOpenAI


class Voice:
    alloy: str = 'alloy'
    ash: str = 'ash'
    ballad: str = 'ballad'
    coral: str = 'coral'
    echo: str = 'echo'
    fable: str = 'fable'
    nova: str = 'nova'
    onyx: str = 'onyx'
    sage: str = 'sage'
    shimmer: str = 'shimmer'


# https://developers.openai.com/api/docs/guides/text-to-speech

client = AzureOpenAI()

completion = client.chat.completions.create(
    model="gpt-audio-mini-2025-10-06",
    modalities=["text", "audio"],
    audio={"voice": Voice.alloy, "format": "wav"},
    messages=[{
        "role": "user",
        "content": "Why can't we say that black is white?"
    }]
)

wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
filename = Path(__file__).parent / f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
filename.write_bytes(wav_bytes)
print(f"Saved: {filename}")
