import base64
from datetime import datetime
from pathlib import Path

from openai import AzureOpenAI


# https://developers.openai.com/api/docs/guides/audio#add-audio-to-your-existing-application

client = AzureOpenAI()
audio_path = Path(__file__).parent / "question.mp3"
b64_audio = base64.b64encode(audio_path.read_bytes()).decode()

completion = client.chat.completions.create(
    model="gpt-audio-mini-2025-10-06",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[{
        "role": "user",
        "content": [{"type": "input_audio", "input_audio": {"data": b64_audio, "format": "mp3"}}]
    }]
)

wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
filename = Path(__file__).parent / f"answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
filename.write_bytes(wav_bytes)
print(f"Saved: {filename}")
