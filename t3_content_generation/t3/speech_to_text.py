import base64
from pathlib import Path

from openai import AzureOpenAI


# https://developers.openai.com/api/docs/guides/speech-to-text

client = AzureOpenAI()
audio_path = Path(__file__).parent / "audio_sample.mp3"
b64_audio = base64.b64encode(audio_path.read_bytes()).decode()

completion = client.chat.completions.create(
    model="gpt-audio-mini-2025-10-06",
    messages=[
        {"role": "system", "content": "Transcribe the audio exactly as spoken, word for word. Output only the transcription, nothing else."},
        {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": b64_audio, "format": "mp3"}}]},
    ]
)
print(completion.choices[0].message)
