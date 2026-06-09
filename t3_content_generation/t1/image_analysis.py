import base64
from pathlib import Path

from commons.constants import OPENAI_HOST
from t3_content_generation._openai_client import OpenAIClientT3


# https://developers.openai.com/api/docs/guides/images-vision?format=url&lang=curl
# https://developers.openai.com/api/docs/guides/images-vision?format=base64-encoded

logo_path = Path(__file__).parent / "logo.png"
b64_logo = base64.b64encode(logo_path.read_bytes()).decode()

client = OpenAIClientT3(OPENAI_HOST + "/v1/chat/completions")
client.call(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Generate poem based on images"},
            {"type": "image_url", "image_url": {"url": "https://a-z-animals.com/media/2019/11/Elephant-male-1024x535.jpg"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_logo}"}},
        ]
    }]
)
