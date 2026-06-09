import base64
from datetime import datetime
from pathlib import Path

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST


# https://developers.openai.com/api/reference/resources/images/methods/edit
# ---
# Request (multipart/form-data, NOT json):
# curl -X POST "https://api.openai.com/v1/images/edits" \
#     -H "Authorization: Bearer $OPENAI_API_KEY" \
#     -F "model=gpt-image-1" \
#     -F "image=@logo.png" \
#     -F "prompt=Add magical sparkles and glowing aura around the logo"
# Response:
# {
#   "created": 1699900000,
#   "data": [
#     {
#       "b64_json": "Qt0n6ArYAEABGOhEoYgVAJFdt8jM79uW2DO..."
#     }
#   ]
# }

headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
logo_path = Path(__file__).parent / "logo.png"

with logo_path.open("rb") as f:
    response = requests.post(
        OPENAI_HOST + "/v1/images/edits",
        headers=headers,
        files={"image": (logo_path.name, f, "image/png")},
        data={"model": "gpt-image-1.5", "prompt": "Add magical sparkles and glowing aura around the logo"},
    )

if response.status_code != 200:
    raise Exception(f"HTTP {response.status_code}: {response.text}")

b64_data = response.json()["data"][0]["b64_json"]
filename = Path(__file__).parent / f"edited_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
filename.write_bytes(base64.b64decode(b64_data))
print(f"Saved: {filename}")