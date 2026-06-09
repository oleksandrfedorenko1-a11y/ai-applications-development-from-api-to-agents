import base64
from datetime import datetime
from pathlib import Path

from commons.constants import OPENAI_HOST
from t3_content_generation._openai_client import OpenAIClientT3


# https://developers.openai.com/api/reference/resources/images/methods/generate
# ---
# Request:
# curl -X POST "https://api.openai.com/v1/images/generations" \
#     -H "Authorization: Bearer $OPENAI_API_KEY" \
#     -H "Content-type: application/json" \
#     -d '{
#         "model": "gpt-image-2",
#         "prompt": "smiling catdog."
#     }'
# Response:
# {
#   "created": 1699900000,
#   "data": [
#     {
#       "b64_json": Qt0n6ArYAEABGOhEoYgVAJFdt8jM79uW2DO...,
#     }
#   ]
# }

client = OpenAIClientT3(OPENAI_HOST + "/v1/images/generations")
response = client.call(model="gpt-image-1.5", prompt="Smiling catdog")

b64_data = response["data"][0]["b64_json"]
filename = Path(__file__).parent / f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
filename.write_bytes(base64.b64decode(b64_data))
print(f"Saved: {filename}")
