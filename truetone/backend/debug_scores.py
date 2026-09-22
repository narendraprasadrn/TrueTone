import requests
import json
with open("../sample_audio/genuine_sample.wav", 'rb') as f:
    res = requests.post("http://localhost:5000/test/analyze-audio", files={"file": f}, data={"enrolled_identity_id": "user_demo"})
print(json.dumps(res.json(), indent=2))
