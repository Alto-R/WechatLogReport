import requests

url = "http://127.0.0.1:5030/api/v1/chatlog"
params = {
    "time": "2025-01-01~2025-12-31",
    "talker": "xxx",
    "format": "json",
}
r = requests.get(url, params=params)
print("status:", r.status_code)
print("url:", r.url)
with open("../analysis_output/chat.json", "wb") as f:
    f.write(r.content)
