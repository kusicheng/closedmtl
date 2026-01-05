import requests
import json

normal_sys=""
new_sys=""

def translation_request(formatted_text, type="normal", target_language="en", key=None):
    response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",

    headers={
        "Authorization": f"Bearer {key if key != None else "apikeyhere"}",
        "Content-Type": "application/json",
        "HTTP-Referer": "1",
        "X-Title": "1",
    },

    data=json.dumps({
        "model": "google/gemini-2.5-flash-lite",
        "messages": [
        {
            "role": "user",
            "content": [
            
            ]
        }
        ]
    })
    )