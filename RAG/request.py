import requests
import dotenv
import os
import json

dotenv.load_dotenv()

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT")


class LiteLMClient:
    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint

    def get_ollama_embedding(self, text: str, model: str) -> list:
        url = f"{self.endpoint}/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "input": text
        }
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # 假設回傳格式與 OpenAI 類似：{"data": [{"embedding": [...]}], ...}
        return data["data"][0]["embedding"]
    
    def get_ollama_message(self, messages: str, model: str) -> str:
        url = f"{self.endpoint}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": messages}],
            "stream": False
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload))
        resp.raise_for_status()
    
        data = resp.json()
        return data['choices'][0]['message']['content']
        resp.raise_for_status()
        data = resp.json()
        print(data)
