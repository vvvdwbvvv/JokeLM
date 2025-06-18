import os
from anthropic import Anthropic
from prompts import answer_query_level_three

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def build_prompt(query: str) -> str:
    ctx = {"vars": {"query": query}}
    return answer_query_level_three(ctx)    

def call_claude(prompt: str,
                model: str = "claude-3-haiku-20240307",
                max_tokens: int = 256,
                temperature: float = 0.7) -> str:
    """
    將 prompt 丟給 Claude，並回傳第一段文字結果
    """
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    # Claude v3 API 回傳 content 為 list[Message.ContentBlock]
    return response.content[0].text.strip()

# === 4. 封裝成主流程 ===
def generate_joke(query: str) -> str:
    """
    使用者輸入主題 query，回傳 Claude 生成的單段中文笑話
    """
    prompt = build_prompt(query)
    joke   = call_claude(prompt)
    return joke


# === 5. CLI 測試 ===
if __name__ == "__main__":
    while True:
        try:
            q = input("\n請輸入笑話主題（或輸入 quit 離開）： ").strip()
            if q.lower() in {"quit", "exit"}:
                break
            print("\n--- Claude 生成的笑話 ---")
            print(generate_joke(q))
        except KeyboardInterrupt:
            break
