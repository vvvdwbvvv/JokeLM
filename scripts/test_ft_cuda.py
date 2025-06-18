import torch
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, "./output/qwen3-pun-lora/checkpoint-15")
model = model.to(device)
model.eval()

# 2. Prompt formatter
def format_chat_prompt(question):
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def test_pun_explanation(question):
    prompt = format_chat_prompt(question)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    thread = threading.Thread(
        target=model.generate,
        kwargs={
            **inputs,
            "max_length": tokenizer.model_max_length,
            "temperature": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "streamer": streamer
        },
    )
    thread.start()

    print(f"問題: {question}")
    print("回答: ", end="", flush=True)
    for new_text in streamer:
        print(new_text, end="", flush=True)
    print("\n" + "-" * 160)

if __name__ == "__main__":
    print("測試微調後的中文諧音梗模型")
    questions = [
        '為什麼 "讓台灣人放棄諧音梗已經 Taiwan 且 Tainan 了" 很好笑？',
        '為什麼 "吃我慶記啦" 很好笑？'
    ]
    for q in questions:
        test_pun_explanation(q)
