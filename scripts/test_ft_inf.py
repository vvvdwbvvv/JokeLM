import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import ast

# Load the base model and tokenizer
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Load the fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "./output/qwen3-pun-lora/checkpoint-120")
model.eval()


def format_chat_prompt(question):
    """Format the input as a chat message like in training data"""
    messages = [
        {"role": "user", "content": question}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def test_pun_explanation(question):
    # Format the input
    input_text = format_chat_prompt(question)

    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

    # Move to device
    if torch.backends.mps.is_available():
        inputs = inputs.to("mps")

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant's response
    if "<|im_start|>assistant" in full_response:
        response = full_response.split("<|im_start|>assistant")[1].strip()
        if response.startswith("\n"):
            response = response[1:]
    else:
        response = full_response

    print(f"問題: {question}")
    print(f"回答: {response}")
    print("-" * 80)


# Test with new Chinese puns/wordplay
test_questions = [
    "為什麼 \"很蝦\" 很好笑？",
    "為什麼 \"拿鐵\" 很好笑？"
]

print("測試微調後的中文諧音梗模型")
print("=" * 80)

for question in test_questions:
    test_pun_explanation(question)