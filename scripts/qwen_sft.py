from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, TaskType, get_peft_model
import ast
import torch
from datasets import Dataset
from trl import SFTConfig

model_name = "Qwen/Qwen3-4B"
dataset_path = "./generated_post_finetune_jokes.json"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Qwen 無 pad_token 時需設為 eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)
model = get_peft_model(model, peft_config)

def format_dataset(example):
    user_prompt = (
        f"### Topic:\n{example['topic']}\n\n"
        f"### Context:\n{example['context']}\n\n"
        f"### Instruction:\n{example['instruction']}"
    )

    messages = [{"role": "user", "content": user_prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True 
    )
    return {"text": formatted_prompt}

dataset = load_dataset("json", data_files=dataset_path, split="train")
dataset = dataset.map(format_dataset, remove_columns=list(dataset.features))

training_args = TrainingArguments(
    output_dir="./output/qwen3-joke-rewriter-lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=1e-4,
    max_grad_norm=1.0,
    fp16=True,
    logging_dir="./logs",
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    args=training_args,
)

trainer.train()

print("Training complete. Saving the final model.")
trainer.save_model(training_args.output_dir)
print(f"Model saved to {training_args.output_dir}")
