from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

model_name = "Qwen/Qwen1.5-4B-Chat"
dataset_path = "./data/finetune.jsonl"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Qwen 無 pad_token 時需設為 eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

dataset = load_dataset("json", data_files=dataset_path, split="train")

training_args = TrainingArguments(
    output_dir="./output/qwen3-pun-lora",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-5,
    fp16=True,
    logging_dir="./logs",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="messages",
    args=training_args
)

trainer.train()
