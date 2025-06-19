#!/usr/bin/env python3
# save as post_train_pipeline.py
import argparse, os, json, torch, getpass
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

<<<<<<< HEAD

'''
python post_train_pipeline.py \
  --base_model Qwen/Qwen3-4B \
  --lora_dir ./output/qwen3-pun-lora \
  --merged_dir ./qwen3-pun-merged

# ❷ 加上 4-bit 量化
python post_train_pipeline.py \
  --base_model Qwen/Qwen3-4B \
  --lora_dir ./output/qwen3-pun-lora \
  --quant \
  --quant_dir ./qwen3-pun-merged-4bit

# ❸ 直接推上 Hugging Face（需先設定 HF_TOKEN）
export HF_TOKEN=hf_xxx
python post_train_pipeline.py \
  --base_model Qwen/Qwen3-4B \
  --lora_dir ./output/qwen3-pun-lora \
  --merged_dir ./qwen3-pun-merged \
  --push \
  --repo_id your_username/qwen3-4b-pun-tw
'''


=======
# ---------- CLI 參數 ----------
>>>>>>> origin/main
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True,
                   help="原始（未微調）Model ID 或本地路徑")
    p.add_argument("--lora_dir",  required=True,
                   help="LoRA checkpoint 目錄 (trainer.save_model 的結果)")
    p.add_argument("--merged_dir", default="./merged_model",
                   help="輸出：合併後模型儲存路徑")
    p.add_argument("--quant_dir",  default="./merged_model_4bit",
                   help="輸出：4-bit 量化模型儲存路徑")
    p.add_argument("--quant", action="store_true",
                   help="啟用 4-bit bnb 量化")
    p.add_argument("--push",  action="store_true",
                   help="推送至 HuggingFace Hub")
    p.add_argument("--repo_id",
                   help="Hub Repository，例如 username/qwen3-pun-4b")
    p.add_argument("--sample_file",
                   help="JSON lines，每行一個 prompt；若無則用內建樣例")
    p.add_argument("--max_new_tokens", type=int, default=128)
    return p.parse_args()

<<<<<<< HEAD
=======
# ---------- 工具函數 ----------
>>>>>>> origin/main
def load_prompts(file_path:str|None):
    if file_path and Path(file_path).is_file():
        return [json.loads(l)["prompt"] if l.strip().startswith("{") else l.strip()
                for l in Path(file_path).read_text().splitlines() if l.strip()]
<<<<<<< HEAD
=======
    # 預設測試集
>>>>>>> origin/main
    return [
        "說一個跟貓有關的笑話",
        "說一個跟綠豆有關的笑話。",
        "請講一則跟「工程師」相關的冷笑話。"
    ]

def quick_test(model, tok, prompts, device, max_new):
    model.eval()
    for i, prompt in enumerate(prompts, 1):
        print(f"\n=== [樣例 {i}] {prompt}")
        inp = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp,
                                 max_new_tokens=max_new,
                                 temperature=0.8,
                                 top_p=0.9)
        print(tok.decode(out[0], skip_special_tokens=True))

def bnb_quantize(src_dir:str, tgt_dir:str):
    cfg = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        src_dir,
        quantization_config=cfg,
        device_map="auto"
    )
    model.save_pretrained(tgt_dir)
    print(f"✅ 4-bit 量化模型已存至 {tgt_dir}")

# ---------- 主流程 ----------
def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🚚 1. 讀取 base model 與 LoRA...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map="auto")
    tok  = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.lora_dir)
    model.to(device)

    print("🔗 2. 合併 LoRA 權重 (merge_and_unload)...")
    merged = model.merge_and_unload()
    Path(args.merged_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.merged_dir)
    tok.save_pretrained(args.merged_dir)
    print(f"✅ 合併後模型存至 {args.merged_dir}")

    print("🧪 3. 快速推理測試")
    prompts = load_prompts(args.sample_file)
    quick_test(merged, tok, prompts, device, args.max_new_tokens)

    if args.quant:
        print("🔧 4. 進行 4-bit bnb 量化")
        bnb_quantize(args.merged_dir, args.quant_dir)

    if args.push:
        if not args.repo_id:
            raise ValueError("--push 需要指定 --repo_id")
        from huggingface_hub import login
        token = os.getenv("HF_TOKEN") or getpass.getpass("HF token ≫ ")
        login(token=token)
        print(f"☁️ 5. 上傳至 Hub：{args.repo_id}")
        merged.push_to_hub(args.repo_id)
        tok.push_to_hub(args.repo_id)
        if args.quant:
            from huggingface_hub import upload_folder
            upload_folder(repo_id=args.repo_id,
                          folder_path=args.quant_dir,
                          commit_message="Add 4-bit quantized version",
                          path_in_repo="4bit")
        print("✅ 推送完成")

if __name__ == "__main__":
    main()
