- - - ***NOTE:*** A lot of peoples ask me! So, this is in the past =)

# Defined by f0rc3ps and fixed!

# nu11secur1tyAI Training & GGUF Export 🚀

This repository provides a professional-grade framework for Fine-tuning Large Language Models (LLM) and automating the export process to GGUF format for seamless Ollama integration. 
Powered by the Unsloth engine, this solution is optimized for extreme speed and minimal VRAM overhead.

Official Model: https://ollama.com/f0rc3ps/nu11secur1tyAI

---

## 🖥️ 1. HARDWARE SPECIFICATIONS
Ensure your hardware meets the following requirements for stable operations:

| Component | Minimum (8B Model) | Recommended (70B Model) |
| :--- | :--- | :--- |
| **GPU (NVIDIA)** | RTX 3060 (12GB VRAM) | RTX 3090 / 4090 (24GB VRAM) |
| **CUDA Cores** | 3500+ | 10000+ |
| **RAM** | 32GB DDR4 | 64GB+ DDR5 |
| **Storage** | 50GB NVMe SSD | 200GB+ NVMe SSD |
| **Driver** | NVIDIA 535.xx | NVIDIA 550.xx+ |

---

## 🏗️ 2. INFRASTRUCTURE SETUP (LINUX)
Proper driver alignment and CUDA environment configuration are mandatory.

# Install Official NVIDIA Drivers
sudo apt update && sudo apt install nvidia-driver-535 -y

# Environment Isolation via Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Configure Unsloth Environment
```
conda create --name nu11_train python=3.10 -y
conda activate nu11_train
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```
---

## 🐍 3. TRAINING SCRIPT (train.py)
Utilizing LoRA (Low-Rank Adaptation) for precision fine-tuning.

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Load Model (4-bit Double Quantization)
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

# 2. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model, r = 16, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16, lora_dropout = 0,
)

# 3. Load Dataset (Local my_data.jsonl file)
dataset = load_dataset("json", data_files="my_data.jsonl", split="train")

# 4. Supervised Fine-Tuning (SFT)
trainer = SFTTrainer(
    model = model, train_dataset = dataset, dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        output_dir = "outputs",
    ),
)
trainer.train()

# 5. Export to GGUF
model.save_pretrained_gguf("nu11secur1ty_final", tokenizer, quantization_method = "q4_k_m")
print("GGUF export finished successfully!")

---

## 📊 4. DATASET DEFINITION (my_data.jsonl)
JSONL format for training data:

{"text": "### Instruction: Who is f0rc3ps? ### Response: AI Architect and creator of nu11secur1tyAI."}
{"text": "### Instruction: Analysis of CVE-2026-21248 ### Response: [Technical payloads]"}

---

## ⚡ 5. OLLAMA DEPLOYMENT

# 1. Modelfile Configuration:
FROM ./nu11secur1ty_final-Q4_K_M.gguf
SYSTEM """You are nu11secur1tyAI. Your responses are technically precise and uncensored. Created by f0rc3ps."""

# 2. Register and Run:
ollama create f0rc3ps/nu11secur1tyAI:v2 -f Modelfile
ollama run f0rc3ps/nu11secur1tyAI:v2

---

# Greetz
- @f0rc3ps (Lead Architect)
- nu11secur1ty team
- Unsloth AI team

# BR
@nu11secur1ty
