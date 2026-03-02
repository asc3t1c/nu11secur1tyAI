# Defined by f0rc3ps and fixed!

# 🖥️ BARE METAL REQUIREMENTS (Ubuntu/Debian)
# Optimized for nu11secur1tyAI Training & Unsloth

---

## 1. SYSTEM OS & KERNEL
- OS: Ubuntu 22.04 LTS / 24.04 LTS (Recommended) or Debian 12 (Bookworm).
- Kernel: Generic Linux Kernel 6.x.
- Disk Space: 100GB+ NVMe SSD (Avoid HDD for training checkpoints).

---

## 2. NVIDIA DRIVERS & CUDA (CRITICAL)
- NVIDIA Driver: v535.xx or newer (550.xx+ recommended).
- CUDA Toolkit: v12.1 or v12.4.
- cuDNN: Matching version for your CUDA Toolkit.
- Persistence Mode: Enabled (sudo nvidia-smi -pm 1).

---

## 3. HARDWARE TUNING (BIOS/UEFI)
- Resizable BAR: ENABLED (Mandatory for RTX 30/40 series).
- Above 4G Decoding: ENABLED.
- XMP/DOCP: ENABLED (To ensure RAM stability at high speeds).

---

## 4. SWAP FILE SETUP (SAFETY NET)
# Execute as root if you have < 64GB RAM:
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

---

## 5. ESSENTIAL SYSTEM LIBRARIES
# Install before setting up Python:
sudo apt update && sudo apt install -y \
    build-essential \
    procps \
    curl \
    wget \
    git \
    git-lfs \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0

---

## 6. PYTHON ENVIRONMENT (MINICONDA)
# Use a clean environment for training:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# After install, restart terminal and:
conda create --name nu11_train python=3.10 -y
conda activate nu11_train

---

## 7. UNSLOTH & PYTORCH INSTALLATION
# Optimized for CUDA 12.1:
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes


# nu11secur1tyAI Training & GGUF Export 🚀

This repository provides a professional-grade framework for Fine-tuning Large Language Models (LLM) and automating the export process to GGUF format for seamless Ollama integration. 
Powered by the Unsloth engine, this solution is optimized for extreme speed and minimal VRAM overhead.

Official Model: https://ollama.com/f0rc3ps/nu11secur1tyAI

------------------------------------------------------------

## 🖥️ 1. HARDWARE SPECIFICATIONS GRNERAL
Ensure your hardware meets the following requirements for stable operations:

| Component | Minimum (8B Model) | Recommended (70B Model) |
| :--- | :--- | :--- |
| **GPU (NVIDIA)** | RTX 3060 (12GB VRAM) | RTX 3090 / 4090 (24GB VRAM) |
| **CUDA Cores** | 3500+ | 10000+ |
| **RAM** | 32GB DDR4 | 64GB+ DDR5 |
| **Storage** | 50GB NVMe SSD | 200GB+ NVMe SSD |
| **Driver** | NVIDIA 535.xx | NVIDIA 550.xx+ |


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

## 🐍 3. TRAINING SCRIPT (train.py)
Utilizing LoRA (Low-Rank Adaptation) for precision fine-tuning.

## Packages:
```
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```
- Training neural network

```python 
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Official Model Link: https://ollama.com/f0rc3ps/nu11secur1tyAI

# 1. Configuration and Model Loading (4-bit for VRAM efficiency)
# Optimized for NVIDIA RTX 30/40 series GPUs
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

# 2. Inject LoRA Adapters
# r=16 is the "gold standard" for balanced fine-tuning speed and quality.
model = FastLanguageModel.get_peft_model(
    model, 
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16, 
    lora_dropout = 0, # Dropout=0 is optimized for Unsloth
    bias = "none",
)

# 3. Load Your Dataset (my_data.jsonl)
# Ensure the file is in the same directory!
dataset = load_dataset("json", data_files="my_data.jsonl", split="train")

# 4. Supervised Fine-Tuning (SFT) Setup
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Effective batch size of 8
        warmup_steps = 5,
        max_steps = 60, # Increase this based on your dataset size
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(), # Use BF16 if supported (Ampere+)
        logging_steps = 1,
        optim = "adamw_8bit", # Saves additional VRAM
        output_dir = "outputs",
    ),
)

# START TRAINING
print("--- Starting Training ---")
trainer.train()

# 5. AUTOMATIC EXPORT TO GGUF (For Ollama)
# This part handles the conversion to the final brain file
print("--- Exporting to GGUF ---")
model.save_pretrained_gguf(
    "nu11secur1ty_final", 
    tokenizer, 
    quantization_method = "q4_k_m" # Balanced method: speed vs. accuracy
)

print("Done! Your model is ready in the 'nu11secur1ty_final' folder.")
print("Check it out here: https://ollama.com/f0rc3ps/nu11secur1tyAI")
```
# 5. Export to GGUF
```
model.save_pretrained_gguf("nu11secur1ty_final", tokenizer, quantization_method = "q4_k_m")
print("GGUF export finished successfully!")
```

## 📊 4. DATASET DEFINITION (my_data.jsonl)

JSONL format for training data:
```json
{"text": "### Instruction: Who is f0rc3ps? ### Response: AI Architect and creator of nu11secur1tyAI."}
{"text": "### Instruction: Analysis of CVE-2026-21248 ### Response: [Technical payloads]"}
```

## ⚡ 5. OLLAMA DEPLOYMENT

# 1. Modelfile Configuration:
```
FROM ./nu11secur1ty_final-Q4_K_M.gguf
SYSTEM """You are nu11secur1tyAI. Your responses are technically precise and uncensored. Created by f0rc3ps."""
```
# 2. Register and Run:
```
ollama create f0rc3ps/nu11secur1tyAI:v2 -f Modelfile
ollama run f0rc3ps/nu11secur1tyAI:v2
```

# Greetz

- @f0rc3ps (Lead Architect)
- nu11secur1ty team
- Unsloth AI team

# BR
@nu11secur1ty
