# ArgoLM — Personal AI Assistant SLM
## Product Requirements Document (PRD)

---

## 📋 প্রকল্পের সংক্ষিপ্ত বর্ণনা

**ArgoLM** হলো একটি **Small Language Model (SLM)** যা:
- ✅ বাংলা এবং ইংরেজি দুটো ভাষায় কথা বলে
- ✅ **শুধুমাত্র Conversation** এবং **General QA** করে
- ✅ একজন ব্যক্তিগত AI assistant-এর মতো কাজ করে
- ✅ JARVIS (FastAPI) আর Colab T4-এ train হয়
- ✅ ভবিষ্যতে যেকোনো developer সহজে update করতে পারবে

---

## 🎯 লক্ষ্য (Goals)

| লক্ষ্য | বর্ণনা |
|------|--------|
| **Small Model** | মাত্র 1B parameters (active: 250M) |
| **Bangla-focused** | বাংলা language support priority |
| **Conversation** | প্রশ্ন-উত্তর এবং casual chat |
| **Low Cost** | Colab free GPU-তে train হয় |
| **Modular** | নতুন data add করলে code change না করে train হয় |
| **Easy Maintenance** | যেকোনো developer সহজে বুঝতে পারবে |

---

## 🏗️ Architecture (মডেল ডিজাইন)

### Model Specifications

```
নাম:                  ArgoLM
Parameters:           1 Billion (1B)
Active Parameters:    250-300M (per token)
Architecture:         MoE Transformer + MLA Attention + MTP

Configuration Details:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
vocab_size          : 16,000      (Bangla + English vocab)
n_embd              : 1,024       (Embedding dimension)
n_head              : 16          (Attention heads)
n_layer             : 24          (Transformer layers)
block_size          : 1,024       (Context window)
n_experts           : 8           (MoE experts)
top_k               : 2           (Active experts per token)
hidden_dim          : 4,096       (FFN hidden dimension)
dropout             : 0.1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VRAM Usage:
Training:           ~8 GB  (Colab T4)
Inference:          ~2 GB
Quantized (int4):   ~500 MB
```

### Architecture কম্পোনেন্ট

```
ArgoLM Structure:

┌──────────────────────────────────┐
│    Input Text (Bangla/English)   │
└────────────────┬─────────────────┘
                 ↓
        ┌────────────────┐
        │  Tokenizer     │
        │ (16K vocab)    │
        └────────┬───────┘
                 ↓
      ┌─────────────────────┐
      │ Token Embedding     │
      │ (→ 1024-dim vector) │
      └──────────┬──────────┘
                 ↓
     ┌──────────────────────────┐
     │ Transformer Block × 24   │
     │                          │
     │  ┌──────────────────────┐│
     │  │ MLA Attention        ││
     │  │ (16 heads, low-rank) ││
     │  └──────────────────────┘│
     │             ↓            │
     │  ┌──────────────────────┐│
     │  │ MoE FFN              ││
     │  │ (8 experts, 2 active)││
     │  └──────────────────────┘│
     │             ↓            │
     │       (× 24 times)       │
     └──────────────┬───────────┘
                    ↓
         ┌──────────────────────┐
         │  Output Layer        │
         │ (Next word probab.)  │
         └──────────┬───────────┘
                    ↓
          ┌─────────────────────┐
          │ Next Word Selection │
          │ (Bangla/English)    │
          └─────────────────────┘
```

---

## 📊 Training Pipeline (প্রশিক্ষণ প্রক্রিয়া)

### সম্পূর্ণ Training Steps

```
PHASE 1 — Pretrain (ভাষা শেখা)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

লক্ষ্য:      Model-কে ভাষা বুঝতে শেখানো
ডেটা:       Wikipedia + CC-100 + TinyStories
সময়:       3-4 দিন
GPU:        Colab T4
ফলাফল:     Model জানবে বাংলা-ইংরেজি কী জিনিস

Data Mix:
  • Bangla Wikipedia      : 40%
  • CC-100 Bangla         : 20%
  • TinyStories English   : 20%
  • Bangla Sangraha       : 20%

Output: checkpoint_pretrain_final.pt


PHASE 2 — SFT (Conversation শেখা)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

লক্ষ্য:      Model-কে কথা বলতে শেখানো
ডেটা:       নিজের লেখা Q&A data
সময়:       1 দিন
GPU:        Colab T4
ফলাফল:     Model helpful conversation করতে পারবে

Data Required:
  • Expert 1 (Bangla)        : 50 Q&A pairs
  • Expert 2 (General QA)    : 50 Q&A pairs
  • Expert 6 (Conversation)  : 100 Q&A pairs ⭐
  • Expert 7 (Instruction)   : 50 Q&A pairs

Format: JSONL file
{
  "instruction": "প্রশ্ন এখানে",
  "response": "উত্তর এখানে"
}

Output: checkpoint_sft_final.pt


PHASE 3 — GRPO (Reasoning শেখা) - Optional
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

লক্ষ্য:      Model-কে চিন্তা করতে শেখানো
ডেটা:       Math + Logic problems
সময়:       1-2 দিন
GPU:        Kaggle (free)
ফলাফল:     Model reasoning করতে পারবে

Data Required:
  • Math problems          : 100-200
  • Logic questions        : 50-100

Format: JSONL file
{
  "question": "গণিত প্রশ্ন",
  "answer": "সঠিক উত্তর",
  "solution": "সমাধান পদ্ধতি"
}

Output: checkpoint_grpo_final.pt
```

---

## 📁 ডেটা স্ট্রাকচার (Data Structure)

### Google Drive-এ Folder Layout

```
📦 ArgoLM_Training/
│
├── 📁 /data/                          ← এখানে সব input data থাকবে
│   │
│   ├── 📁 /pretrain/                  ← Pretrain ডেটা
│   │   ├── 📄 bangla_wiki.txt         (Wikipedia Bangla - clean)
│   │   ├── 📄 cc100_bangla.txt        (CC-100 Bangla - clean)
│   │   ├── 📄 tinystories.txt         (TinyStories - clean)
│   │   └── 📄 sangraha_bangla.txt     (Bangla Sangraha - clean)
│   │
│   ├── 📁 /sft/                       ← SFT ডেটা (Q&A pairs)
│   │   ├── 📄 expert1_bangla.jsonl
│   │   ├── 📄 expert2_qa.jsonl
│   │   ├── 📄 expert6_conversation.jsonl  ⭐ সবচেয়ে গুরুত্বপূর্ণ
│   │   └── 📄 expert7_instruction.jsonl
│   │
│   └── 📁 /grpo/                      ← GRPO ডেটা (reasoning)
│       ├── 📄 math_problems.jsonl
│       └── 📄 logic_questions.jsonl
│
├── 📁 /checkpoints/                   ← Trained models save হবে
│   ├── 📄 checkpoint_pretrain_final.pt
│   ├── 📄 checkpoint_sft_final.pt
│   └── 📄 checkpoint_grpo_final.pt
│
├── 📁 /code/                          ← Training code
│   ├── 📄 config.py                   (Configuration)
│   ├── 📄 model.py                    (Model architecture)
│   ├── 📄 train.py                    (Main training script)
│   ├── 📄 data_loader.py              (Data loading)
│   └── 📄 utils.py                    (Helper functions)
│
└── 📄 PRD.md                          (এই ফাইল)
```

---

## 📥 ডেটা কী, কখন এবং কোথা থেকে আনবে?

### Phase 1 — Pretrain Data

| নাম | উৎস | লিংক | সাইজ | কেন দরকার |
|-----|------|------|------|-----------|
| **Bangla Wikipedia** | Hugging Face | `wikimedia/wikipedia` bn | 2-3GB | বাংলা ভাষা শেখায় |
| **CC-100 Bangla** | Hugging Face | `cc_news` (Bangla) | 1-2GB | Bangla diversity |
| **TinyStories** | Hugging Face | `roneneldan/TinyStories` | 1-2GB | Conversation style |
| **Bangla Sangraha** | Hugging Face | `ai4bharat/sangraha` bn | 500MB | Bangla text corpus |

### Phase 2 — SFT Data

**সম্পূর্ণভাবে নিজে লিখতে হবে JSONL format-এ**

```
Required Q&A pairs: 250 total

Expert 1 (Bangla)           : 50 pairs
Expert 2 (General QA)       : 50 pairs
Expert 6 (Conversation) ⭐  : 100 pairs (সবচেয়ে গুরুত্বপূর্ণ)
Expert 7 (Instruction)      : 50 pairs

Example Format:
{
  "instruction": "তুমি কেমন আছো?",
  "response": "আমি ভালো আছি! তোমাকে কীভাবে সাহায্য করতে পারি?"
}
```

### Phase 3 — GRPO Data (Optional)

**নিজে লিখতে হবে Math + Logic problems-এ**

```
Required: 150-300 problems

Math problems       : 100-200
Logic questions     : 50-100

Example Format:
{
  "question": "৫ জন মিলে একটা কাজ ১২ দিনে করে।
               ৪ জনে কতদিন লাগবে?",
  "answer": "১৫ দিন",
  "solution": "মোট কাজ = ৫ × ১২ = ৬০ unit
              ৪ জনে = ৬০ ÷ ৪ = ১৫ দিন"
}
```

---

## 🧹 ডেটা ক্লিনিং (Data Cleaning)

### কীভাবে Data Clean করবে?

#### Pretrain Data Cleaning

```python
# Wikipedia থেকে ডেটা clean করার process

import re

def clean_wiki_text(text):
    """Wikipedia markup এবং HTML বাদ দিয়ে clean text রিটার্ন করে"""
    
    # Step 1: Wiki markup বাদ দাও
    text = re.sub(r'\[\[.*?\]\]', '', text)      # [[links]]
    text = re.sub(r'\{\{.*?\}\}', '', text)      # {{templates}}
    text = re.sub(r'<ref>.*?</ref>', '', text)   # <ref></ref>
    
    # Step 2: HTML tags বাদ দাও
    text = re.sub(r'<[^>]+>', '', text)
    
    # Step 3: Bangla/English ছাড়া অন্য character বাদ দাও
    text = re.sub(r'[^\u0980-\u09FF\u0020-\u007Ea-zA-Z0-9।,;:?!\s]', '', text)
    
    # Step 4: Extra whitespace পরিষ্কার করো
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 5: অনেক ছোট line বাদ দাও (< 50 characters)
    lines = [l for l in text.split('\n') if len(l) > 50]
    
    return '\n'.join(lines)

# কীভাবে use করবে:
raw_text = "আপনার raw Wikipedia text এখানে"
clean_text = clean_wiki_text(raw_text)
print(clean_text)
```

#### SFT Data Format

```
নিজে লিখার সময় এই format অনুসরণ করো:

{
  "instruction": "স্পষ্ট প্রশ্ন বা instruction",
  "response": "Helpful এবং বিস্তারিত উত্তর"
}

✅ ভালো উদাহরণ:
{
  "instruction": "বাংলাদেশের প্রধানমন্ত্রী কে?",
  "response": "আমি বর্তমান সময় পর্যন্ত তথ্য জানি না। আপনি 
             সরকারি ওয়েবসাইট চেক করতে পারেন।"
}

❌ খারাপ উদাহরণ:
{
  "instruction": "কি",
  "response": "না জানি"
}

Rules:
• Instruction হবে স্পষ্ট এবং গঠনসম্মত
• Response হবে helpful এবং বিস্তারিত
• দুটোই Bangla বা English হতে পারে
• Response কমপক্ষে 20 word হওয়া উচিত
```

---

## 💻 Colab Setup (প্রথম বার Setup)

### Step 1 — Google Drive Mount করো

```python
# Colab cell-এ এই code চালাও

from google.colab import drive
drive.mount('/content/drive')

# Output:
# Mounted at /content/drive
```

### Step 2 — Repository Clone করো

```python
import os
os.chdir('/content/drive/MyDrive')

# তোমার ArgoLM project folder-এ যাও
os.chdir('ArgoLM_Training')

print("✅ Setup complete!")
print("আপনার /data folder এ এই files থাকতে হবে:")
print("  /data/pretrain/")
print("  /data/sft/")
print("  /data/grpo/")
```

### Step 3 — Dependencies Install করো

```python
!pip install torch transformers datasets huggingface_hub -q
!pip install bitsandbytes -q
!pip install peft -q

print("✅ সব dependencies install হয়েছে!")
```

---

## 🚀 Training কীভাবে চালাবে?

### Phase 1 — Pretrain (প্রথম বার)

```python
# train.py চালাও Colab-এ

# Cell এ type করো:
%run /content/drive/MyDrive/ArgoLM_Training/code/train.py \
  --phase pretrain \
  --data_dir /content/drive/MyDrive/ArgoLM_Training/data/pretrain \
  --output_dir /content/drive/MyDrive/ArgoLM_Training/checkpoints \
  --epochs 2 \
  --batch_size 8 \
  --save_steps 500

# কী ঘটছে:
# ✅ /data/pretrain/ থেকে সব .txt files পড়ছে
# ✅ Model train হচ্ছে
# ✅ প্রতি 500 steps-এ checkpoint save হচ্ছে
# ✅ Drive-এ /checkpoints/ folder-এ save হচ্ছে
```

### Phase 2 — SFT (Pretrain এর পরে)

```python
# train.py চালাও এই arguments দিয়ে

%run /content/drive/MyDrive/ArgoLM_Training/code/train.py \
  --phase sft \
  --data_dir /content/drive/MyDrive/ArgoLM_Training/data/sft \
  --pretrain_checkpoint /content/drive/MyDrive/ArgoLM_Training/checkpoints/checkpoint_pretrain_final.pt \
  --output_dir /content/drive/MyDrive/ArgoLM_Training/checkpoints \
  --epochs 3 \
  --batch_size 8

# কী ঘটছে:
# ✅ /data/sft/ থেকে সব .jsonl files পড়ছে
# ✅ Pretrain checkpoint load হচ্ছে
# ✅ SFT training চলছে
# ✅ checkpoint_sft_final.pt save হচ্ছে
```

### Phase 3 — GRPO (Optional)

```python
%run /content/drive/MyDrive/ArgoLM_Training/code/train.py \
  --phase grpo \
  --data_dir /content/drive/MyDrive/ArgoLM_Training/data/grpo \
  --sft_checkpoint /content/drive/MyDrive/ArgoLM_Training/checkpoints/checkpoint_sft_final.pt \
  --output_dir /content/drive/MyDrive/ArgoLM_Training/checkpoints \
  --grpo_steps 5000

# কী ঘটছে:
# ✅ /data/grpo/ থেকে math_problems.jsonl পড়ছে
# ✅ GRPO training চলছে
# ✅ checkpoint_grpo_final.pt save হচ্ছে
```

---

## 📝 কোড Structure (যা ফিউচার Developer বুঝবে)

### train.py (Main Training Script)

```python
# এই file-এ main training logic থাকবে

# Structure:
"""
train.py
├── Load config from config.py
├── Initialize model from model.py
├── Load data from data_loader.py
│   ├── Automatically reads all files from /data/pretrain/
│   ├── Automatically reads all files from /data/sft/
│   └── Automatically reads all files from /data/grpo/
├── Setup training arguments
├── Start training
└── Save checkpoints to /checkpoints/
"""

# কীভাবে কাজ করে:
# 1. /data/ folder থেকে সব files automatically পড়ে
# 2. Format অনুযায়ী process করে
# 3. কোনো hardcoded filename থাকে না
# 4. নতুন file add করলেই তা training-এ include হয়
```

### data_loader.py (Automatic Data Loading)

```python
# এই file-এ data loading logic থাকবে

def load_pretrain_data(data_dir):
    """
    /data/pretrain/ থেকে সব .txt files পড়ে
    কোনো filename hardcoded নেই
    """
    all_texts = []
    for file in os.listdir(data_dir):
        if file.endswith('.txt'):
            with open(os.path.join(data_dir, file)) as f:
                all_texts.append(f.read())
    return all_texts

def load_sft_data(data_dir):
    """
    /data/sft/ থেকে সব .jsonl files পড়ে
    কোনো filename hardcoded নেই
    """
    all_qa = []
    for file in os.listdir(data_dir):
        if file.endswith('.jsonl'):
            with open(os.path.join(data_dir, file)) as f:
                for line in f:
                    all_qa.append(json.loads(line))
    return all_qa
```

---

## 🔄 Future-এ নতুন Data কীভাবে যোগ করবে?

### নিয়ম (খুবই সহজ)

```
নতুন data add করার step:

1. নতুন file তৈরি করো সঠিক format-এ
   Pretrain:  /data/pretrain/নাম.txt
   SFT:       /data/sft/নাম.jsonl
   GRPO:      /data/grpo/নাম.jsonl

2. Drive-এ upload করো

3. Training code চালাও (same command)
   code automatically নতুন file pick করবে

4. Done! কোনো code change লাগবে না ✅
```

### Example — নতুন SFT Data যোগ করা

```
Old structure:
/data/sft/
├── expert1_bangla.jsonl
├── expert2_qa.jsonl
└── expert6_conversation.jsonl

নতুন file যোগ করো:
/data/sft/
├── expert1_bangla.jsonl
├── expert2_qa.jsonl
├── expert6_conversation.jsonl
└── expert5_creative.jsonl  ← নতুন

Training command (একই থাকবে):
%run train.py --phase sft --data_dir /data/sft/

Result:
Code automatically expert5_creative.jsonl-ও পড়বে
কোনো hardcoding change লাগবে না! ✅
```

---

## 🛠️ Developer Guidelines (ভবিষ্যতের Developer-দের জন্য)

### কোড লেখার নিয়ম

```
1. Modular Code লেখো
   ❌ হার্ডকোডেড filenames
   ✅ Dynamic file discovery (os.listdir)

2. Clear Comments লেখো
   Bengali comment থাকবে
   English comment থাকবে
   কোডের প্রতিটি section explain করবে

3. Error Handling
   File না পেলে clear error message
   Data format ভুল হলে বলবে কোথায় ভুল

4. Logging
   প্রতিটি step-এ print করো কী হচ্ছে
   Progress bar থাকবে training-এ

5. Configuration
   সব settings config.py-তে থাকবে
   train.py-তে hardcoded value থাকবে না
```

### Code Template (আগামী Developer ব্যবহার করবে)

```python
# train.py template

import os
import json
import torch
from pathlib import Path
from datetime import datetime

class ArgoLMTrainer:
    def __init__(self, config_path):
        """Initialize trainer with config"""
        self.config = self.load_config(config_path)
        self.setup_logging()
        
    def load_config(self, config_path):
        """Load configuration from config.py"""
        with open(config_path) as f:
            return json.load(f)
    
    def load_data_automatically(self, data_dir, file_type='.txt'):
        """
        কোনো filename hardcoding ছাড়াই
        data_dir থেকে সব file পড়ে
        """
        print(f"📂 Loading data from {data_dir}...")
        all_data = []
        
        for file in Path(data_dir).iterdir():
            if file.suffix == file_type:
                print(f"  ✓ Loading {file.name}...")
                with open(file) as f:
                    if file_type == '.jsonl':
                        all_data.extend([json.loads(line) for line in f])
                    else:
                        all_data.append(f.read())
        
        print(f"✅ Total files loaded: {len(all_data)}")
        return all_data
    
    def train(self, phase):
        """Train model for given phase"""
        print(f"🚀 Starting {phase} training...")
        
        if phase == 'pretrain':
            data = self.load_data_automatically(
                self.config['pretrain_dir'], '.txt'
            )
        elif phase == 'sft':
            data = self.load_data_automatically(
                self.config['sft_dir'], '.jsonl'
            )
        
        # Training logic here...
        
        self.save_checkpoint(phase)
    
    def save_checkpoint(self, phase):
        """Save model checkpoint"""
        save_path = os.path.join(
            self.config['checkpoint_dir'],
            f'checkpoint_{phase}_{datetime.now().strftime("%Y%m%d")}.pt'
        )
        print(f"💾 Saving checkpoint to {save_path}...")
        torch.save(self.model.state_dict(), save_path)

if __name__ == '__main__':
    trainer = ArgoLMTrainer('config.py')
    trainer.train('pretrain')
```

---

## 📊 Training Status & Metrics (কীভাবে progress দেখবে)

### Each Phase-এ কী দেখবে

```
PHASE 1 — Pretrain
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch 1/2 | Step 500/5000 | Loss: 4.23 | LR: 3e-4
Epoch 1/2 | Step 1000/5000 | Loss: 3.89 | LR: 3e-4
Epoch 1/2 | Step 1500/5000 | Loss: 3.45 | LR: 3e-4
...
✅ Checkpoint saved: checkpoint_pretrain_final.pt


PHASE 2 — SFT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loading data from /data/sft/...
  ✓ Loading expert1_bangla.jsonl... (50 pairs)
  ✓ Loading expert2_qa.jsonl... (50 pairs)
  ✓ Loading expert6_conversation.jsonl... (100 pairs)
  ✓ Loading expert7_instruction.jsonl... (50 pairs)
✅ Total files loaded: 250 pairs

Epoch 1/3 | Step 100/300 | Loss: 2.34 | LR: 2e-4
Epoch 1/3 | Step 200/300 | Loss: 2.10 | LR: 2e-4
...
✅ Checkpoint saved: checkpoint_sft_final.pt
```

---

## ⚠️ সাধারণ সমস্যা এবং সমাধান

| সমস্যা | কারণ | সমাধান |
|--------|------|--------|
| **"File not found"** | /data/ folder path ভুল | Path check করো: `/content/drive/MyDrive/ArgoLM_Training/data/` |
| **CUDA out of memory** | Batch size অনেক বড় | batch_size কমাও (8 থেকে 4) |
| **JSON decode error** | JSONL file format ভুল | প্রতিটি line একটা valid JSON হওয়া উচিত |
| **Colab session crash** | Training অনেক দীর্ঘ | Pretrain 2 epoch দিয়ে রাখো |

---

## 📞 Developer যোগাযোগ

Developer যখন প্রশ্ন করবে:

```
নিম্নলিখিত তথ্য জিজ্ঞেস করো:
1. কোন Phase train করছ? (pretrain/sft/grpo)
2. Error message কী?
3. /data/ folder-এ কোন files আছে?
4. Colab session কি crash হয়েছে?
```

---

## 📚 Reference & Resources

### Papers & Links
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Hugging Face Datasets: https://huggingface.co/datasets
- PyTorch Docs: https://pytorch.org/docs

---

## 🔄 ভবিষ্যতের আপডেট

```
Next Versions Planned:

v2.0 (আগামী ৩ মাস):
  □ Vision support (images বোঝা)
  □ Tool calling ability
  □ Extended context window (2048 tokens)
  □ Fine-tuned Bengali model

v3.0 (আগামী ৬ মাস):
  □ Multimodal capabilities
  □ Real-time web search
  □ Memory/long-term context
```

---

## ✅ Checklist (শুরু করার আগে)

```
Start করার আগে এই সব check করো:

□ Google Drive এ ArgoLM_Training folder আছে
□ /data/ subfolder আছে (pretrain, sft, grpo)
□ Pretrain data download করা হয়েছে
□ Code files সব upload করা হয়েছে
□ Colab notebook প্রস্তুত আছে
□ এই PRD.md বুঝেছ

তাহলে শুরু করার জন্য প্রস্তুত! 🚀
```

---

## 📄 Document Information

```
নাম:        ArgoLM PRD (Product Requirements Document)
সংস্করণ:   1.0
তৈরির তারিখ: 2026-04-23
ভাষা:      Bengali (বাংলা)
ডেভেলপার:  যে কেউ (Modular & Flexible)

শেষ আপডেট: এই ফাইল সর্বদা update থাকবে
যখন প্রয়োজন তখনই নতুন section যোগ হবে
```

---

**আপনার নিজের AI Assistant তৈরির যাত্রা শুরু করুন! 🚀**