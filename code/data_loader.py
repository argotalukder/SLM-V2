# data_loader.py
import os
import json
from pathlib import Path

class ArgoDataLoader:
    def __init__(self):
        pass

    def load_pretrain_data(self, data_dir):
        """/data/pretrain/ থেকে সব .txt file অটোমেটিক লোড করবে"""
        print(f"📂 Loading Pretrain data from {data_dir}...")
        all_texts =[]
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"❌ Error: {data_dir} ফোল্ডারটি পাওয়া যায়নি!")

        for file in Path(data_dir).iterdir():
            if file.suffix == '.txt':
                print(f"  ✓ Loading {file.name}...")
                with open(file, 'r', encoding='utf-8') as f:
                    all_texts.append(f.read())
                    
        print(f"✅ Total Pretrain files loaded: {len(all_texts)}")
        return all_texts

    def load_sft_data(self, data_dir):
        """/data/sft/ থেকে সব .jsonl file অটোমেটিক লোড করবে"""
        print(f"📂 Loading SFT data from {data_dir}...")
        all_qa =[]
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"❌ Error: {data_dir} ফোল্ডারটি পাওয়া যায়নি!")

        for file in Path(data_dir).iterdir():
            if file.suffix == '.jsonl':
                print(f"  ✓ Loading {file.name}...")
                with open(file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():  # Skip empty lines
                            all_qa.append(json.loads(line))
                            
        print(f"✅ Total SFT pairs loaded: {len(all_qa)}")
        return all_qa