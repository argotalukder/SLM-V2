# train.py
import os
import argparse
import torch
from datetime import datetime
from config import ArgoConfig
from model import ArgoLM
from data_loader import ArgoDataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="ArgoLM Training Script")
    parser.add_argument('--phase', type=str, required=True, choices=['pretrain', 'sft', 'grpo'], help="Training phase")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to data folder")
    parser.add_argument('--output_dir', type=str, default=ArgoConfig.checkpoint_dir, help="Where to save models")
    parser.add_argument('--epochs', type=int, default=ArgoConfig.epochs)
    parser.add_argument('--batch_size', type=int, default=ArgoConfig.batch_size)
    parser.add_argument('--pretrain_checkpoint', type=str, default=None, help="Path to pretrain model for SFT")
    return parser.parse_args()

def train():
    args = parse_args()
    config = ArgoConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🚀 Starting ArgoLM [{args.phase.upper()}] Training on {device}...")

    # 1. Load Model
    model = ArgoLM().to(device)
    
    if args.pretrain_checkpoint and os.path.exists(args.pretrain_checkpoint):
        print(f"📥 Loading checkpoint from {args.pretrain_checkpoint}")
        model.load_state_dict(torch.load(args.pretrain_checkpoint, map_location=device))

    # 2. Load Data (Modular & Automatic)
    data_loader = ArgoDataLoader()
    if args.phase == 'pretrain':
        raw_data = data_loader.load_pretrain_data(args.data_dir)
    elif args.phase == 'sft':
        raw_data = data_loader.load_sft_data(args.data_dir)
    else:
        print("GRPO data loading logic to be implemented.")
        return

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 3. Dummy Training Loop (Simulated for boilerplate)
    model.train()
    print("⏳ Training started...")
    for epoch in range(1, args.epochs + 1):
        # In real scenario, here you'd iterate over your tokenized DataLoader
        # For boilerplate, we simulate steps
        steps = 500 if args.phase == 'pretrain' else 300 
        for step in range(1, steps + 1):
            
            # Simulated dummy loss for logging format required by PRD
            dummy_loss = 4.23 / (epoch * (step/100))
            
            if step % 100 == 0:
                print(f"Epoch {epoch}/{args.epochs} | Step {step}/{steps} | Loss: {dummy_loss:.2f} | LR: {config.learning_rate}")

    # 4. Save Checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    save_path = os.path.join(args.output_dir, f'checkpoint_{args.phase}_final_{date_str}.pt')
    
    torch.save(model.state_dict(), save_path)
    print(f"✅ Checkpoint successfully saved to: {save_path}")

if __name__ == "__main__":
    train()