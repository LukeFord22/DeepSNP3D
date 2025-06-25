import os
import sys
import torch
import random
import traceback
import numpy as np
from datetime import datetime

# Add parent directory for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from BERTModel.configs.config import default_config
from BERTModel.training.train import train_model
from BERTModel.training.setup_deepspeed import setup_training

os.environ["NCCL_P2P_DISABLE"] = "1"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_gpu_info():
    """Print detailed GPU information"""
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count == 0:
        print("❌ No CUDA GPUs detected. Exiting.")
        sys.exit(1)
    print(f"🔍 Detected {gpu_count} GPU(s):")
    for i in range(gpu_count):
        gpu_props = torch.cuda.get_device_properties(i)
        memory_gb = gpu_props.total_memory / (1024**3)
        print(f"  GPU {i}: {gpu_props.name} ({memory_gb:.1f}GB)")
    print()

def train_process_torchrun(rank, world_size, local_rank):
    """Training process for torchrun launch."""
    if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
        torch.cuda.set_device(local_rank)
        print(f"[Rank {rank}] Set CUDA device to GPU {local_rank}")

    try:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        config = default_config
        config.experiment_name = f"Training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(f"[Rank {rank}] Setting up training...")
        (model_engine, optimizer, scheduler, train_loader, test_loader, 
         device, local_rank, dna_tokenizer, multi_dataset, curriculum_sampler, 
         world_size_from_setup, rank_from_setup) = setup_training(config)
        print(f"[Rank {rank}] Training setup complete.")

        config.deepspeed_runtime = {
            "model_engine": model_engine,
            "device": device,
            "rank": local_rank,
            "world_size": world_size,
            "dna_tokenizer": dna_tokenizer,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "multi_dataset": multi_dataset,
            "curriculum_sampler": curriculum_sampler
        }

        print(f"[Rank {rank}] Starting training: {config.training.num_epochs} epochs")
        train_model(config)
        print(f"[Rank {rank}] ✅ Training completed successfully!")

    except Exception as e:
        print(f"[Rank {rank}] ❌ Exception during training:")
        print(f"[Rank {rank}] Error: {str(e)}")
        print(f"[Rank {rank}] Full traceback:")
        traceback.print_exc()
        raise

def main():
    set_seed(42)
    print("🚀 DeepSNP3D Training Launcher (auto torchrun)")
    print("=" * 50)
    print_gpu_info()

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Only relaunch with torchrun if not already under torchrun and more than 1 GPU
    if "LOCAL_RANK" not in os.environ and gpu_count > 1:
        print(f"🔄 Relaunching with torchrun for {gpu_count} GPUs...")
        cmd = [
            "torchrun",
            f"--nproc_per_node={gpu_count}",
            os.path.abspath(__file__)
        ]
        os.execvp(cmd[0], cmd)

    # If running under torchrun or single GPU, proceed as normal
    if gpu_count == 0:
        print("❌ No CUDA GPUs detected. Exiting.")
        sys.exit(1)
    elif gpu_count == 1 or "LOCAL_RANK" not in os.environ:
        print("✅ Single GPU detected or not under torchrun. Running in single process mode.")
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        train_process_torchrun(0, 1, 0)
    else:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        print(f"✅ Launched by torchrun - Rank {rank}, Local Rank {local_rank}, World Size {world_size}")
        train_process_torchrun(rank, world_size, local_rank)

if __name__ == "__main__":
    main()