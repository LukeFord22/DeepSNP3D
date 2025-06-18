

import os
import sys
import torch
import deepspeed
import wandb
import random
import traceback
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
from datetime import datetime

# Add parent directory for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from BertModel.configs.config import default_config
from BertModel.training.train import train_model
from BertModel.training.setup_deepspeed import setup_training

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

os.environ['WANDB_API_KEY'] = 'f9ef5f01cec347e6598e41294b2dce5380dedbc1'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_gpus():
    return torch.cuda.device_count()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.init_process_group(
        backend='nccl',
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

def cleanup():
    dist.destroy_process_group()


# =============================================================================
# TRAINING PROCESS
# =============================================================================

def train_process(rank, world_size):
    """Main training loop for a single rank."""
    
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["NCCL_DEBUG"] = "WARN"
    
    torch.cuda.set_device(rank)
    
    try:
        print(f"[Rank {rank}] Starting training process...")
        
        if world_size > 1:
            print(f"[Rank {rank}] Setting up distributed training...")
            setup(rank, world_size)
            deepspeed.init_distributed()
        
        device = torch.device(f"cuda:{rank}")
        config = default_config
        config.experiment_name = f"Training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        print(f"[Rank {rank}] Loading models and setting up training...")
        # Use our setup_training function instead of setup_fusion_and_deepspeed
        (model_engine, optimizer, scheduler, train_loader, test_loader, 
         device, local_rank, dna_tokenizer, multi_dataset, curriculum_sampler, 
         world_size_from_setup, rank_from_setup) = setup_training(config)
         
        print(f"[Rank {rank}] Training setup complete.")

        # Update config runtime references
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
        
        if world_size > 1 and dist.is_initialized():
            print(f"[Rank {rank}] Synchronizing processes...")
            dist.barrier()

        print(f"[Rank {rank}] Starting training: {config.training.num_epochs} epochs")

        train_model(config)
        
        print(f"[Rank {rank}] ✅ Training completed successfully!")

    except Exception as e:
        # Clean error message
        print(f"\n[Rank {rank}] ❌ Exception during training:")
        print(f"[Rank {rank}] Error: {str(e)}")
        print(f"[Rank {rank}] Full traceback:\n{traceback.format_exc()}")

        # Attempt WANDB logging
        try:
            if wandb.run is not None:
                wandb.log({
                    "error_message": str(e),
                    "error_traceback": traceback.format_exc(),
                    "rank": rank
                })
                wandb.finish(exit_code=1)
        except Exception as wandb_error:
            print(f"[Rank {rank}] ⚠️ Failed to log to wandb: {wandb_error}")

        # Flush any pending stdout/stderr (important for multi-GPU clusters)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except:
            pass

        # Always re-raise the original exception
        raise

    finally:
        # Final distributed cleanup
        if world_size > 1:
            try:
                cleanup()
                print(f"[Rank {rank}] ✅ Cleanup successful.")
            except Exception as cleanup_error:
                print(f"[Rank {rank}] ⚠️ Cleanup warning: {cleanup_error}")
# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main function to launch training."""
    try:
        set_seed(42)

        available_gpus = count_gpus()
        if available_gpus == 0:
            print("No GPUs available, running on CPU")
            world_size = 1
        else:
            world_size = available_gpus

        print(f"Launching training with {world_size} {'GPU' if world_size == 1 else 'GPUs'}")
        print(f"Available GPUs: {available_gpus}")
        print(f"Configuration: {default_config.training.num_epochs} epochs, batch size {default_config.training.batch_size}")

        if world_size == 1:
            # Single GPU/CPU mode - run directly
            print("Starting single GPU training...")
            train_process(0, 1)
            print("✅ Single GPU training completed successfully")
        else:
            # Multi-GPU mode - use multiprocessing
            print(f"Starting multi-GPU training with {world_size} processes...")
            mp.spawn(
                train_process,
                args=(world_size,),
                nprocs=world_size,
                join=True
            )
            print("✅ Multi-GPU training completed successfully")

    except Exception as e:
        print(f"❌ Training launch failed:")
        print(f"Error: {str(e)}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        print("\n" + "="*50)
        print("DEBUGGING INFO:")
        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
        print("="*50)
        return 1

    print("🎉 Training pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    if exit_code == 0:
        print("Script finished successfully")
    else:
        print(f"Script failed with exit code: {exit_code}")
    sys.exit(exit_code)

