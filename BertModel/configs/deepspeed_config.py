"""DeepSpeed configuration utilities."""

import json
import os

def create_deepspeed_config(config, world_size=1):
    """Create DeepSpeed configuration based on training parameters."""
    
    # Calculate microbatch size
    effective_batch_size = config.training.batch_size * config.training.grad_accum_steps * world_size
    micro_batch_size = config.training.batch_size

    config = {
        "train_batch_size": effective_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": config.training.grad_accum_steps,

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.training.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": config.training.learning_rate,
                "warmup_num_steps": 100,
            }
        },

        # Ensure memory optimization is enabled
        "zero_optimization": {
            "stage": config.deepspeed.zero_stage,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "memory_efficient_linear": True, 
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            }
        },

        "gradient_clipping": config.deepspeed.gradient_clipping,

        # Disable FP16 to avoid dtype mismatch
        "fp16": {
            "enabled": config.deepspeed.fp16_enabled,
        },
        
    }
    
    return config

