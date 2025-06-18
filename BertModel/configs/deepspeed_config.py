"""DeepSpeed configuration utilities."""

import json
import os

def create_deepspeed_config(training_config, world_size=1):
    """Create DeepSpeed configuration based on training parameters."""
    
    # Calculate microbatch size
    effective_batch_size = training_config.batch_size * training_config.grad_accum_steps * world_size
    micro_batch_size = training_config.batch_size
    
    config = {
        "train_batch_size": effective_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": training_config.grad_accum_steps,
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": training_config.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": training_config.learning_rate,
                "warmup_num_steps": training_config.warmup_steps,
                "total_num_steps": 1000  # Will be updated during training
            }
        },

                "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False,  # Keep on GPU for better performance
            "contiguous_memory_optimization": True,
            "synchronize_checkpoint_boundary": True,
            "profile": False  # Set to True for debugging
        },
        
        # Ensure memory optimization is enabled
        "zero_optimization": {
            # ...your existing zero config...
            "memory_efficient_linear": True,  # Additional memory savings
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            }
        },

        "gradient_clipping": 5.0,
        
        # Disable FP16 to avoid dtype mismatch
        "fp16": {
            "enabled": False
        },
        
        "zero_optimization": {
            "stage": 1,
            "contiguous_gradients": True,
            "overlap_comm": True
        },
        
        "wall_clock_breakdown": False,
        "steps_per_print": 2147483647,  # Set to max int to effectively disable
        "dump_state": False,
        
        # Correct logging configuration
        "tensorboard": {
            "enabled": False
        },
        
        # DeepSpeed internal logging control
        "disable_allgather": False,
        "prescale_gradients": False,
        "sparse_gradients": False
    }
    
    return config

def save_deepspeed_config(config, config_path):
    """Save DeepSpeed configuration to file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return config_path
