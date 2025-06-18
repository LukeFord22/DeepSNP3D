"""Default configuration for DNA SNP detection training."""

import os
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 48
    num_epochs: int = 10
    learning_rate: float = 3e-5
    grad_accum_steps: int = 2
    train_data_fraction: float = 1
    warmup_steps: int = 100
    max_token_length: int = 384 
    random_seed: int = 42  
    
    # Training strategies
    curriculum_start: int = 5
    consistency_start_epoch: int = 5
    consistency_warmup_epochs: int = 3
    
    # Loss weights
    token_snp_weight: float = 10.0
    token_conf_weight: float = 2.0
    dna_snp_weight: float = 10.0
    dna_conf_weight: float = 1.0
    token_fp_penalty_weight: float = 5.0
    dna_fp_penalty_weight: float = 2.0
    token_hnm_weight: float = 1.0
    dna_hnm_weight: float = 50.0
    token_rna_weight: float = 1.0
    token_cons_weight: float = 10.0
    token_gerp_weight: float = 1.0
    token_dnase_weight: float = 1.0
    token_h3k27ac_weight: float = 1.0
    token_h3k4me3_weight: float = 1.0
    token_h3k36me3_weight: float = 1.0
    token_h3k9me3_weight: float = 1.0
    token_h3k27me3_weight: float = 1.0
    token_h2afz_weight: float = 1.0

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    num_attention_heads: int = 16

@dataclass
class DataConfig:
    """Data configuration parameters."""
    data_dir: str = "data/SNPdata"

    # .pt file configuration 
    pt_file_pattern: str = "_tokenized_batch_*.pt"  # Pattern to match .pt batch files
    pt_cache_size: int = 3  # Number of .pt files to keep in memory cache

    # Data loading settings
    test_fraction: float = 0.02
    num_workers: int = 2  # Number of data loading workers
    pin_memory: bool = True  # Faster GPU transfer
    persistent_workers: bool = True  # Keep workers alive between epochs

@dataclass
class DeepSpeedConfig:
    """DeepSpeed configuration parameters."""
    enabled: bool = True
    zero_stage: int = 2
    fp16_enabled: bool = True
    offload_optimizer: bool = True
    gradient_clipping: float = 5.0
    steps_per_print: int = 100

@dataclass
class Config:
    """Combined configuration for the entire project."""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    
    # Experiment settings
    experiment_name: str = field(default_factory=lambda: f"dna-snp-detection-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    project_name: str = "dna-snp-detection-deepspeed"
    output_dir: str = "checkpoints"
    
    # Device settings
    local_rank: int = -1
    device: str = "cuda"

# Create default configuration instance
default_config = Config()