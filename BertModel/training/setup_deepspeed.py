"""Setup function for DeepSpeed training initialization."""

import torch
import deepspeed
import os
import sys
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, BertConfig

# Add parent directories to Python path for absolute imports
script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.dirname(script_dir)
bert_model_dir = os.path.dirname(training_dir)
deepsnp_dir = os.path.dirname(bert_model_dir)
sys.path.insert(0, deepsnp_dir)
sys.path.insert(0, bert_model_dir)
sys.path.insert(0, training_dir)
sys.path.insert(0, script_dir)

# Import from packages using absolute imports
from BertModel.model.Model import DNABERTClassifier
from BertModel.data.dataset_loading import OptimizedPTDataset, CurriculumSampler, collate_fn
from BertModel.configs.deepspeed_config import create_deepspeed_config


def setup_training(config):
    """
    Prepares the model, optimizer, scheduler, dataloaders, device, and ranks.
    Initializes DeepSpeed model engine with proper configuration.
    """
      # Initialize distributed training if not already done
    if not torch.distributed.is_initialized():
        # Get local rank from environment or use -1 for single GPU
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        
        if local_rank == -1:
            # Single GPU mode
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                local_rank = 0
            world_size = 1
            rank = 0
        else:
            # Multi-GPU case - initialize distributed
            deepspeed.init_distributed()
            device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(local_rank)
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
    else:
        local_rank = torch.distributed.get_rank()
        device = torch.device(f'cuda:{local_rank}')
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    

    # === DATA LOADING SETUP  ===
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", config.data.data_dir)
    data_dir = os.path.abspath(data_dir)
    
    if local_rank <= 0:
        print(f"📁 Resolved data directory: {data_dir}")
    
    # Find positive and negative .pt files
    pos_files = list(Path(data_dir).glob(f"positive*{config.data.pt_file_pattern}"))
    neg_files = list(Path(data_dir).glob(f"negative*{config.data.pt_file_pattern}"))
    
    # Convert Path objects to strings
    pos_files = [str(f) for f in pos_files]
    neg_files = [str(f) for f in neg_files]
    
    if local_rank <= 0:
        print(f"Found {len(pos_files)} positive .pt files:")
        for f in pos_files:
            print(f"  ⚡ {f}")
        print(f"Found {len(neg_files)} negative .pt files:")
        for f in neg_files:
            print(f"  ⚡ {f}")
    
    # Create the optimized PT dataset
    dataset = OptimizedPTDataset(
        pos_pt_files=pos_files,
        neg_pt_files=neg_files,
        load_fraction=config.training.train_data_fraction
    )

    # Create test indices for evaluation
    test_indices = np.random.choice(
        range(len(dataset)), 
        size=int(len(dataset) * config.data.test_fraction),
        replace=False
    )
    
    # Create test dataset and loader
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size, 
        shuffle=False,
        pin_memory=config.data.pin_memory, 
        num_workers=config.data.num_workers,
        persistent_workers=config.data.persistent_workers and config.data.num_workers > 0,
        collate_fn=collate_fn
    )

    # === LOCAL DNABERT-2 SETUP ===
    hf_model_name = "zhihan1996/DNABERT-2-117M"

    local_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "model", "DNABERT-2-117M"
    )

    # Check if weights exist, download if not
    weight_files = ["pytorch_model.bin", "model.safetensors"]
    weights_exist = any(os.path.exists(os.path.join(local_path, f)) for f in weight_files)
    
    if not weights_exist:
        if local_rank <= 0:
            print(f"🔄 Model weights not found locally. Downloading from {hf_model_name}...")
        download_model_weights(hf_model_name, local_path, local_rank)
        if world_size > 1:
            torch.distributed.barrier()
    else:
        if local_rank <= 0:
            print(f"✅ Model weights found locally at {local_path}")
        if world_size > 1:
            torch.distributed.barrier()

    # Load tokenizer and model configuration
    dna_tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
    bert_config = BertConfig.from_pretrained(local_path)

    # Step 1: Load local model architecture
    base_model = AutoModel.from_pretrained(
        local_path,
        config=bert_config,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    
    model = DNABERTClassifier(base_model, config=config)

    # Create DeepSpeed configuration
    ds_config = create_deepspeed_config(
        config=config,
        world_size=world_size
    )
    
    # Initialize DeepSpeed model engine
    model_engine, optimizer, train_dataloader, scheduler = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )

    # Create curriculum sampler for training with multi-GPU support
    curriculum_sampler = CurriculumSampler(
        dataset=dataset,
        start_ratio=0.5,  # Start with balanced distribution
        end_ratio=None,   # Calculate from data
        total_epochs=config.training.num_epochs,
        transition_start=config.training.curriculum_start,
        seed=getattr(config.training, 'random_seed', 42),  # Use getattr instead of get
        world_size=world_size,
        rank=rank
    )
    # We'll create train_loader dynamically each epoch with curriculum sampler
    # For now, return None as train_loader placeholder
    train_loader = None

    return (model_engine, optimizer, scheduler, train_loader, test_loader, 
            device, local_rank, dna_tokenizer, dataset, curriculum_sampler, 
            world_size, rank)


def download_model_weights(hf_model_name, local_path, local_rank=0):
    """
    Download only the model weight files from Hugging Face to local directory
    """
    import os
    from huggingface_hub import hf_hub_download
    
    # Create local directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)
    
    # Files to download (model weights)
    weight_files = [
        "pytorch_model.bin",
    ]
    
    downloaded_files = []
    
    for filename in weight_files:
        try:
            if local_rank <= 0:
                print(f"📥 Downloading {filename}...")
            
            # Download file to local directory
            downloaded_path = hf_hub_download(
                repo_id=hf_model_name,
                filename=filename,
                local_dir=local_path,
                local_dir_use_symlinks=False  # Create actual files, not symlinks
            )
            downloaded_files.append(filename)
            
            if local_rank <= 0:
                print(f"✅ Downloaded {filename}")
                
        except Exception as e:
            if local_rank <= 0:
                print(f"⚠️ Could not download {filename}: {e}")
    
    if local_rank <= 0:
        print(f"📦 Downloaded {len(downloaded_files)} files to {local_path}")
        print(f"   Files: {downloaded_files}")
    
    return downloaded_files

