import torch
import numpy as np
from transformers import AutoTokenizer
import os


#=======================================================================================================
#                                                                                        Dataset Loading
#======================================================================================================= 


class OptimizedPTDataset(torch.utils.data.Dataset):
    """
    Ultra-fast dataset loading using .pt files
    Replaces CSV-based MultiFileDataset with 10-50x faster loading
    """
    def __init__(self, pos_pt_files, neg_pt_files, load_fraction=1.0):
        """
        Dataset that loads from .pt batch files with intelligent caching
        
        Args:
            pos_pt_files: List of .pt files containing positive samples (SNPs)
            neg_pt_files: List of .pt files containing negative samples (non-SNPs)
            load_fraction: Fraction of data to load (for debugging)
        """
        super().__init__()
        self.pos_pt_files = pos_pt_files if isinstance(pos_pt_files, list) else [pos_pt_files]
        self.neg_pt_files = neg_pt_files if isinstance(neg_pt_files, list) else [neg_pt_files]
        self.load_fraction = load_fraction
        
        # Build sample index mapping: global_idx -> (file_type, file_path, local_idx)
        self.sample_index = []
        self.file_metadata = {'positive': {}, 'negative': {}}
        
        # Index positive files
        total_pos_samples = 0
        for pt_file in self.pos_pt_files:
            # Quick metadata load to get sample count (much faster than CSV)
            try:
                data = torch.load(pt_file, map_location='cpu')
                file_samples = len(data['input_ids'])
                
                self.file_metadata['positive'][pt_file] = {
                    'num_samples': file_samples,
                    'loaded': False,
                    'data': None
                }
                
                # Add global indices for this file
                for local_idx in range(file_samples):
                    self.sample_index.append(('positive', pt_file, local_idx))
                
                total_pos_samples += file_samples
                print(f"Indexed positive file: {pt_file} ({file_samples} samples)")
                
            except Exception as e:
                print(f"Warning: Could not load {pt_file}: {e}")
                continue
        
        # Index negative files
        total_neg_samples = 0
        for pt_file in self.neg_pt_files:
            try:
                data = torch.load(pt_file, map_location='cpu')
                file_samples = len(data['input_ids'])
                
                self.file_metadata['negative'][pt_file] = {
                    'num_samples': file_samples,
                    'loaded': False,
                    'data': None
                }
                
                # Add global indices for this file
                for local_idx in range(file_samples):
                    self.sample_index.append(('negative', pt_file, local_idx))
                
                total_neg_samples += file_samples
                print(f"Indexed negative file: {pt_file} ({file_samples} samples)")
                
            except Exception as e:
                print(f"Warning: Could not load {pt_file}: {e}")
                continue
        
        # Apply load fraction
        if load_fraction < 1.0:
            import random
            random.shuffle(self.sample_index)
            keep_count = int(len(self.sample_index) * load_fraction)
            self.sample_index = self.sample_index[:keep_count]
            print(f"Applied load fraction {load_fraction}: keeping {keep_count}/{len(self.sample_index)} samples")
        
        # Cache management
        self.cache_size = 3  # Number of .pt files to keep in memory
        self.file_cache = {}  # file_path -> data
        self.cache_order = []  # LRU tracking
        
        print(f"OptimizedPTDataset initialized:")
        print(f"  - Positive files: {len(self.pos_pt_files)} ({total_pos_samples} samples)")
        print(f"  - Negative files: {len(self.neg_pt_files)} ({total_neg_samples} samples)")
        print(f"  - Total samples: {len(self.sample_index)}")
        print(f"  - Cache size: {self.cache_size} files")
    
    def _load_pt_file(self, pt_file):
        """Load and cache a .pt file with LRU eviction"""
        # Check if already in cache
        if pt_file in self.file_cache:
            # Move to end (most recently used)
            self.cache_order.remove(pt_file)
            self.cache_order.append(pt_file)
            return self.file_cache[pt_file]
        
        # Load the file - this is ultra-fast for .pt files!
        data = torch.load(pt_file, map_location='cpu')
        
        # Manage cache size
        if len(self.file_cache) >= self.cache_size:
            # Remove least recently used file
            oldest_file = self.cache_order.pop(0)
            del self.file_cache[oldest_file]
        
        # Add to cache
        self.file_cache[pt_file] = data
        self.cache_order.append(pt_file)
        
        return data
    
    def __len__(self):
        return len(self.sample_index)
    
    def __getitem__(self, idx):
        """Get a single sample - properly extract sparse labels per sample"""
        file_type, pt_file, local_idx = self.sample_index[idx]
        
        # Load file data (cached for speed)
        file_data = self._load_pt_file(pt_file)
        
        # Extract sample using tensor indexing
        sample = {
            'input_ids': file_data['input_ids'][local_idx],
            'attention_mask': file_data['attention_mask'][local_idx], 
            'has_mutation': file_data['has_mutation'][local_idx],
        }
        
        # Add metadata if available
        if 'metadata' in file_data:
            sample['chrom'] = file_data['metadata']['chrom'][local_idx]
            sample['pos'] = file_data['metadata']['pos'][local_idx] 
            sample['gene'] = file_data['metadata']['gene'][local_idx]
        
        # FIXED: Extract sparse labels for THIS specific sample only
        # Filter snp_labels to only include rows for this local_idx
        snp_labels = file_data.get('snp_labels', torch.empty((0, 3)))
        if len(snp_labels) > 0:
            # Keep only rows where batch_idx matches local_idx
            mask = snp_labels[:, 0] == local_idx
            sample['snp_labels'] = snp_labels[mask]
            # Remove the batch index column since it's no longer needed
            if sample['snp_labels'].numel() > 0:
                sample['snp_labels'] = sample['snp_labels'][:, 1:]  # Keep only [token_idx, value]
        else:
            sample['snp_labels'] = torch.empty((0, 2))  # [token_idx, value] format
        
        # Same for cons_labels
        cons_labels = file_data.get('cons_labels', torch.empty((0, 3)))
        if len(cons_labels) > 0:
            mask = cons_labels[:, 0] == local_idx
            sample['cons_labels'] = cons_labels[mask]
            if sample['cons_labels'].numel() > 0:
                sample['cons_labels'] = sample['cons_labels'][:, 1:]  # Keep only [token_idx, value]
        else:
            sample['cons_labels'] = torch.empty((0, 2))
        
        # FIXED: Extract sparse epigenetic data for this sample
        sample['sparse_data'] = {}
        for feature_key in ['rna_data', 'gerp_data', 'dnase_data', 'h3k27ac_data', 
                        'h3k4me3_data', 'h3k36me3_data', 'h3k9me3_data', 
                        'h3k27me3_data', 'h2afz_data']:
            feature_data = file_data.get('sparse_data', {}).get(feature_key, torch.empty((0, 3)))
            if len(feature_data) > 0:
                # Keep only rows for this sample
                mask = feature_data[:, 0] == local_idx
                filtered_data = feature_data[mask]
                if filtered_data.numel() > 0:
                    # Remove batch index column, keep [snp_token, feature_value]
                    sample['sparse_data'][feature_key] = filtered_data[:, 1:]
                else:
                    sample['sparse_data'][feature_key] = torch.empty((0, 2), dtype=torch.float)
            else:
                sample['sparse_data'][feature_key] = torch.empty((0, 2), dtype=torch.float)
        
        return sample



class CurriculumSampler:
    """Updated to work with OptimizedPTDataset"""
    def __init__(self, dataset, start_ratio=0.5, end_ratio=None, 
                 total_epochs=50, transition_start=5, seed=42, world_size=1, rank=0):
        """
        Creates a curriculum learning sampler compatible with OptimizedPTDataset
        """
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.transition_start = transition_start
        self.seed = seed
        self.world_size = world_size
        self.rank = rank
        
        # Count positive and negative samples from the sample index
        if hasattr(dataset, 'sample_index'):
            # OptimizedPTDataset
            self.pos_indices = np.array([i for i, (file_type, _, _) in enumerate(dataset.sample_index) 
                                       if file_type == 'positive'])
            self.neg_indices = np.array([i for i, (file_type, _, _) in enumerate(dataset.sample_index) 
                                       if file_type == 'negative'])

        self.num_pos = len(self.pos_indices)
        self.num_neg = len(self.neg_indices)
        self.total_samples = self.num_pos + self.num_neg
        
        # Calculate natural distribution if not provided
        if end_ratio is None:
            self.end_ratio = self.num_pos / self.total_samples
        else:
            self.end_ratio = end_ratio
            
        self.start_ratio = start_ratio
        
        if self.rank == 0:
            print(f"Curriculum Learning Setup (Updated for .pt files):")
            print(f"- Positive samples: {self.num_pos}")
            print(f"- Negative samples: {self.num_neg}")
            print(f"- Natural distribution: {self.end_ratio:.4f}")
            print(f"- Starting with ratio: {self.start_ratio:.4f}")
            print(f"- Will reach natural distribution by epoch {self.total_epochs}")
    
    def get_indices_for_epoch(self, epoch):
        """Use ALL positive samples every epoch, adjust negatives for ratio"""
        current_ratio = self.get_current_ratio(epoch)
        
        # Create different seeds for each GPU
        seed = self.seed + epoch * 1000 + self.rank * 100
        rng = np.random.RandomState(seed)
        
        # ALWAYS use ALL positive samples available to this GPU
        if self.world_size > 1:
            # Distribute ALL positive samples across GPUs
            pos_per_gpu = len(self.pos_indices) // self.world_size
            pos_remainder = len(self.pos_indices) % self.world_size
            
            # Give remainder to first few GPUs
            if self.rank < pos_remainder:
                num_pos_this_gpu = pos_per_gpu + 1
                start_pos = self.rank * (pos_per_gpu + 1)
            else:
                num_pos_this_gpu = pos_per_gpu
                start_pos = pos_remainder * (pos_per_gpu + 1) + (self.rank - pos_remainder) * pos_per_gpu
            
            end_pos = start_pos + num_pos_this_gpu
            pos_indices = self.pos_indices[start_pos:end_pos]
        else:
            # Single GPU - use ALL positive samples
            pos_indices = self.pos_indices.copy()
            num_pos_this_gpu = len(pos_indices)
        
        # Calculate how many negatives needed to achieve target ratio
        # ratio = pos / (pos + neg)
        # Solve for neg: neg = pos * (1 - ratio) / ratio
        
        if current_ratio >= 1.0:
            # 100% positive - no negatives
            num_neg_needed = 0
        elif current_ratio <= 0.0:
            # This shouldn't happen in your setup, but handle edge case
            num_neg_needed = len(self.neg_indices) // self.world_size
        else:
            # Calculate negatives needed for target ratio
            num_neg_needed = int(num_pos_this_gpu * (1 - current_ratio) / current_ratio)
        
        # Cap negatives by what's available
        max_neg_available = len(self.neg_indices)
        if self.world_size > 1:
            max_neg_available = max_neg_available // self.world_size
        
        num_neg_to_sample = min(num_neg_needed, max_neg_available)
        
        # Sample negative indices
        if num_neg_to_sample > 0:
            neg_indices = rng.choice(self.neg_indices, num_neg_to_sample, replace=False)
        else:
            neg_indices = np.array([], dtype=int)
        
        # Combine and shuffle
        combined_indices = np.concatenate([pos_indices, neg_indices])
        rng.shuffle(combined_indices)
        
        return combined_indices
     
    def get_dataset_for_epoch(self, epoch):
        """
        Returns a subset of the dataset for the given epoch
        This is the recommended approach for multi-GPU training
        """
        indices = self.get_indices_for_epoch(epoch)
        return torch.utils.data.Subset(self.dataset, indices)
    
    def get_sampler_for_epoch(self, epoch):
        """
        Returns a sampler for the given epoch with appropriate class weights
        For multi-GPU training, this returns a DistributedSampler-compatible sampler
        """
        indices = self.get_indices_for_epoch(epoch)
        
        # For multi-GPU, we don't need additional DistributedSampler since we already
        # split the indices appropriately in get_indices_for_epoch
        if self.world_size > 1:
            # Create a simple sequential sampler since data is already distributed
            return torch.utils.data.SequentialSampler(
                torch.utils.data.Subset(self.dataset, indices)
            )
        else:
            # Single GPU case
            return torch.utils.data.SubsetRandomSampler(indices)
    
    def get_current_ratio(self, epoch):
        """Returns the current SNP ratio for the epoch"""
        if epoch < self.transition_start:
            return self.start_ratio
        else:
            progress = min(1.0, (epoch - self.transition_start) / 
                          (self.total_epochs - self.transition_start))
            return self.start_ratio + progress * (self.end_ratio - self.start_ratio)
        


def collate_fn(batch):
    """FIXED collate function that properly creates dense labels"""
    batch_size = len(batch)
    
    if batch_size == 0:
        return {}
    
    seq_len = batch[0]['input_ids'].shape[0]
    
    # Stack dense data
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    has_mutation = torch.stack([item['has_mutation'] for item in batch])
    
    # Initialize dense labels
    snp_labels = torch.zeros((batch_size, seq_len), dtype=torch.long)
    cons_labels = torch.zeros((batch_size, seq_len), dtype=torch.float)
    
    # Initialize sparse data collectors
    combined_sparse = {
        'rna_data': [],
        'gerp_data': [], 
        'dnase_data': [],
        'h3k27ac_data': [],
        'h3k4me3_data': [],
        'h3k36me3_data': [],
        'h3k9me3_data': [],
        'h3k27me3_data': [],
        'h2afz_data': []
    }
    
    # Process each sample in batch
    for batch_idx, item in enumerate(batch):
        # FIXED: Process SNP labels for this sample
        # Now item['snp_labels'] format is [token_idx, value] (batch index already removed)
        if len(item['snp_labels']) > 0:
            for snp_row in item['snp_labels']:
                if len(snp_row) >= 2:
                    token_idx, value = snp_row[0], snp_row[1]
                    if 0 <= token_idx < seq_len:
                        snp_labels[batch_idx, int(token_idx)] = int(value)
        
        # FIXED: Process CONS labels for this sample  
        if len(item['cons_labels']) > 0:
            for cons_row in item['cons_labels']:
                if len(cons_row) >= 2:
                    token_idx, value = cons_row[0], cons_row[1]
                    if 0 <= token_idx < seq_len:
                        cons_labels[batch_idx, int(token_idx)] = float(value)
        
        # FIXED: Combine sparse epigenetic data
        for feature_key in combined_sparse.keys():
            feature_data = item['sparse_data'].get(feature_key, torch.empty((0, 2)))
            if len(feature_data) > 0:
                # Convert [snp_token, feature_value] to [batch_idx, snp_token, feature_value]
                adjusted_data = []
                for row in feature_data:
                    if len(row) >= 2:
                        adjusted_data.append([batch_idx, row[0], row[1]])
                combined_sparse[feature_key].extend(adjusted_data)
    
    # Convert to tensors
    for key in combined_sparse:
        if combined_sparse[key]:
            combined_sparse[key] = torch.tensor(combined_sparse[key], dtype=torch.float)
        else:
            combined_sparse[key] = torch.empty((0, 3), dtype=torch.float)
    
    local_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "model", "DNABERT-2-117M"
    )

    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
    CLS_ID = tokenizer.cls_token_id
    SEP_ID = tokenizer.sep_token_id

    # Mask for special tokens
    special_token_mask = (input_ids == CLS_ID) | (input_ids == SEP_ID)

    # Apply padding and special token mask
    padding_mask = (attention_mask == 0) | special_token_mask
    snp_labels[padding_mask] = -100
    cons_labels[padding_mask] = -100.0
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'snp_label': snp_labels,     
        'cons_label': cons_labels,   
        'sparse_data': combined_sparse,
        'has_mutation': has_mutation,
        **{field: [item.get(field, '') for item in batch] 
           for field in ['chrom', 'pos', 'gene'] if field in batch[0]}
    }