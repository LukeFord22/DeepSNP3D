# DeepSNP: Multi-Task DNA SNP Detection and  Score Prediction

A high-performance deep learning system for detecting Single Nucleotide Polymorphisms (SNPs) and predicting  functional scores in DNA sequences using DNABERT-2 transformer architecture with DeepSpeed optimization.

## 🧬 Overview

DeepSNP leverages the state-of-the-art DNABERT-2 model (117M parameters) for multi-task learning on genomic data. The system performs:

### Primary Tasks
- **SNP Classification**: Binary classification of sequences containing SNPs at both token and DNA sequence levels
- **Epigenetic Score Prediction**: Token-level regression for comprehensive functional scores:
  - **RNA Expression**: EncodetotalRNA_sum - RNA-based functional impact scores
  - **Conservation**: ConsScore - Evolutionary conservation scores  
  - **GERP**: GerpRS - Genomic Evolutionary Rate Profiling scores
  - **DNase Hypersensitivity**: EncodeDNase_sum - Chromatin accessibility
  - **Histone Modifications**: Multiple histone marks (H3K27ac, H3K4me3, H3K36me3, H3K9me3, H3K27me3, H2AFZ)
- **Confidence Prediction**: Model uncertainty estimation for both token and sequence-level predictions

### Advanced Features
- **Transformer Architecture**: DNABERT-2 pre-trained on genomic data with enhanced attention mechanisms
- **Multi-Task Learning**: Joint optimization of classification and regression objectives across 10+ epigenetic features
- **Sparse Data Handling**: Efficient processing of sparse epigenetic annotations using coordinate-based storage
- **Multi-GPU Training**: DeepSpeed integration for distributed training with ZeRO optimization
- **Memory Optimization**: ZeRO optimizer with CPU offloading and gradient checkpointing
- **Enhanced Attention**: Multi-scale attention with rotary embeddings and SNP-aware modulation
- **Temperature Scaling**: Learnable temperature parameters for all prediction heads
- **Token-Level Predictions**: Fine-grained epigenetic score prediction at nucleotide resolution
- **Curriculum Learning**: Progressive training from balanced to natural class distributions
- **Data Augmentation**: Overlapping window generation for improved model robustness

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU(s)
- 8GB+ GPU memory recommended

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd DeepSNP
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Interact with Git:
   ```bash
   git config --global user.name "LukeFord22"
   git config --global user.email "lhford1213@gmail.com"
   cd DeepSNP3D
   git add .
   git commit -m "Update: DeepSNP DNA SNP detection system with DNABERT-2 and DeepSpeed"
   git push -u origin main

   # remove all staged files and start over
   git reset --soft HEAD~2
   git reset HEAD .
   ```


### Training
Navigate to the Training directory and use the auto-launcher:
```bash
cd Training
python launch_training.py
```

The launcher automatically:
- Detects your GPU configuration
- Chooses optimal training settings
- Handles distributed setup
- Manages DeepSpeed configuration

## 🏗️ Architecture

### Model Components
- **Base Model**: DNABERT-2-117M (zhihan1996/DNABERT-2-117M)
- **Multi-Task Architecture**: Separate processing pipelines for SNP detection and epigenetic prediction
- **Enhanced Attention**: 
  - **Token Attention**: Multi-scale windowed attention with rotary embeddings
  - **DNA Attention**: SNP-aware sequence-level attention with landmark-based pooling  
- **Epigenetic Backbone**: Dedicated processing pipeline for chromatin features
- **Temperature Scaling**: Individual learnable temperatures for all 12+ prediction heads
- **Residual Processing**: Deep residual blocks with layer normalization and dropout
- **Sparse Data Integration**: Efficient coordinate-based sparse feature handling

### Multi-Task Learning Architecture
The system uses a shared DNABERT-2 encoder with specialized processing pipelines:

1. **SNP Detection Pipeline**:
   - Token-level SNP classification with confidence estimation
   - DNA-level sequence classification with SNP-aware attention
   - Binary cross-entropy loss with temperature scaling

2. **Epigenetic Prediction Pipeline**:
   - Dedicated epigenetic feature backbone
   - Individual prediction heads for each chromatin mark
   - MSE loss for continuous epigenetic scores
   - Sparse data integration for memory efficiency

3. **Attention Mechanisms**:
   - **Multi-Scale Token Attention**: Motif (16bp), regulatory (64bp), and domain (256bp) windows
   - **Rotary Position Embeddings**: Critical for SNP position sensitivity
   - **SNP-Aware Modulation**: Learnable attention gating based on SNP predictions
   - **Landmark Attention**: Efficient processing of long sequences

### Enhanced Data Loading
- **Optimized PT Dataset**: 10-50x faster loading using pre-tokenized .pt files
- **Sparse Data Structure**: Coordinate-based storage [batch_idx, token_idx, value] for epigenetic features
- **Memory Efficient Collation**: Dense tensors for common features, sparse for rare annotations
- **Curriculum Sampling**: Progressive difficulty with balanced-to-natural class distribution transition

###  Score Processing
- **RNA Scores**: Log-transformed and min-max normalized
- **Consequnce Scores**: Original scale preservation
- **GERP Scores**: Log-transformed and min-max normalized
- **Smoothing**: Gaussian smoothing windows around SNP positions
- **Token Masking**: Strategic masking for token scores

### Enhanced Training Features
- **Optimized Data Loading**: 10-50x faster loading with pre-tokenized .pt files
- **Sparse Feature Processing**: Memory-efficient coordinate-based epigenetic data handling
- **Multi-Scale Attention**: Biologically-motivated attention windows (motif/regulatory/domain)
- **Advanced Loss Functions**: Focal loss, consistency regularization, adaptive weighting
- **Temperature Scaling**: Individual learnable temperatures for all 12+ prediction heads
- **Curriculum Learning**: Progressive transition from balanced to natural class distributions
- **Data Augmentation**: Overlapping window generation for improved robustness
- **Flash Attention Compatibility**: Automatic fallback with comprehensive patches
- **Mixed Precision**: FP16 training with automatic loss scaling and gradient clipping

## 🧪 Epigenetic Score Prediction

### Overview
DeepSNP predicts comprehensive epigenetic and functional scores at token level, providing fine-grained chromatin state and evolutionary constraint assessment for each nucleotide position.

### Epigenetic Features (9 Chromatin Marks)

#### 1. RNA Expression (EncodetotalRNA_sum)
- **Purpose**: RNA-based functional impact assessment
- **Processing**: Sparse coordinate storage, sigmoid activation
- **Range**: [0, 1] probability of RNA expression activity
- **Storage**: Coordinate-based for memory efficiency

#### 2. DNase Hypersensitivity (EncodeDNase_sum)  
- **Purpose**: Chromatin accessibility measurement
- **Processing**: Sparse storage, sigmoid normalization
- **Range**: [0, 1] probability of open chromatin state
- **Significance**: Regulatory element identification

#### 3. Histone Modifications (6 marks)
- **H3K27ac**: Active enhancers and promoters
- **H3K4me3**: Active promoters and transcription start sites
- **H3K36me3**: Actively transcribed gene bodies
- **H3K9me3**: Heterochromatin and gene silencing
- **H3K27me3**: Polycomb-mediated gene repression
- **H2AFZ**: Nucleosome positioning and chromatin dynamics

#### 4. Conservation Scores
- **ConsScore**: Evolutionary conservation assessment
- **GerpRS**: GERP (Genomic Evolutionary Rate Profiling) scores
- **Processing**: Individual temperature scaling and sigmoid activation

### Sparse Data Architecture

#### Storage Format
```python
# Coordinate-based sparse storage: [batch_idx, token_idx, value]
sparse_data = {
    'rna_data': torch.tensor([[0, 45, 0.8], [0, 67, 0.3], [1, 23, 0.9]]),
    'dnase_data': torch.tensor([[0, 44, 0.7], [1, 89, 0.4]]),
    # ... other epigenetic features
}
```

#### Benefits
- **Memory Efficiency**: 90%+ reduction in memory usage for sparse annotations
- **Fast Access**: O(1) lookup via coordinate mapping
- **Scalable**: Handles millions of sequences with sparse epigenetic data
- **Flexible**: Easy addition of new epigenetic marks

### Token-Level Prediction Pipeline

1. **Sparse Data Loading**: Epigenetic features loaded in coordinate format for memory efficiency
2. **Feature Processing**: Individual temperature scaling and sigmoid activation for each epigenetic mark
3. **Token Alignment**: Sparse coordinates mapped to DNABERT-2 token positions
4. **Multi-Head Prediction**: Separate prediction heads for each epigenetic feature
5. **Loss Calculation**: Individual MSE losses with adaptive weighting
6. **Sparse Integration**: Efficient coordinate-based lookup during training and inference
7. **Visualization**: Token-by-token epigenetic state visualization in HTML tables

### Multi-Task Loss Function
```python
# Enhanced multi-task loss with adaptive weighting
Total Loss = λ₁ × Token_SNP_Loss + λ₂ × DNA_SNP_Loss + 
             λ₃ × Token_Conf_Loss + λ₄ × DNA_Conf_Loss +
             Σᵢ αᵢ × Epigenetic_Loss_i

# Where i includes: RNA, DNase, H3K27ac, H3K4me3, H3K36me3, 
#                   H3K9me3, H3K27me3, H2AFZ, ConsScore, GERP
```

### Advanced Loss Components
- **Focal Loss**: Addresses class imbalance in SNP detection
- **Consistency Loss**: Regularization between token and DNA predictions  
- **Temperature Scaling**: Individual learnable temperatures for calibration
- **Adaptive Weighting**: Dynamic loss component balancing during training


## 📊 Performance Improvements

### DeepSpeed Optimizations
- **50% Memory Reduction**: ZeRO stage 2 with optimizer state partitioning
- **Multi-GPU Scaling**: Automatic data parallelism across available GPUs  
- **CPU Offloading**: Optimizer states offloaded to CPU memory
- **Mixed Precision**: FP16 training without manual AMP management
- **Gradient Optimization**: Automatic accumulation and clipping (max norm: 5.0)
- **Communication Efficiency**: Overlap computation with communication

### Training Enhancements
- **Adaptive Batch Sizing**: Automatic batch size based on GPU memory
- **Smart Checkpointing**: Regular model state saves with resume capability
- **Robust Error Handling**: Automatic fallback strategies for training failures
- **Real-time Monitoring**: Comprehensive Wandb integration for experiment tracking

## 📁 Project Structure

```
DeepSNP/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── LICENSE                           # Project license
├── DataFetch/DeepGene3/
│   ├── DATABERT.py                   # Enhanced data generation with overlapping windows
│   └── SNPTOKENBERT.py               # Multi-task tokenization with sparse epigenetic processing
└── BertModel/                        # Main training directory
    ├── scripts/
    │   └── launch_training.py        # Auto-launcher with GPU detection
    ├── configs/
    │   ├── config.py                 # Enhanced training configuration
    │   └── deepspeed_config.py       # DeepSpeed configuration with sparse data support
    ├── model/
    │   ├── Model.py                  # Enhanced DNABERTClassifier with 9+ epigenetic heads
    │   └── Loss.py                   # Advanced multi-task loss functions
    ├── training/
    │   ├── train.py                  # Multi-task training with sparse data integration
    │   ├── setup_deepspeed.py        # DeepSpeed setup with memory optimization
    │   └── evaluate.py               # Comprehensive evaluation with epigenetic visualization
    └── data/
        ├── dataset_loading.py        # OptimizedPTDataset with sparse coordinate handling
        └── SNPdata/                  # Training data directory
            ├── positive_*.pt         # Pre-tokenized positive samples with sparse annotations
            └── negative_*.pt         # Pre-tokenized negative samples
    
```

#### Key Components

#### DATABERT.py
- Enhanced data generation pipeline
- Overlapping window creation for data augmentation
- Same SNPs at different positions (2-3 additional positives per SNP)
- Overlapping negatives without SNPs (3-4 additional samples)
- Window size variation (±20%) for robustness

#### SNPTOKENBERT.py
- Multi-task tokenization pipeline with sparse epigenetic processing
- Coordinate-based sparse feature extraction and storage
- Token-level label alignment for 9+ epigenetic marks
- Memory-efficient .pt file generation

#### dataset_loading.py (OptimizedPTDataset)
- 10-50x faster data loading using pre-tokenized .pt files
- Sparse coordinate-based epigenetic feature handling
- Memory-efficient batch collation with dense/sparse tensor separation
- Curriculum sampling with balanced-to-natural progression

#### Model.py (Enhanced Architecture)
- Multi-scale token attention with rotary embeddings
- SNP-aware DNA attention with landmark pooling
- Dedicated epigenetic backbone for chromatin feature processing
- Individual temperature scaling for 12+ prediction heads

#### Loss.py (Advanced Loss Functions)  
- Focal loss for class imbalance handling
- Consistency regularization between token and DNA predictions
- Adaptive multi-task loss weighting
- Individual MSE losses for all epigenetic features

#### train.py (Enhanced Training)
- Sparse data integration throughout training loop
- Comprehensive multi-task metric tracking
- Advanced visualization with epigenetic overlays
- Temperature parameter monitoring and optimization

## 🧪 Dataset

### Data Format
- **Positive Samples**: DNA sequences containing SNPs with comprehensive epigenetic annotations
- **Negative Samples**: DNA sequences without SNPs, with overlapping regions for robustness
- **Enhanced Data Augmentation**: 
  - **Overlapping Positives**: Same SNPs at different window positions (2-3 additional samples per SNP)
  - **Overlapping Negatives**: Adjacent regions without SNPs (3-4 additional samples per original)
  - **Window Size Variation**: ±20% size variation for robustness
- **Tokenization**: Pre-tokenized using DNABERT-2 tokenizer with optimized .pt file storage
- **Sequence Length**: Up to 1536 tokens (dynamic with curriculum learning)
- **Class Distribution**: Natural imbalance (14.3% positive, 85.7% negative) with curriculum progression

### Multi-Task Labels
Each sample includes comprehensive genomic annotations:

#### Classification Labels
- **Token-Level SNP**: Binary labels (0/1) for each token position
- **Sequence-Level SNP**: Binary label for entire DNA sequence
- **Confidence Labels**: Ground truth confidence scores for calibration

#### Sparse Epigenetic Labels (Token-Level)
- **RNA Expression**: EncodetotalRNA_sum scores (sparse coordinates)
- **Chromatin Accessibility**: EncodeDNase_sum values (sparse)
- **Active Marks**: H3K27ac, H3K4me3, H3K36me3 (sparse coordinates)
- **Repressive Marks**: H3K9me3, H3K27me3 (sparse coordinates)  
- **Structural**: H2AFZ nucleosome positioning (sparse)
- **Conservation**: ConsScore and GerpRS evolutionary constraints

#### Dense Labels
- **Conservation Scores**: ConsScore values for all positions
- **Attention Masks**: Valid token positions
- **Mutation Flags**: Sequence-level mutation indicators

### Enhanced Data Processing Pipeline
1. **Optimized Loading**: Pre-tokenized .pt files for 10-50x faster data loading
2. **Sparse Annotation Processing**: Coordinate-based epigenetic feature storage
3. **Data Augmentation**: 
   - Overlapping positive windows with same SNPs at different positions
   - Overlapping negative windows excluding SNP positions
   - Window size variation (±20%) for robustness
4. **Efficient Collation**: 
   - Dense tensors for common features (input_ids, attention_mask)
   - Sparse coordinate tensors for epigenetic annotations
   - Memory-efficient batch assembly
5. **Curriculum Integration**: Balanced-to-natural distribution progression
6. **Multi-GPU Compatibility**: Distributed data loading with proper synchronization

### Curriculum Learning
The training uses curriculum learning to handle class imbalance:
1. **Phase 1**: Start with balanced dataset (50:50 ratio)
2. **Phase 2**: Gradually transition to natural distribution
3. **Phase 3**: Full natural distribution training

## 📊 Monitoring & Evaluation

### Wandb Integration
Comprehensive experiment tracking including:

#### Classification Metrics
- **Training Metrics**: Loss curves, learning rate schedules
- **Validation Metrics**: Accuracy, precision, recall, F1-score
- **Confusion Matrices**: Class-wise performance analysis
- **Confidence Distributions**: Model calibration assessment

#### Regression Metrics
- **Epigenetic Loss Tracking**: Individual loss curves for all 9 chromatin features
- **Token-Level MSE**: Mean squared error for each epigenetic score type
- **Sparse Feature Analysis**: Coordinate-based prediction accuracy assessment
- **Multi-Scale Evaluation**: Performance across different genomic scales
- **Feature Correlation**: Cross-epigenetic feature prediction consistency

#### Visualization & Analysis
- **Enhanced Token Visualizations**: Attention patterns with comprehensive epigenetic overlays
- **Interactive HTML Tables**: Token-by-token comparison across all 9+ epigenetic features
- **Sparse Data Visualization**: Coordinate-based feature highlighting
- **Multi-Scale Attention Maps**: Motif, regulatory, and domain-level attention patterns
- **Hardware Utilization**: GPU memory and compute usage with sparse data efficiency
- **Temperature Analysis**: Individual temperature parameter evolution for all prediction heads

### Key Metrics

#### Classification Performance
- **Primary**: Balanced accuracy (handles class imbalance)
- **Secondary**: Precision, Recall, F1-score
- **Calibration**: Expected Calibration Error (ECE)
- **Consistency**: Agreement between multiple predictions

#### Regression Performance
- **Individual Epigenetic MSE**: Mean squared error for each of 9 chromatin marks
- **Sparse Feature Accuracy**: Coordinate-based prediction precision
- **Multi-Task Consistency**: Cross-feature prediction agreement
- **Token-Level Correlation**: Pearson correlation between predicted and true epigenetic scores
- **Genomic Context Analysis**: Performance across different chromatin states and genomic regions

## 🔧 Advanced Features

### Flash Attention Compatibility
The system includes comprehensive patches for DNABERT-2's flash attention implementation:
- **Triton Kernel Fallback**: Automatic fallback when Triton version conflicts occur
- **Standard Attention Replacement**: PyTorch-native attention as backup
- **Import Interception**: Catches and redirects flash attention imports
- **Cache Patching**: Handles cached transformers modules

### Multi-GPU Support
- **Automatic Detection**: Detects available GPUs and optimal configuration
- **Distributed Training**: NCCL backend for efficient communication
- **Fault Tolerance**: Automatic fallback to single-GPU if distributed fails
- **Load Balancing**: Even data distribution across GPUs

## 🚀 Usage Examples

### Basic Training
```bash
cd BertModel/scripts
python launch_training.py
```

### View Configuration
```bash
cd BertModel/configs
python -c "from config import default_config; print(default_config)"
```

### Multi-GPU Training
The launcher automatically detects and uses all available GPUs:
```bash
cd BertModel/scripts
python launch_training.py  # Automatically uses all GPUs
```

## 🛠️ Troubleshooting

### Common Issues
1. **Flash Attention Errors**: Automatically handled by fallback mechanisms
2. **Memory Errors**: Reduce batch size or enable CPU offloading
3. **NCCL Errors**: Use fallback environment variables (automatically set)
4. **Triton Conflicts**: Automatic patching resolves version incompatibilities

### Performance Tuning
- **Batch Size**: Increase until GPU memory is ~90% utilized
- **Gradient Accumulation**: Increase if batch size must be small
- **CPU Offloading**: Enable for very large models or limited GPU memory
- **Mixed Precision**: Enable FP16 for 2x speedup (enabled by default)

## 📝 Citation

If you use DeepSNP3 in your research, please cite:

```bibtex
@misc{deepsnp2025,
  title={DeepSNP3: Multi-Task DNA SNP Detection and Epigenetic Score Prediction with Enhanced DNABERT-2},
  author={LukeFord},
  year={2025},
  note={Advanced multi-task learning system for SNP classification and comprehensive epigenetic regression with sparse data optimization},
  url={https://github.com/LukeFord22/DeepSNP3}
}
```

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🤝 Contributing



