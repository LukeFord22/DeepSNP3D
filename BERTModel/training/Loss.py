import torch
import torch.nn.functional as F 
import torch.nn as nn


#=======================================================================================================
#                                                                                      Loss Calculations
#=======================================================================================================

def compute_dense_losses(outputs, batch, device, config):
    # == Extract outputs ==
    logits_snp = outputs['token_snp']
    logits_cons = outputs['token_cons']
    logits_snp_conf = outputs['token_conf'] 

    # === Extract label types ===
    # Use dense labels for existing loss functions (SNP and CONS)
    snp_label = batch['snp_label'].to(device).long()           # Dense SNP labels
    cons_label = batch['cons_label'].to(device).float()    # Dense CONS labels

    # === logits and labels for token-level tasks ===
    attention_mask = batch['attention_mask'].to(device)
    valid_token_mask = (attention_mask == 1) & (snp_label != -100)
    valid_cons_mask = valid_token_mask & (cons_label != -100.0)

    # === Token SNP Loss (using dense labels) ===
    if valid_token_mask.any():
        valid_logits = logits_snp[valid_token_mask]
        valid_labels = snp_label[valid_token_mask]

        loss_snp = LearnableWeightFocalLoss(
            gamma=5,
            initial_weight=[1.0, 2500.0],
            weight_constraint='exp',
            num_classes=2
        )(valid_logits, valid_labels)
    else:
        loss_snp = torch.tensor(0.0, device=device)

    # === Token CONS Loss (using dense labels) ===
    if valid_cons_mask.any():
        loss_cons = SNPConsequenceFocalLoss(
            snp_alpha=1.0,           # Focus heavily on SNP magnitude
            snp_gamma=1.0,           # Moderate focus on difficult cases
            background_weight=0.005,  # Very low penalty for background
            snp_threshold=0.1,       # Values > 0.1 considered "high consequence"
            background_threshold=0.05 # Values < 0.05 considered "background"
        )(
            logits_cons[valid_cons_mask].squeeze(-1),
            cons_label[valid_cons_mask]
        )
    else:
        loss_cons = torch.tensor(0.0, device=device)

    # === Other existing losses (unchanged) ===
    # Token-level Confidence Loss
    if valid_token_mask.any():
        valid_snp_logits = logits_snp[valid_token_mask]
        valid_snp_labels = snp_label[valid_token_mask]
        valid_conf_logits = logits_snp_conf[valid_token_mask].squeeze(-1)

        predicted_classes = valid_snp_logits.argmax(dim=1)
        token_correctness = (predicted_classes == valid_snp_labels).float()
        
        loss_snp_conf = TokenConfidenceLoss(
            num_bins=15,
            ece_weight=0.1,
            overconf_penalty=2.0,
            brier_weight=1.0,
            focal_gamma=2.0,
            focal_alpha=0.25
        )(valid_conf_logits, token_correctness) 
    else:
        loss_snp_conf = torch.tensor(0.0, device=device)

    # === Token False Positive Loss ===
    if valid_token_mask.any():
        valid_snp_logits = logits_snp[valid_token_mask]
        valid_snp_labels = snp_label[valid_token_mask]
        valid_conf_logits = logits_snp_conf[valid_token_mask].squeeze(-1)

        predicted_classes = valid_snp_logits.argmax(dim=1)
        fp_mask = (predicted_classes == 1) & (valid_snp_labels == 0)
        
        if fp_mask.any():
            fp_conf = valid_conf_logits[fp_mask]
            token_fp_focal = FalsePositiveFocalPenalty(
                gamma=2.5,
                alpha=2.0,
                confidence_threshold=0.6
            )
            loss_snp_fp = token_fp_focal(fp_conf) 
        else:
            loss_snp_fp = torch.tensor(0.0, device=device)
    else:
        loss_snp_fp = torch.tensor(0.0, device=device)

    # === Token Hard Negative Mining ===
    if valid_token_mask.any():
        valid_logits = logits_snp[valid_token_mask]
        valid_labels = snp_label[valid_token_mask]

        mined_logits, mined_labels = mine_hard_negatives(valid_logits, valid_labels, k_ratio=0.1)
        loss_snp_hnm = WeightedFocalLoss(
            gamma=5, 
            weight=torch.tensor([1.0, 2500.0], device=device),
        )(mined_logits, mined_labels) 
    else:
        loss_snp_hnm = torch.tensor(0.0, device=device)

    #=== Precision Loss ===
    if valid_token_mask.any():
        snp_precision_loss = PrecisionWeightedLoss(
            token_weight=1.0, 
            base_loss_fn=WeightedFocalLoss(gamma=2.0, weight=torch.tensor([1.0, 1000.0], device=logits_snp.device))
        )(logits_snp[valid_token_mask], snp_label[valid_token_mask], calculate_metrics=True)
    else:
        snp_precision_loss = torch.tensor(0.0, device=device)



    return {
        'loss_snp': loss_snp * config.training.snp_weight,
        'loss_snp_conf': loss_snp_conf * config.training.conf_weight,
        'loss_snp_fp': loss_snp_fp * config.training.fp_penalty_weight,
        'loss_snp_hnm': loss_snp_hnm * config.training.hnm_weight,
        'loss_cons': loss_cons * config.training.cons_weight,
        'loss_snp_precision': snp_precision_loss * config.training.snp_weight,
    }
    



def compute_sparse_losses(outputs, batch, device, config):
    """
    Compute sparse losses using BATCHED processing for maximum efficiency
    """
    # Initialize all expected loss keys with zero values
    losses = {
        'loss_rna': torch.tensor(0.0, device=device, requires_grad=True),
        'loss_gerp': torch.tensor(0.0, device=device, requires_grad=True),
        'loss_dnase': torch.tensor(0.0, device=device, requires_grad=True),
        'loss_h3k27ac': torch.tensor(0.0, device=device, requires_grad=True),
        'loss_h3k4me3': torch.tensor(0.0, device=device, requires_grad=True),
        'loss_h3k36me3': torch.tensor(0.0, device=device, requires_grad=True),
        'loss_h3k9me3': torch.tensor(0.0, device=device, requires_grad=True),
        'loss_h3k27me3': torch.tensor(0.0, device=device, requires_grad=True),
        'loss_h2afz': torch.tensor(0.0, device=device, requires_grad=True)
    }
    
    sparse_data = batch['sparse_data']
    
    # Get max_token_length from config
    max_token_length = config.training.max_token_length
    
    # Define smoothing configurations (must match tokenization script)
    smoothing_configs = {
        'rna': {'window_size': 20, 'baseline_ratio': 0.15},
        'gerp': {'window_size': 20, 'baseline_ratio': 0.25},
        'dnase': {'window_size': 15, 'baseline_ratio': 0.10},
        'h3k27ac': {'window_size': 25, 'baseline_ratio': 0.20},
        'h3k4me3': {'window_size': 20, 'baseline_ratio': 0.30},
        'h3k36me3': {'window_size': 30, 'baseline_ratio': 0.15},
        'h3k9me3': {'window_size': 35, 'baseline_ratio': 0.35},
        'h3k27me3': {'window_size': 30, 'baseline_ratio': 0.25},
        'h2afz': {'window_size': 15, 'baseline_ratio': 0.20}
    }
    
    # Define BATCHED loss functions with feature-specific parameters
    loss_functions = {}
    for feature_name, config_params in smoothing_configs.items():
        loss_functions[feature_name] = BatchedUltraEfficientSpatialLoss(
            window_size=config_params['window_size'],
            baseline_ratio=config_params['baseline_ratio'],
            center_weight=4.0,
            neighbor_weight=2.0,
            edge_weight=1.0,
            spatial_reg=0.15
        ).to(device)
    
    # FIXED: Move entire sparse_data to device once
    sparse_data_on_device = {}
    for key, value in sparse_data.items():
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            sparse_data_on_device[key] = value.to(device)
        else:
            sparse_data_on_device[key] = value
    
    # Process each feature with BATCHED processing
    feature_mappings = [
        ('rna', 'rna_data', 'token_rna'),
        ('gerp', 'gerp_data', 'token_gerp'), 
        ('dnase', 'dnase_data', 'token_dnase'),
        ('h3k27ac', 'h3k27ac_data', 'token_h3k27ac'),
        ('h3k4me3', 'h3k4me3_data', 'token_h3k4me3'),
        ('h3k36me3', 'h3k36me3_data', 'token_h3k36me3'),
        ('h3k9me3', 'h3k9me3_data', 'token_h3k9me3'),
        ('h3k27me3', 'h3k27me3_data', 'token_h3k27me3'),
        ('h2afz', 'h2afz_data', 'token_h2afz')
    ]
    
    for feature_name, data_key, output_key in feature_mappings:
        if output_key in outputs:
            try:
                # Check if we have data for this feature
                has_data = (data_key in sparse_data_on_device and 
                           isinstance(sparse_data_on_device[data_key], torch.Tensor) and
                           sparse_data_on_device[data_key].numel() > 0)
                
                if has_data:
                    # Compute loss for this feature
                    loss_value = loss_functions[feature_name](
                        outputs[output_key], 
                        sparse_data_on_device,  # Pass the device-corrected dict
                        data_key,
                        max_token_length
                    )
                    weight = getattr(config.training, f'{feature_name}_weight', 1.0)
                    losses[f'loss_{feature_name}'] = loss_value * weight
                    
            except Exception as e:
                print(f"Warning: Error computing {feature_name} loss: {e}")
                # Loss already initialized to zero above
    
    return losses



def mine_hard_negatives(logits, labels, k_ratio=0.3):
    """Keep only the k% hardest negative examples for training."""
    with torch.no_grad():
        # Identify negative examples (non-SNPs)
        neg_mask = (labels == 0)
        pos_mask = (labels == 1)
        
        # If no negatives or no positives, return original data
        if not neg_mask.any() or not pos_mask.any():
            return logits, labels
            
        # Get difficulty scores for negatives (higher prob = harder example)
        neg_probs = torch.softmax(logits[neg_mask], dim=-1)[:, 1]
        
        # Keep top k% hardest negatives
        k = max(1, int(neg_mask.sum() * k_ratio))
        _, hard_indices = neg_probs.topk(k)
        
        # Get indices of all negatives
        neg_indices = torch.nonzero(neg_mask, as_tuple=True)[0]
        # Select the hard negative indices
        hard_neg_indices = neg_indices[hard_indices]
        
        # Create mask for all positives and hard negatives
        keep_mask = torch.zeros_like(labels, dtype=torch.bool)
        keep_mask[pos_mask] = True  # Keep all positives
        keep_mask[hard_neg_indices] = True  # Keep hard negatives
        
        return logits[keep_mask], labels[keep_mask]


class PrecisionWeightedLoss(nn.Module):
    def __init__(self, token_weight=0.5, base_loss_fn=None, smoothing=0.95):
        super().__init__()
        self.token_weight = token_weight
        self.smoothing = smoothing
        self.base_loss_fn = base_loss_fn or nn.CrossEntropyLoss()
        self.precision_history = {'token': 0.0}
        self.running_steps = 0
        
    def forward(self, outputs, targets, calculate_metrics=True):
        # Base loss calculation
        loss = self.base_loss_fn(outputs, targets)
        
        if calculate_metrics:
            # Calculate precision metrics
            with torch.no_grad():
                # Get predictions - handle different tensor shapes
                if outputs.dim() > 1 and outputs.size(-1) > 1:  # Classification with logits
                    preds = outputs.argmax(dim=-1)
                else:  # Binary prediction
                    preds = (outputs > 0).long()
                
                # Handle different shapes for targets
                if targets.dim() > 1 and targets.size(-1) > 1:
                    targets = targets.argmax(dim=-1)
                
                # Flatten both tensors to 1D for comparison
                preds = preds.view(-1)
                targets = targets.view(-1)
                
                # Skip padding tokens for token-level tasks
                if self.token_weight > 0:
                    valid_mask = (targets != -100)
                    preds = preds[valid_mask]
                    targets = targets[valid_mask]
                
                # Calculate TP and FP
                tp = ((preds == 1) & (targets == 1)).sum()
                fp = ((preds == 1) & (targets == 0)).sum()
                
                # Calculate precision
                prec = tp.float() / (tp + fp + 1e-8)
                
                # Update precision history with smoothing
                self.precision_history['current'] = prec.item()
                if self.running_steps == 0:
                    self.precision_history['smoothed'] = prec.item()
                else:
                    self.precision_history['smoothed'] = (self.smoothing * self.precision_history['smoothed'] + 
                                                        (1 - self.smoothing) * prec.item())
                self.running_steps += 1
                
                # Scale loss inversely with precision
                precision_factor = 1.0 / (self.precision_history['smoothed'] + 0.1)
                loss = loss * precision_factor
        
        return loss


class TokenConfidenceLoss(nn.Module):
    """
    State-of-the-art token-level confidence loss focused on calibration.
    Does not use sigmoid internally - expects raw logits.
    """
    def __init__(self, num_bins=15, ece_weight=0.1, overconf_penalty=2.0, 
                 brier_weight=1.0, focal_gamma=2.0, focal_alpha=0.25):
        super().__init__()
        self.num_bins = num_bins
        self.ece_weight = ece_weight
        self.overconf_penalty = overconf_penalty
        self.brier_weight = brier_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        
    def expected_calibration_error(self, conf_probs, correctness_labels):
        """Compute ECE using probability bins"""
        with torch.no_grad():
            bin_boundaries = torch.linspace(0, 1, self.num_bins + 1, device=conf_probs.device)
            
            ece = 0.0
            for i in range(self.num_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                
                # Find samples in this bin
                in_bin = (conf_probs > bin_lower) & (conf_probs <= bin_upper)
                prop_in_bin = in_bin.float().mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = correctness_labels[in_bin].mean()
                    avg_confidence_in_bin = conf_probs[in_bin].mean()
                    ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return ece
    
    def forward(self, conf_logits, correctness_labels):
        """
        Args:
            conf_logits: Raw confidence logits [N] - no sigmoid applied
            correctness_labels: Binary correctness labels [N] (0 or 1)
        """
        conf_logits = conf_logits.view(-1)
        correctness_labels = correctness_labels.view(-1).float()
        
        if conf_logits.size(0) == 0:
            return torch.tensor(0.0, device=conf_logits.device, requires_grad=True)
        
        # === 1. Brier Score (proper scoring rule) ===
        brier_score = (conf_logits - correctness_labels) ** 2
        brier_loss = brier_score.mean() * self.brier_weight
        
        # === 2. Expected Calibration Error (ECE) ===
        ece = self.expected_calibration_error(conf_logits, correctness_labels)
        ece_loss = ece * self.ece_weight
        
        # === 3. Overconfidence Penalty ===
        wrong_mask = (correctness_labels == 0)
        if wrong_mask.any():
            wrong_confidences = conf_logits[wrong_mask]
            # Penalize high confidence when wrong
            overconf_penalty = (wrong_confidences ** self.overconf_penalty).mean()
        else:
            overconf_penalty = torch.tensor(0.0, device=conf_logits.device)
        
        # === 4. Focal-style weighting for hard examples ===
        # Focus more on examples where confidence doesn't match correctness
        confidence_error = torch.abs(conf_logits - correctness_labels)
        focal_weight = confidence_error ** self.focal_gamma
        
        # Apply class balancing (similar to focal loss alpha)
        alpha_weight = torch.ones_like(correctness_labels)
        alpha_weight[correctness_labels == 1] = self.focal_alpha
        alpha_weight[correctness_labels == 0] = 1 - self.focal_alpha
        
        # Weighted brier score with focal attention
        focal_brier = brier_score * focal_weight * alpha_weight
        focal_loss = focal_brier.mean()
        
        # === Combine all losses ===
        total_loss = (brier_loss + 
                     ece_loss + 
                     0.1 * overconf_penalty + 
                     0.5 * focal_loss)
        
        return total_loss


class WeightedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=-100, reduction='mean', eps=1e-8):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        targets = targets.view(-1)
        logits = logits.view(-1, num_classes)

        valid_mask = (targets != self.ignore_index)
        targets = targets[valid_mask]
        logits = logits[valid_mask]

        if targets.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        targets_onehot = F.one_hot(targets, num_classes=num_classes).float()

        pt = (probs * targets_onehot).sum(dim=-1).clamp(min=self.eps)  # per-sample correct class prob
        focal_weight = (1 - pt) ** self.gamma

        ce_loss = F.nll_loss(log_probs, targets, weight=self.weight, reduction='none')
        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class LearnableWeightFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, initial_weight=None, num_classes=2, ignore_index=-100, 
                 reduction='mean', eps=1e-8, weight_constraint='softplus'):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.eps = eps
        self.num_classes = num_classes
        self.weight_constraint = weight_constraint
        self.max_weight = 100000.0
        
        # Initialize learnable weights
        if initial_weight is None:
            initial_weight = torch.ones(num_classes)
        else:
            # Make sure initial_weight is a tensor
            initial_weight = torch.tensor(initial_weight) if not isinstance(initial_weight, torch.Tensor) else initial_weight
        
        # Register as parameter so it's included in model.parameters() for optimization
        self.raw_weight = nn.Parameter(torch.log(initial_weight + 1e-6), requires_grad=True)
        
    def get_normalized_weight(self):
        """Apply constraint to ensure weights are positive and well-behaved"""
        if self.weight_constraint == 'softplus':
            weight = F.softplus(self.raw_weight)
        elif self.weight_constraint == 'exp':
            weight = torch.exp(torch.clamp(self.raw_weight, max=10.0))  # FIXED: Clamp before exp
        elif self.weight_constraint == 'softmax':
            weight = F.softmax(self.raw_weight, dim=0) * self.num_classes
        else:
            weight = F.softplus(self.raw_weight)
        
        # FIXED: Clamp final weights to prevent extreme values
        return torch.clamp(weight, min=1e-6, max=self.max_weight)
        
    def forward(self, logits, targets):
        """
        Args:
            logits: [N, C] where C is the number of classes
            targets: [N] class indices
        """
        # Get current weights
        weight = self.get_normalized_weight()
        
        # Standard focal loss implementation
        num_classes = logits.size(-1)
        targets = targets.view(-1)
        logits = logits.view(-1, num_classes)

        # Handle padding tokens
        valid_mask = (targets != self.ignore_index)
        targets = targets[valid_mask]
        logits = logits[valid_mask]

        if targets.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute focal loss
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        targets_onehot = F.one_hot(targets, num_classes=num_classes).float()

        # Per-sample probability of the correct class
        pt = (probs * targets_onehot).sum(dim=-1).clamp(min=self.eps)
        
        # Focal weight based on how hard the example is
        focal_weight = (1 - pt) ** self.gamma

        # FIXED: Manually apply class weights instead of using weight parameter in nll_loss
        # Step 1: Calculate unweighted loss 
        ce_loss = -torch.sum(targets_onehot * log_probs, dim=-1)
        
        # Step 2: Apply the learnable weights to the correct classes
        weight = weight.to(logits.device)
        class_weights = torch.sum(targets_onehot * weight.unsqueeze(0), dim=-1)
        
        # Step 3: Combine focal weight with weighted cross entropy
        loss = focal_weight * ce_loss * class_weights

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    


class BatchedUltraEfficientSpatialLoss(nn.Module):
    """
    Ultra-efficient BATCHED spatial loss that processes all samples simultaneously.
    
    Performance improvements:
    - 10-50x faster than sample-by-sample processing
    - Vectorized operations across all samples
    - Efficient GPU memory usage
    - Single forward pass for all sparse data
    """
    
    def __init__(self, window_size, baseline_ratio, center_weight=4.0, 
                 neighbor_weight=2.0, edge_weight=1.0, spatial_reg=0.15,
                 loss_type='smooth_l1'):
        super().__init__()
        
        # Store configuration
        self.window_size = window_size
        self.baseline_ratio = baseline_ratio
        self.center_weight = center_weight
        self.neighbor_weight = neighbor_weight
        self.edge_weight = edge_weight
        self.spatial_reg = spatial_reg
        self.loss_type = loss_type
        
        # Pre-compute sigma for Gaussian smoothing
        self.sigma = max(1.0, window_size / 2.0)
        
        # Pre-compute neighbor threshold
        self.neighbor_threshold = window_size * 0.3
    
    def generate_batched_windows(self, sparse_data, max_token_length):
        """
        Generate all window data in a single FULLY VECTORIZED operation
        
        Args:
            sparse_data: [N, 3] - (batch_idx, snp_token, feature_value)
            max_token_length: int - maximum sequence length
            
        Returns:
            window_info: dict containing all window data for batched processing
        """
        if sparse_data.numel() == 0:
            return None
        
        device = sparse_data.device
        batch_indices = sparse_data[:, 0].long()
        snp_tokens = sparse_data[:, 1].long()
        feature_values = sparse_data[:, 2]
        
        # Generate all window bounds
        start_tokens = torch.clamp(snp_tokens - self.window_size, min=0)
        end_tokens = torch.clamp(snp_tokens + self.window_size + 1, max=max_token_length)
        
        # Calculate window lengths for each sample
        window_lengths = end_tokens - start_tokens
        max_window_size = window_lengths.max().item()
        num_samples = len(sparse_data)
        
        if max_window_size <= 0:
            return None
        
        # === VECTORIZED POSITION GENERATION ===
        # Create a range tensor [0, 1, 2, ..., max_window_size-1]
        position_offsets = torch.arange(max_window_size, device=device).unsqueeze(0)  # [1, max_window_size]
        
        # Expand start_tokens to [num_samples, 1] and add offsets
        start_tokens_expanded = start_tokens.unsqueeze(1)  # [num_samples, 1]
        all_positions = start_tokens_expanded + position_offsets  # [num_samples, max_window_size]
        
        # === VECTORIZED MASK GENERATION ===
        # Create mask for valid positions within each window
        window_lengths_expanded = window_lengths.unsqueeze(1)  # [num_samples, 1]
        window_masks = position_offsets < window_lengths_expanded  # [num_samples, max_window_size]
        
        # === VECTORIZED DISTANCE CALCULATION ===
        # Calculate distances from SNP token for all positions
        snp_tokens_expanded = snp_tokens.unsqueeze(1)  # [num_samples, 1]
        all_distances = torch.abs(all_positions - snp_tokens_expanded).float()  # [num_samples, max_window_size]
        
        # === VECTORIZED SMOOTHING WEIGHTS ===
        # Apply Gaussian smoothing to all distances at once
        gaussian_weights = torch.exp(-0.5 * (all_distances / self.sigma) ** 2)
        smoothing_weights = self.baseline_ratio + (1.0 - self.baseline_ratio) * gaussian_weights
        
        # Set full weight at SNP positions (distance == 0)
        smoothing_weights = torch.where(all_distances == 0, 1.0, smoothing_weights)
        
        # Apply feature values to get targets
        feature_values_expanded = feature_values.unsqueeze(1)  # [num_samples, 1]
        all_targets = feature_values_expanded * smoothing_weights  # [num_samples, max_window_size]
        
        # === VECTORIZED IMPORTANCE WEIGHTS ===
        all_importance = torch.where(
            all_distances == 0,
            self.center_weight,
            torch.where(
                all_distances <= self.neighbor_threshold,
                self.neighbor_weight,
                self.edge_weight
            )
        )
        
        # === APPLY MASKS TO ZERO OUT INVALID POSITIONS ===
        # Zero out invalid positions (beyond window bounds)
        all_positions = torch.where(window_masks, all_positions, -1)
        all_distances = torch.where(window_masks, all_distances, float('inf'))
        all_targets = torch.where(window_masks, all_targets, 0.0)
        all_importance = torch.where(window_masks, all_importance, 0.0)
        
        return {
            'batch_indices': batch_indices,
            'positions': all_positions,
            'targets': all_targets,
            'importance': all_importance,
            'masks': window_masks,
            'distances': all_distances
        }
    
    def generate_smoothing_weights(self, distances):
        """Generate Gaussian smoothing weights (vectorized)"""
        # Gaussian decay
        gaussian_weights = torch.exp(-0.5 * (distances / self.sigma) ** 2)
        
        # Apply baseline floor
        weights = self.baseline_ratio + (1.0 - self.baseline_ratio) * gaussian_weights
        
        # Full weight at SNP position
        weights = torch.where(distances == 0, 1.0, weights)
        
        return weights
    
    def generate_importance_weights(self, distances):
        """Generate importance weights (vectorized)"""
        importance = torch.where(
            distances == 0,
            self.center_weight,
            torch.where(
                distances <= self.neighbor_threshold,
                self.neighbor_weight,
                self.edge_weight
            )
        )
        return importance
    
    def extract_batched_predictions(self, predictions, window_info):
        """
        Extract predictions for all windows in a single operation (fully vectorized)
        Args:
            predictions: [batch_size, seq_len, 1] - model predictions
            window_info: dict - window information from generate_batched_windows
        Returns:
            extracted_preds: [num_samples, max_window_size] - predictions for all windows
        """
        batch_indices = window_info['batch_indices']  # [num_samples]
        positions = window_info['positions']          # [num_samples, max_window_size]
        masks = window_info['masks']                  # [num_samples, max_window_size]

        num_samples, max_window_size = positions.shape

        # Clamp positions to valid range
        positions_clamped = torch.clamp(positions, 0, predictions.shape[1] - 1)

        # Expand batch_indices to match positions shape
        batch_indices_expanded = batch_indices.unsqueeze(1).expand(-1, max_window_size)

        # Gather predictions using advanced indexing
        preds = predictions[batch_indices_expanded, positions_clamped, 0]  # [num_samples, max_window_size]

        # Mask out invalid positions
        preds = preds * masks

        return preds
    
    def compute_batched_spatial_regularization(self, predictions, window_info):
        """
        Compute spatial regularization for all windows simultaneously (vectorized)
        Args:
            predictions: [num_samples, max_window_size] - extracted predictions
            window_info: dict - window information
        Returns:
            spatial_penalty: scalar - total spatial regularization penalty
        """
        if self.spatial_reg <= 0:
            return torch.tensor(0.0, device=predictions.device)

        masks = window_info['masks']  # [num_samples, max_window_size]
        # Find valid lengths for each sample
        valid_lengths = masks.sum(dim=1)  # [num_samples]
        # Only keep samples with at least 2 valid positions
        valid_mask = valid_lengths > 1
        if not valid_mask.any():
            return torch.tensor(0.0, device=predictions.device)

        # Gather valid predictions for all samples
        preds = predictions[valid_mask]  # [num_valid_samples, max_window_size]
        masks_valid = masks[valid_mask]  # [num_valid_samples, max_window_size]
        lengths = valid_lengths[valid_mask]  # [num_valid_samples]

        # For each sample, get only the valid predictions
        # Pad to max_window_size, but mask out invalid entries
        preds_masked = preds * masks_valid  # zeros where invalid

        # To compute adjacent differences, shift and subtract
        diffs = preds_masked[:, 1:] - preds_masked[:, :-1]
        # Mask out diffs that cross invalid boundaries
        valid_diffs_mask = masks_valid[:, 1:] & masks_valid[:, :-1]
        diffs = diffs * valid_diffs_mask

        # Sum squared diffs and normalize by valid length
        sum_sq_diffs = (diffs ** 2).sum(dim=1)
        norm = valid_diffs_mask.sum(dim=1).clamp(min=1)
        penalties = sum_sq_diffs / norm

        # Average over all valid samples
        spatial_penalty = penalties.mean()
        return spatial_penalty
    
    def forward(self, predictions, sparse_data, feature_key, max_token_length):
        """FIXED to handle empty features gracefully"""
        # Check if we have data for this feature
        if (feature_key not in sparse_data or 
            len(sparse_data[feature_key]) == 0 or 
            sparse_data[feature_key].numel() == 0):
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        feature_data = sparse_data[feature_key]  # [N, 3] tensor
        
        # Additional safety check
        if feature_data.numel() == 0 or feature_data.shape[0] == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Ensure data is on correct device and has correct format
        if feature_data.device != predictions.device:
            feature_data = feature_data.to(predictions.device)
        
        # Ensure we have the right format [N, 3]
        if feature_data.shape[1] != 3:
            print(f"Warning: {feature_key} has shape {feature_data.shape}, expected [N, 3]")
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # === BATCHED PROCESSING ===
        try:
            # 1. Generate all window information in one go
            window_info = self.generate_batched_windows(feature_data, max_token_length)
            if window_info is None:
                return torch.tensor(0.0, device=predictions.device, requires_grad=True)
            
            # 2. Extract predictions for all windows
            extracted_preds = self.extract_batched_predictions(predictions, window_info)
            
            # 3. Compute loss for all samples simultaneously
            targets = window_info['targets']
            importance = window_info['importance']
            masks = window_info['masks']
            
            # Apply mask to get only valid positions
            valid_preds = extracted_preds[masks]
            valid_targets = targets[masks]
            valid_importance = importance[masks]
            
            if len(valid_preds) == 0:
                return torch.tensor(0.0, device=predictions.device, requires_grad=True)
            
            # Compute base loss (vectorized across all valid positions)
            if self.loss_type == 'mse':
                base_loss = F.mse_loss(valid_preds, valid_targets, reduction='none')
            elif self.loss_type == 'smooth_l1':
                base_loss = F.smooth_l1_loss(valid_preds, valid_targets, reduction='none')
            elif self.loss_type == 'huber':
                base_loss = F.huber_loss(valid_preds, valid_targets, reduction='none')
            else:
                base_loss = F.smooth_l1_loss(valid_preds, valid_targets, reduction='none')
            
            # Apply importance weighting (vectorized)
            weighted_loss = (base_loss * valid_importance).mean()
            
            # Add spatial regularization
            if self.spatial_reg > 0:
                spatial_penalty = self.compute_batched_spatial_regularization(extracted_preds, window_info)
                weighted_loss = weighted_loss + self.spatial_reg * spatial_penalty
            
            return weighted_loss
            
        except Exception as e:
            # Graceful fallback for any processing errors
            print(f"Warning: Error in BatchedUltraEfficientSpatialLoss for {feature_key}: {e}")
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    

class SNPConsequenceFocalLoss(nn.Module):
    """
    Focal regression loss specifically designed for SNP consequence prediction.
    - Heavily focuses on SNP positions (should have high consequence)
    - Allows/encourages non-SNP positions to be zero/low
    - Minimal penalty for predicting zero at non-SNP positions
    """
    def __init__(self, snp_alpha=5.0, snp_gamma=2.0, background_weight=0.01, 
                 snp_threshold=0.1, background_threshold=0.05):
        super().__init__()
        self.snp_alpha = snp_alpha              # High importance for SNP positions
        self.snp_gamma = snp_gamma              # Focus on difficult SNP predictions
        self.background_weight = background_weight  # Very low weight for background
        self.snp_threshold = snp_threshold      # What counts as "high consequence"
        self.background_threshold = background_threshold  # What counts as "effectively zero"
        
    def forward(self, predictions, targets, snp_mask=None):
        """
        Args:
            predictions: [N] - predicted consequence scores (0-1)
            targets: [N] - true consequence scores (0-1)  
            snp_mask: [N] - Boolean mask indicating SNP positions (optional)
        """
        if snp_mask is None:
            # If no SNP mask provided, assume high targets indicate SNP positions
            snp_mask = targets > self.snp_threshold
        
        background_mask = ~snp_mask
        total_loss = 0.0
        
        # === SNP positions: High importance, focal weighting ===
        if snp_mask.any():
            snp_preds = predictions[snp_mask]
            snp_targets = targets[snp_mask]
            
            # Base MSE loss for SNP positions
            mse_loss = F.mse_loss(snp_preds, snp_targets, reduction='none')
            
            # Focal weighting: focus on difficult predictions
            abs_error = torch.abs(snp_preds - snp_targets)
            focal_weight = torch.pow(abs_error + 1e-8, self.snp_gamma)
            
            # Target magnitude weighting: higher consequence = higher importance
            magnitude_weight = torch.pow(snp_targets + 1e-8, self.snp_alpha)
            
            # Combined weighted loss for SNPs
            snp_loss = (mse_loss * focal_weight * magnitude_weight).mean()
            total_loss += snp_loss
        
        # === Background positions: Minimal penalty for being low/zero ===
        if background_mask.any():
            bg_preds = predictions[background_mask]
            bg_targets = targets[background_mask]
            
            # Only penalize if prediction is significantly higher than target
            # This encourages low predictions at background positions
            excess_prediction = torch.clamp(bg_preds - bg_targets, min=0.0)
            background_loss = (excess_prediction ** 2).mean() * self.background_weight
            
            total_loss += background_loss
        
        return total_loss
    


class FalsePositiveFocalPenalty(nn.Module):
    """
    Focal-style penalty specifically designed for false positive reduction.
    Uses confidence-aware weighting to focus on high-confidence false positives.
    """
    def __init__(self, gamma=2.0, alpha=2.0, confidence_threshold=0.5):
        super().__init__()
        self.gamma = gamma  # Focusing parameter (higher = more focus on confident FPs)
        self.alpha = alpha  # Scale parameter
        self.confidence_threshold = confidence_threshold  # Threshold for "confident" predictions
        
    def forward(self, fp_confidences):
        """
        Args:
            fp_confidences: Confidence scores for false positive predictions [N]
        """
        if len(fp_confidences) == 0:
            return torch.tensor(0.0, device=fp_confidences.device)
        
        # Focal weighting: higher confidence FPs get exponentially more penalty
        # (1 - confidence) makes low confidence have higher weight in standard focal
        # We invert this: confidence^gamma makes high confidence have higher weight
        focal_weight = torch.pow(fp_confidences, self.gamma)
        
        # Base penalty is the confidence itself (higher confidence = higher base penalty)
        base_penalty = fp_confidences
        
        # Combined focal penalty
        penalty = base_penalty * focal_weight * self.alpha
        
        return penalty.mean()

