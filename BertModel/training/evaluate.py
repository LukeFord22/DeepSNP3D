import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

#=======================================================================================================
#                                                                               Evaluation Calculations
#=======================================================================================================


def evaluate(model, dataloader, device):
    model.eval()
    loss_total = 0

    # === Accumulators ===
    all_token_preds, all_token_labels = [], []
    token_conf_scores = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Add defensive type checking
            if not isinstance(batch, dict):
                print(f"Batch {batch_idx} is not a dictionary but {type(batch)}")
                if isinstance(batch, (np.number, np.ndarray, int, float)):
                    print(f"Received numerical batch: {batch}")
                    continue  # Skip this batch

    with torch.no_grad():
        for batch in dataloader:
            # Only move tensor data to device, handle non-tensor data separately
            device_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    device_batch[k] = v.to(device)
                else:
                    device_batch[k] = v  # Keep non-tensor data as is


            outputs = model(device_batch['input_ids'], device_batch['attention_mask'])


            # === Extract outputs ===
            logits_token_snp = outputs['token_snp']
            logits_token_conf = outputs['token_conf']
            
            # === Process token-level outputs ===
            batch_size, seq_len, n_features = logits_token_snp.shape
            logits_token_flat = logits_token_snp.reshape(-1, n_features)
            labels_flat = device_batch['snp_label'].view(-1)
            
            # Create mask for valid tokens
            valid_token_mask = (labels_flat != -100)
            
            # Get predictions and labels for valid tokens
            token_preds = logits_token_flat.argmax(dim=1)[valid_token_mask].cpu().numpy()
            token_labels = labels_flat[valid_token_mask].cpu().numpy()
            
            # Extract token-level confidence
            token_conf = torch.sigmoid(logits_token_conf).view(-1)[valid_token_mask].cpu().numpy()
            
            # Append results
            all_token_preds.extend(token_preds.tolist())
            all_token_labels.extend(token_labels.tolist())
            token_conf_scores.extend(token_conf.tolist())

            # Track loss
            token_loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_token_flat, labels_flat)
            loss_total += token_loss.item()

    # === Compute Metrics ===
    val_loss = loss_total / len(dataloader)
    
    # Token-level metrics
    token_accuracy = accuracy_score(all_token_labels, all_token_preds)
    token_precision, token_recall, token_f1, _ = precision_recall_fscore_support(
        all_token_labels, all_token_preds, average='weighted', zero_division=0
    )
    token_cm = confusion_matrix(all_token_labels, all_token_preds)

    return {
        'val_loss': val_loss,
        'token_metrics': {
            'accuracy': token_accuracy, 
            'precision': token_precision, 
            'recall': token_recall, 
            'f1': token_f1,
            'confusion_matrix': token_cm
        },
        'token_conf': token_conf_scores,
        'token_preds': all_token_preds,
        'token_labels': all_token_labels,
    }




# Create custom confusion matrix plots with matplotlib
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks and label them with the respective list entries
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes, yticklabels=classes,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            
    fig.tight_layout()
    return fig




# == confidence box plots ==
def create_confidence_boxplots(token_conf_correct, token_conf_incorrect):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Token-level confidence box plot
    token_data = [token_conf_correct, token_conf_incorrect]
    token_labels = ['Correct', 'Incorrect']
    token_colors = ['#2ecc71', '#e74c3c']
    
    box1 = ax1.boxplot(token_data, labels=token_labels, patch_artist=True)
    for patch, color in zip(box1['boxes'], token_colors):
        patch.set_facecolor(color)
    
    ax1.set_title('Token-Level Confidence', fontsize=14)
    ax1.set_ylabel('Confidence Score', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add individual points with jitter for better visualization
    for i, data in enumerate(token_data):
        # Add some random jitter to the x-position
        x = np.random.normal(i+1, 0.08, size=len(data))
        ax1.scatter(x, data, alpha=0.3, s=10, c=token_colors[i])
    
    # Add statistics as text
    stats_token = f"Correct: mean={np.mean(token_conf_correct):.3f}, median={np.median(token_conf_correct):.3f}, n={len(token_conf_correct)}\n"
    stats_token += f"Incorrect: mean={np.mean(token_conf_incorrect):.3f}, median={np.median(token_conf_incorrect):.3f}, n={len(token_conf_incorrect)}"
    ax1.text(0.5, -0.15, stats_token, transform=ax1.transAxes, ha='center', fontsize=10)
    
    plt.tight_layout(pad=3.0)
    plt.suptitle('Confidence Distribution: Correct vs. Incorrect Predictions', fontsize=16, y=1.05)
    
    return fig



# == histograms for more detailed distribution ==
def create_confidence_histograms(token_conf_correct, token_conf_incorrect):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Token confidence histogram
    bins = np.linspace(0, 1, 21)
    ax1.hist([token_conf_correct, token_conf_incorrect], bins=bins, stacked=True, 
                color=['#2ecc71', '#e74c3c'], alpha=0.7, label=['Correct', 'Incorrect'])
    ax1.set_title('Token-Level Confidence Distribution', fontsize=14)
    ax1.set_xlabel('Confidence Score', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    return fig



# == Highlighting visualization ==
def create_token_highlighting(model, dataloader, device, tokenizer, max_samples=10, max_len=128):
    model.eval()
    samples = []
    
    sample_count = 0
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= max_samples:
                break
                
            # Only move tensor data to device, handle non-tensor data separately
            device_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    device_batch[k] = v.to(device)
                elif k == 'sparse_data':
                    # Handle sparse data - move tensor parts to device
                    device_sparse = {}
                    for sparse_k, sparse_v in v.items():
                        if isinstance(sparse_v, torch.Tensor):
                            device_sparse[sparse_k] = sparse_v.to(device)
                        else:
                            device_sparse[sparse_k] = sparse_v
                    device_batch[k] = device_sparse
                else:
                    device_batch[k] = v  # Keep non-tensor data as is
            
            # Get predictions
            outputs = model(device_batch['input_ids'], device_batch['attention_mask'])
            
            # Extract predictions and confidence
            token_preds = outputs['token_snp'].argmax(dim=-1)
            token_conf = outputs['token_conf'].squeeze(-1)
            
            # Extract epigenetic score predictions (these exist in your model)
            rna_preds = outputs['token_rna'].squeeze(-1) if 'token_rna' in outputs else None
            cons_preds = outputs['token_cons'].squeeze(-1) if 'token_cons' in outputs else None
            gerp_preds = outputs['token_gerp'].squeeze(-1) if 'token_gerp' in outputs else None
            dnase_preds = outputs.get('token_dnase', torch.zeros_like(token_conf)) if 'token_dnase' in outputs else None
            h3k27ac_preds = outputs.get('token_h3k27ac', torch.zeros_like(token_conf)) if 'token_h3k27ac' in outputs else None
            h3k4me3_preds = outputs.get('token_h3k4me3', torch.zeros_like(token_conf)) if 'token_h3k4me3' in outputs else None
            h3k36me3_preds = outputs.get('token_h3k36me3', torch.zeros_like(token_conf)) if 'token_h3k36me3' in outputs else None
            h3k9me3_preds = outputs.get('token_h3k9me3', torch.zeros_like(token_conf)) if 'token_h3k9me3' in outputs else None
            h3k27me3_preds = outputs.get('token_h3k27me3', torch.zeros_like(token_conf)) if 'token_h3k27me3' in outputs else None
            h2afz_preds = outputs.get('token_h2afz', torch.zeros_like(token_conf)) if 'token_h2afz' in outputs else None

            # Process each sequence in the batch
            for i in range(len(device_batch['input_ids'])):
                if sample_count >= max_samples:
                    break
                    
                sample_count += 1
                
                # Get input tokens and valid mask
                input_ids = device_batch['input_ids'][i]
                attention_mask = device_batch['attention_mask'][i]
                snp_labels = device_batch['snp_label'][i]
                cons_labels = device_batch['cons_label'][i]
                
                # Create sparse label lookups for this sample - FIXED KEYS
                sparse_data = device_batch['sparse_data']
                rna_lookup = create_sparse_lookup(sparse_data, i, 'rna_data')
                gerp_lookup = create_sparse_lookup(sparse_data, i, 'gerp_data')
                dnase_lookup = create_sparse_lookup(sparse_data, i, 'dnase_data')
                h3k27ac_lookup = create_sparse_lookup(sparse_data, i, 'h3k27ac_data')
                h3k4me3_lookup = create_sparse_lookup(sparse_data, i, 'h3k4me3_data')
                h3k36me3_lookup = create_sparse_lookup(sparse_data, i, 'h3k36me3_data')
                h3k9me3_lookup = create_sparse_lookup(sparse_data, i, 'h3k9me3_data')
                h3k27me3_lookup = create_sparse_lookup(sparse_data, i, 'h3k27me3_data')
                h2afz_lookup = create_sparse_lookup(sparse_data, i, 'h2afz_data')
                
                # Only consider tokens with attention mask 1
                valid_indices = attention_mask.bool()
                
                # Truncate if too long
                if valid_indices.sum() > max_len:
                    count = 0
                    for j in range(len(valid_indices)):
                        if valid_indices[j]:
                            count += 1
                        if count > max_len:
                            valid_indices[j:] = False
                            break
                
                # Get tokens and metadata
                tokens = tokenizer.convert_ids_to_tokens(input_ids[valid_indices])
                preds = token_preds[i, valid_indices].cpu().numpy()
                confs = token_conf[i, valid_indices].cpu().numpy()
                true_snp_labels = snp_labels[valid_indices].cpu().numpy()
                true_cons_labels = cons_labels[valid_indices].cpu().numpy()

                predicted_mutation = int((preds == 1).any())
                # DNA-level confidence: max token confidence for predicted SNPs, else max overall
                if (preds == 1).any():
                    dna_confidence = float(confs[preds == 1].max())
                else:
                    dna_confidence = float(confs.max()) if len(confs) > 0 else 0.0
                
                # Get epigenetic predictions
                rna_pred_vals = rna_preds[i, valid_indices].cpu().numpy() if rna_preds is not None else None
                cons_pred_vals = cons_preds[i, valid_indices].cpu().numpy() if cons_preds is not None else None
                gerp_pred_vals = gerp_preds[i, valid_indices].cpu().numpy() if gerp_preds is not None else None
                dnase_pred_vals = dnase_preds[i, valid_indices].cpu().numpy() if dnase_preds is not None else None
                h3k27ac_pred_vals = h3k27ac_preds[i, valid_indices].cpu().numpy() if h3k27ac_preds is not None else None
                h3k4me3_pred_vals = h3k4me3_preds[i, valid_indices].cpu().numpy() if h3k4me3_preds is not None else None
                h3k36me3_pred_vals = h3k36me3_preds[i, valid_indices].cpu().numpy() if h3k36me3_preds is not None else None
                h3k9me3_pred_vals = h3k9me3_preds[i, valid_indices].cpu().numpy() if h3k9me3_preds is not None else None
                h3k27me3_pred_vals = h3k27me3_preds[i, valid_indices].cpu().numpy() if h3k27me3_preds is not None else None
                h2afz_pred_vals = h2afz_preds[i, valid_indices].cpu().numpy() if h2afz_preds is not None else None

                # Create token metadata
                token_data = []
                valid_token_idx = 0
                for original_idx in range(len(valid_indices)):
                    if not valid_indices[original_idx]:
                        continue
                        
                    if valid_token_idx >= len(tokens):
                        break
                        
                    token = tokens[valid_token_idx]
                    pred = preds[valid_token_idx]
                    conf = confs[valid_token_idx]
                    snp_label = true_snp_labels[valid_token_idx]
                    cons_label = true_cons_labels[valid_token_idx]
                    
                    if snp_label == -100:  # Skip padding/ignored
                        valid_token_idx += 1
                        continue
                    
                    # Get epigenetic predictions
                    rna_pred = float(rna_pred_vals[valid_token_idx]) if rna_pred_vals is not None else 0.0
                    cons_pred = float(cons_pred_vals[valid_token_idx]) if cons_pred_vals is not None else 0.0
                    gerp_pred = float(gerp_pred_vals[valid_token_idx]) if gerp_pred_vals is not None else 0.0
                    dnase_pred = float(dnase_pred_vals[valid_token_idx]) if dnase_pred_vals is not None else 0.0
                    h3k27ac_pred = float(h3k27ac_pred_vals[valid_token_idx]) if h3k27ac_pred_vals is not None else 0.0
                    h3k4me3_pred = float(h3k4me3_pred_vals[valid_token_idx]) if h3k4me3_pred_vals is not None else 0.0
                    h3k36me3_pred = float(h3k36me3_pred_vals[valid_token_idx]) if h3k36me3_pred_vals is not None else 0.0
                    h3k9me3_pred = float(h3k9me3_pred_vals[valid_token_idx]) if h3k9me3_pred_vals is not None else 0.0
                    h3k27me3_pred = float(h3k27me3_pred_vals[valid_token_idx]) if h3k27me3_pred_vals is not None else 0.0
                    h2afz_pred = float(h2afz_pred_vals[valid_token_idx]) if h2afz_pred_vals is not None else 0.0

                    # Get sparse true values using lookup - FIXED to use original sequence position
                    rna_true = rna_lookup.get(original_idx, -100.0)
                    gerp_true = gerp_lookup.get(original_idx, -100.0)
                    dnase_true = dnase_lookup.get(original_idx, -100.0)
                    h3k27ac_true = h3k27ac_lookup.get(original_idx, -100.0)
                    h3k4me3_true = h3k4me3_lookup.get(original_idx, -100.0)
                    h3k36me3_true = h3k36me3_lookup.get(original_idx, -100.0)
                    h3k9me3_true = h3k9me3_lookup.get(original_idx, -100.0)
                    h3k27me3_true = h3k27me3_lookup.get(original_idx, -100.0)
                    h2afz_true = h2afz_lookup.get(original_idx, -100.0)
                    cons_true = float(cons_label) if cons_label != -100.0 else -100.0
                        
                    token_data.append({
                        "token": token,
                        "pred": int(pred),
                        "conf": float(conf),
                        "label": int(snp_label),
                        "correct": bool(pred == snp_label),
                        "rna_pred": rna_pred,
                        "rna_true": rna_true,
                        "cons_pred": cons_pred,
                        "cons_true": cons_true,
                        "gerp_pred": gerp_pred,
                        "gerp_true": gerp_true,
                        "dnase_pred": dnase_pred,
                        "dnase_true": dnase_true,
                        "h3k27ac_pred": h3k27ac_pred,
                        "h3k27ac_true": h3k27ac_true,
                        "h3k4me3_pred": h3k4me3_pred,
                        "h3k4me3_true": h3k4me3_true,
                        "h3k36me3_pred": h3k36me3_pred,
                        "h3k36me3_true": h3k36me3_true,
                        "h3k9me3_pred": h3k9me3_pred,
                        "h3k9me3_true": h3k9me3_true,
                        "h3k27me3_pred": h3k27me3_pred,
                        "h3k27me3_true": h3k27me3_true,
                        "h2afz_pred": h2afz_pred,
                        "h2afz_true": h2afz_true

                    })
                    
                    valid_token_idx += 1
                
                samples.append({
                    "id": f"sample_{sample_count}",
                    "tokens": token_data,
                    "has_mutation": bool(device_batch['has_mutation'][i].item()),
                    "predicted_mutation": bool(predicted_mutation),
                    "dna_confidence": dna_confidence
                })
    
    return samples



def create_sparse_lookup(sparse_data, batch_idx, feature_key):
    """Create a lookup dictionary for sparse features for a specific batch item"""
    lookup = {}
    
    if feature_key in sparse_data:
        feature_tensor = sparse_data[feature_key]  # Shape: [N, 3] where cols are [batch_idx, token_idx, value]
        
        if len(feature_tensor) > 0:
            # Find rows that belong to this batch item
            batch_mask = (feature_tensor[:, 0] == batch_idx)
            if batch_mask.any():
                batch_rows = feature_tensor[batch_mask]
                
                # Create lookup dictionary: token_position -> feature_value
                for row in batch_rows:
                    token_pos = int(row[1].item())
                    value = float(row[2].item())
                    lookup[token_pos] = value
    
    return lookup




# Create HTML table visualizations for Wandb
def create_html_token_tables(samples):
    html_tables = []
    
    for sample in samples:
        # Start table with headers - EXPANDED
        html = f"""<h3>Sample {sample['id']}</h3>
        <p>Has mutation: {sample['has_mutation']} | Predicted: {sample['predicted_mutation']} | DNA Confidence: {sample['dna_confidence']:.1%}</p>
        <table style="border-collapse: collapse; width: 100%; font-size: 11px;">
        <tr style="background-color: #f2f2f2;">
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">Token</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">SNP Pred</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">Conf</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">SNP True</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">RNA P</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">RNA T</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">CONS P</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">CONS T</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">GERP P</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">GERP T</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">DNase P</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">DNase T</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">H3K27ac P</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">H3K27ac T</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">H3K4me3 P</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">H3K4me3 T</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">H3K36me3 P</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">H3K36me3 T</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">H3K9me3 P</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">H3K9me3 T</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">H3K27me3 P</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">H3K27me3 T</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">H2AFZ P</th>
            <th style="padding: 4px; text-align: left; border: 1px solid #ddd;">H2AFZ T</th>
        </tr>
        """
        
        # Add token rows
        for t in sample['tokens']:
            # Color based on prediction accuracy
            bg_color = "#ffffff"  # Default white
            
            if t['pred'] == 1:  # Predicted SNP
                if t['label'] == 1:  # Correctly predicted SNP
                    bg_color = "#ccffcc"  # Light green
                else:  # False positive
                    bg_color = "#ffcccc"  # Light red
            elif t['label'] == 1:  # Missed SNP (false negative)
                bg_color = "#ffffcc"  # Light yellow
            
            # Format values with better handling of sparse data
            conf_pct = f"{t['conf']*100:.1f}%"
            
            def format_value(val, default="N/A"):
                if val == -100.0:
                    return default
                elif abs(val) < 0.001:
                    return "0"
                else:
                    return f"{val:.3f}"
            
            # Format all predictions and true values
            rna_pred_str = format_value(t.get('rna_pred', 0.0))
            cons_pred_str = format_value(t.get('cons_pred', 0.0))  
            gerp_pred_str = format_value(t.get('gerp_pred', 0.0))
            dnase_pred_str = format_value(t.get('dnase_pred', 0.0))
            h3k27ac_pred_str = format_value(t.get('h3k27ac_pred', 0.0))
            h3k4me3_pred_str = format_value(t.get('h3k4me3_pred', -100.0))
            h3k36me3_pred_str = format_value(t.get('h3k36me3_pred', -100.0))
            h3k9me3_pred_str = format_value(t.get('h3k9me3_pred', -100.0))
            h3k27me3_pred_str = format_value(t.get('h3k27me3_pred', -100.0))
            h2afz_pred_str = format_value(t.get('h2afz_pred', -100.0))

            rna_true_str = format_value(t.get('rna_true', -100.0))
            cons_true_str = format_value(t.get('cons_true', -100.0))
            gerp_true_str = format_value(t.get('gerp_true', -100.0))
            dnase_true_str = format_value(t.get('dnase_true', -100.0))
            h3k27ac_true_str = format_value(t.get('h3k27ac_true', -100.0))
            h3k4me3_true_str = format_value(t.get('h3k4me3_true', -100.0))
            h3k36me3_true_str = format_value(t.get('h3k36me3_true', -100.0))
            h3k9me3_true_str = format_value(t.get('h3k9me3_true', -100.0))
            h3k27me3_true_str = format_value(t.get('h3k27me3_true', -100.0))
            h2afz_true_str = format_value(t.get('h2afz_true', -100.0))

            html += f"""
            <tr style="background-color: {bg_color};">
                <td style="padding: 4px; text-align: left; border: 1px solid #ddd; font-family: monospace;">{t['token']}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{"SNP" if t['pred'] == 1 else "Norm"}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{conf_pct}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{"SNP" if t['label'] == 1 else "Norm"}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{rna_pred_str}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{rna_true_str}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{cons_pred_str}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{cons_true_str}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{gerp_pred_str}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{gerp_true_str}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{dnase_pred_str}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{dnase_true_str}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{h3k27ac_pred_str}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{h3k27ac_true_str}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{h3k4me3_pred_str}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{h3k4me3_true_str}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{h3k36me3_pred_str}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{h3k36me3_true_str}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{h3k9me3_pred_str}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{h3k9me3_true_str}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{h3k27me3_pred_str}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{h3k27me3_true_str}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{h2afz_pred_str}</td>
                <td style="padding: 4px; text-align: center; border: 1px solid #ddd;">{h2afz_true_str}</td>

            </tr>
            """
        
        # Close table
        html += "</table><br><hr><br>"
        html_tables.append(html)
    
    return "".join(html_tables)



def save_model(model_engine, path='DeepSNP3D.pth'):
    # Save the model to the current directory
    import os
    
    # Make sure we use a path in the current working directory
    if os.path.isabs(path):
        # Extract just the filename from the absolute path
        path = os.path.basename(path)
    
    # Save the underlying model (not the DeepSpeed engine)
    torch.save(model_engine.module.state_dict(), path)
    print(f"Model saved locally to {path}")