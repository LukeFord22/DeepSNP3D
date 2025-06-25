import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, BertConfig
from BERTModel.model.Model import DNABERTClassifier
from BERTModel.training.evaluate import create_token_highlighting, create_html_token_tables, plot_confusion_matrix, create_confidence_boxplots, create_confidence_histograms
from BERTModel.data import get_eval_dataloader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import webbrowser
from datetime import datetime


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model components
local_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "model", "DNABERT-2-117M"
)

if not os.path.exists(local_path):
    raise FileNotFoundError(f"Model path not found: {local_path}")

# Load tokenizer and model configuration
tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
config = BertConfig.from_pretrained(local_path, trust_remote_code=True)
base_model = AutoModel.from_config(config, trust_remote_code=True)

# Initialize custom classifier
model = DNABERTClassifier(base_model)
model.to(device)

# Load trained weights if available
model_path = r"C:\Users\lhfor\AppData\Local\Programs\Python\Python38-32\DeepSNP3D.pth"
if os.path.exists(model_path):
    print(f"Loading trained weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
else:
    print(f"No trained weights found at {model_path}, using untrained model")

model.eval()  # Set to evaluation mode
print("Model loaded successfully")


def comprehensive_evaluate_model(
    model, 
    dataloader, 
    tokenizer, 
    device, 
    max_samples=20, 
    max_len=128,
    output_dir="evaluation_results",
    open_browser=True
):
    """
    Comprehensive model evaluation with pleasing visualizations and reports
    
    Args:
        model: Trained DNABERTClassifier model
        dataloader: DataLoader for evaluation data
        tokenizer: Tokenizer for the model
        device: PyTorch device
        max_samples: Maximum number of samples to visualize
        max_len: Maximum token length per sample
        output_dir: Directory to save results
        open_browser: Whether to automatically open the HTML report
    
    Returns:
        dict: Comprehensive evaluation results
    """
    print("🔬 Starting comprehensive model evaluation...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Run full evaluation to get metrics
    print("📊 Computing evaluation metrics...")
    eval_results = evaluate_full_dataset(model, dataloader, device)
    
    # 2. Create token highlighting samples
    print("🎨 Creating token visualization samples...")
    samples = create_token_highlighting(
        model, dataloader, device, tokenizer, 
        max_samples=max_samples, max_len=max_len
    )
    
    # 3. Generate HTML report
    print("📝 Generating HTML report...")
    html_report = create_comprehensive_html_report(
        eval_results, samples, timestamp
    )
    
    # 4. Save HTML report
    html_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_report)
    print(f"📄 HTML report saved to: {html_path}")
    
    # 5. Generate and save plots
    print("📈 Generating visualization plots...")
    plot_paths = create_evaluation_plots(eval_results, output_dir, timestamp)
    
    # 6. Create summary statistics
    summary_stats = create_summary_statistics(eval_results, samples)
    
    # 7. Save summary JSON
    import json
    summary_path = os.path.join(output_dir, f"evaluation_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=2, default=str)
    print(f"📋 Summary statistics saved to: {summary_path}")
    
    # 8. Open browser if requested
    if open_browser:
        try:
            webbrowser.open(f"file://{os.path.abspath(html_path)}")
            print("🌐 Opening HTML report in browser...")
        except:
            print("⚠️  Could not open browser automatically")
    
    print("✅ Comprehensive evaluation complete!")
    
    return {
        'eval_results': eval_results,
        'samples': samples,
        'summary_stats': summary_stats,
        'html_path': html_path,
        'plot_paths': plot_paths,
        'summary_path': summary_path
    }


def evaluate_full_dataset(model, dataloader, device):
    """Run full evaluation on the dataset"""
    model.eval()
    loss_total = 0
    
    # Accumulators
    all_token_preds, all_token_labels = [], []
    token_conf_scores = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 50 == 0:
                print(f"  Processing batch {batch_idx + 1}/{len(dataloader)}")
            
            # Move data to device
            device_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    device_batch[k] = v.to(device)
                else:
                    device_batch[k] = v
            
            # Forward pass
            outputs = model(device_batch['input_ids'], device_batch['attention_mask'])
            
            # Extract outputs
            logits_token_snp = outputs['token_snp']
            logits_token_conf = outputs['token_conf']
            
            # Process token-level outputs
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
            import torch.nn as nn
            token_loss = nn.CrossEntropyLoss(ignore_index=-100)(logits_token_flat, labels_flat)
            loss_total += token_loss.item()
    
    # Compute metrics
    val_loss = loss_total / len(dataloader)
    
    # Token-level metrics
    token_accuracy = accuracy_score(all_token_labels, all_token_preds)
    token_precision, token_recall, token_f1, _ = precision_recall_fscore_support(
        all_token_labels, all_token_preds, average='weighted', zero_division=0
    )
    token_cm = confusion_matrix(all_token_labels, all_token_preds)
    
    # Confidence analysis
    correct_mask = np.array(all_token_preds) == np.array(all_token_labels)
    token_conf_correct = np.array(token_conf_scores)[correct_mask]
    token_conf_incorrect = np.array(token_conf_scores)[~correct_mask]
    
    return {
        'val_loss': val_loss,
        'token_metrics': {
            'accuracy': token_accuracy,
            'precision': token_precision,
            'recall': token_recall,
            'f1': token_f1,
            'confusion_matrix': token_cm
        },
        'token_conf_correct': token_conf_correct,
        'token_conf_incorrect': token_conf_incorrect,
        'token_preds': all_token_preds,
        'token_labels': all_token_labels,
        'token_conf_scores': token_conf_scores
    }


def create_comprehensive_html_report(eval_results, samples, timestamp):
    """Create a comprehensive HTML report with all results"""
    
    # Extract metrics
    metrics = eval_results['token_metrics']
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>DNABERT-2 Evaluation Report - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
            .metric-card {{ background-color: #ffffff; border: 1px solid #ddd; border-radius: 8px; padding: 15px; text-align: center; }}
            .metric-value {{ font-size: 2em; font-weight: bold; color: #28a745; }}
            .metric-label {{ color: #6c757d; margin-top: 5px; }}
            .section {{ margin-bottom: 30px; }}
            .section h2 {{ color: #343a40; border-bottom: 2px solid #dee2e6; padding-bottom: 10px; }}
            table {{ border-collapse: collapse; width: 100%; font-size: 11px; }}
            th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .snp-correct {{ background-color: #d4edda; }}
            .snp-incorrect {{ background-color: #f8d7da; }}
            .snp-missed {{ background-color: #fff3cd; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧬 DNABERT-2 Model Evaluation Report</h1>
            <p><strong>Generated:</strong> {timestamp}</p>
            <p><strong>Total Samples Analyzed:</strong> {len(samples)}</p>
        </div>
        
        <div class="section">
            <h2>📊 Overall Performance Metrics</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value">{metrics['accuracy']:.1%}</div>
                    <div class="metric-label">Token Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics['precision']:.1%}</div>
                    <div class="metric-label">Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics['recall']:.1%}</div>
                    <div class="metric-label">Recall</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics['f1']:.1%}</div>
                    <div class="metric-label">F1 Score</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🔬 Detailed Token Analysis</h2>
            <p><strong>Legend:</strong></p>
            <ul>
                <li><span style="background-color: #d4edda; padding: 2px 6px;">Green</span> - Correctly predicted SNP</li>
                <li><span style="background-color: #f8d7da; padding: 2px 6px;">Red</span> - Incorrectly predicted SNP (False Positive)</li>
                <li><span style="background-color: #fff3cd; padding: 2px 6px;">Yellow</span> - Missed SNP (False Negative)</li>
                <li><span style="background-color: #ffffff; padding: 2px 6px; border: 1px solid #ddd;">White</span> - Correctly predicted Normal</li>
            </ul>
    """
    
    # Add token tables
    html += create_html_token_tables(samples)
    
    html += """
        </div>
        
        <div class="section">
            <h2>📈 Statistical Summary</h2>
            <p>For detailed plots and confidence analysis, see the generated PNG files in the output directory.</p>
        </div>
        
        <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; color: #6c757d; text-align: center;">
            <p>Generated by DNABERT-2 Evaluation Pipeline</p>
        </footer>
    </body>
    </html>
    """
    
    return html


def create_evaluation_plots(eval_results, output_dir, timestamp):
    """Generate and save evaluation plots"""
    plot_paths = {}
    
    # 1. Confusion Matrix
    if len(eval_results['token_metrics']['confusion_matrix']) > 0:
        fig = plot_confusion_matrix(
            eval_results['token_metrics']['confusion_matrix'],
            classes=['Normal', 'SNP'],
            title='Token-Level Confusion Matrix'
        )
        cm_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
        fig.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_paths['confusion_matrix'] = cm_path
    
    # 2. Confidence Box Plots
    if len(eval_results['token_conf_correct']) > 0 and len(eval_results['token_conf_incorrect']) > 0:
        fig = create_confidence_boxplots(
            eval_results['token_conf_correct'],
            eval_results['token_conf_incorrect']
        )
        box_path = os.path.join(output_dir, f"confidence_boxplots_{timestamp}.png")
        fig.savefig(box_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_paths['confidence_boxplots'] = box_path
    
    # 3. Confidence Histograms
    if len(eval_results['token_conf_correct']) > 0 and len(eval_results['token_conf_incorrect']) > 0:
        fig = create_confidence_histograms(
            eval_results['token_conf_correct'],
            eval_results['token_conf_incorrect']
        )
        hist_path = os.path.join(output_dir, f"confidence_histograms_{timestamp}.png")
        fig.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_paths['confidence_histograms'] = hist_path
    
    return plot_paths


def create_summary_statistics(eval_results, samples):
    """Create summary statistics dictionary"""
    metrics = eval_results['token_metrics']
    
    # Sample-level analysis
    correct_predictions = sum(1 for s in samples if s['has_mutation'] == s['predicted_mutation'])
    sample_accuracy = correct_predictions / len(samples) if samples else 0
    
    # Token-level analysis
    total_tokens = len(eval_results['token_preds'])
    snp_tokens = sum(eval_results['token_labels'])
    normal_tokens = total_tokens - snp_tokens
    
    return {
        'timestamp': datetime.now().isoformat(),
        'dataset_stats': {
            'total_samples': len(samples),
            'total_tokens': total_tokens,
            'snp_tokens': int(snp_tokens),
            'normal_tokens': int(normal_tokens),
            'snp_ratio': float(snp_tokens / total_tokens) if total_tokens > 0 else 0
        },
        'sample_level': {
            'accuracy': sample_accuracy,
            'correct_predictions': correct_predictions
        },
        'token_level': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1'])
        },
        'confidence_stats': {
            'correct_mean': float(np.mean(eval_results['token_conf_correct'])) if len(eval_results['token_conf_correct']) > 0 else 0,
            'incorrect_mean': float(np.mean(eval_results['token_conf_incorrect'])) if len(eval_results['token_conf_incorrect']) > 0 else 0,
            'overall_mean': float(np.mean(eval_results['token_conf_scores'])) if len(eval_results['token_conf_scores']) > 0 else 0
        }
    }


# Example usage function
def run_evaluation_example(dataloader):
    """Example function showing how to use the comprehensive evaluation"""
    results = comprehensive_evaluate_model(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        device=device,
        max_samples=20,
        max_len=128,
        output_dir="evaluation_results",
        open_browser=True
    )
    
    print("\n📋 Evaluation Summary:")
    print(f"Token Accuracy: {results['summary_stats']['token_level']['accuracy']:.1%}")
    print(f"Sample Accuracy: {results['summary_stats']['sample_level']['accuracy']:.1%}")
    print(f"F1 Score: {results['summary_stats']['token_level']['f1_score']:.1%}")
    print(f"Results saved to: {results['html_path']}")
    
    return results


if __name__ == "__main__":
    print("🧬 DNABERT-2 Comprehensive Evaluation Tool Ready!")

    dataloader = get_eval_dataloader(tokenizer, batch_size=16, max_len=128)

    # Run the evaluation
    results = run_evaluation_example(dataloader)