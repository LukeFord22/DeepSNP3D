import torch
import time
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import sys

# Add parent directories to Python path for absolute imports
script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.dirname(script_dir)
sys.path.insert(0, training_dir)
sys.path.insert(0, script_dir)

# Import from packages using absolute imports
from Loss import *
from BertModel.data.dataset_loading import collate_fn
from BertModel.training.evaluate import *
from BertModel.training.setup_deepspeed import setup_training



def train_model(config=None):
    """Main training function for DNA SNP detection with DeepSpeed."""

    # Setup training components with DeepSpeed
    setup_results = setup_training(config)
    (model_engine, optimizer, scheduler, _, test_loader, 
     device, local_rank, dna_tokenizer, multi_dataset, curriculum_sampler, 
     world_size, rank) = setup_results

    # Store runtime info in config for compatibility
    config.deepspeed_runtime = {
        'model_engine': model_engine,
        'device': device,
        'rank': local_rank,
        'world_size': world_size,
        'dna_tokenizer': dna_tokenizer,
        'test_loader': test_loader,
        'multi_dataset': multi_dataset,
        'curriculum_sampler': curriculum_sampler
    }
    optimizer = model_engine.optimizer
    scheduler = model_engine.lr_scheduler


    # Only initialize wandb on rank 0
    if local_rank <= 0:
        wandb.init(
            project="dna-snp-detection-deepspeed",
            config={
                "batch_size": config.training.batch_size,
                "epochs": config.training.num_epochs,
                "learning_rate": config.training.learning_rate,
                "grad_accum_steps": config.training.grad_accum_steps,
                "train_data_fraction": config.training.train_data_fraction,
                "deepspeed": True,
                "local_rank": local_rank,
                "world_size": torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            }
        )

    #=======================================================================================================
    #                                                                                          Training Loop
    #=======================================================================================================

    epoch_timing_history = {}

    # scheduler function
    def delayed_consistency_scheduler(epoch):
        if epoch < config.training.consistency_start_epoch:
            return 0.0  # No consistency loss before threshold epoch
        else:
            # Calculate warmup based on epochs after the threshold
            effective_epoch = epoch - config.training.consistency_start_epoch
            return min(1.0, effective_epoch / config.training.consistency_warmup_epochs)

    consistency_scheduler = delayed_consistency_scheduler

    global_step = 0
    best_eval_metrics = None

    if local_rank <= 0:
        print("Training Model with DeepSpeed...")
        epoch_start_time = time.time()
    for epoch in range(config.training.num_epochs):
        model_engine.train()
        total_loss = 0
        consistency_weight = consistency_scheduler(epoch)        
            
        
        # === Get curriculum-specific indices and sampler ===
        epoch_indices = curriculum_sampler.get_indices_for_epoch(epoch)
        epoch_sampler = curriculum_sampler.get_sampler_for_epoch(epoch)  # This handles distribution
        current_ratio = curriculum_sampler.get_current_ratio(epoch)
        
        # Create subset dataset for this epoch
        epoch_dataset = torch.utils.data.Subset(multi_dataset, epoch_indices)

        # Create DataLoader with curriculum sampler (NO shuffle!)
        train_loader = DataLoader(
            epoch_dataset,
            batch_size=config.training.batch_size,
            sampler=epoch_sampler, 
            shuffle=False,         
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=0,  
            drop_last=True
        )

        
        # Debug curriculum progress
        if local_rank <= 0:
            pos_count = sum(1 for idx in epoch_indices 
                        if multi_dataset.sample_index[idx][0] == 'positive')
            neg_count = len(epoch_indices) - pos_count
            actual_ratio = pos_count / len(epoch_indices) if len(epoch_indices) > 0 else 0
            
            print(f"\n🎯 Epoch {epoch} Curriculum Status:")
            print(f"   Expected ratio: {current_ratio:.3f}")
            print(f"   Actual ratio: {actual_ratio:.3f}")
            print(f"   Samples: {pos_count:,} pos, {neg_count:,} neg")
            print(f"   Total batches: {len(train_loader)}")
            
            wandb.log({
                'curriculum/expected_ratio': current_ratio,
                'curriculum/actual_ratio': actual_ratio,
                'curriculum/pos_samples': pos_count,
                'curriculum/neg_samples': neg_count,
                'curriculum/total_samples': len(epoch_indices),
                'curriculum/batches_per_epoch': len(train_loader),
                'training/epoch': epoch
            })
        
        # Training loop for this epoch
        steps_per_epoch = len(train_loader)
        if local_rank <= 0:
            progress_bar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{config.training.num_epochs}", leave=True)

        for step, batch in enumerate(train_loader):
      
            outputs = model_engine(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            current_step = epoch * len(train_loader) + step
            # Extract outputs

            dense_losses = compute_dense_losses(outputs, batch, device, config, consistency_weight)
            sparse_losses = compute_sparse_losses(outputs, batch, device, config)

            # === Joint Loss ===
            loss = (
                dense_losses['loss_snp'] +
                dense_losses['loss_snp_hnm'] +
                dense_losses['loss_snp_conf'] +
                dense_losses['loss_snp_fp'] +
                dense_losses['loss_cons'] +
                dense_losses['loss_dna'] +
                dense_losses['loss_dna_hnm'] +
                dense_losses['loss_dna_conf'] +
                dense_losses['loss_dna_fp'] +
                dense_losses['loss_consistency'] +
                sparse_losses['loss_rna'] +
                sparse_losses['loss_dnase'] +
                sparse_losses['loss_gerp'] +
                sparse_losses['loss_h3k27ac'] +
                sparse_losses['loss_h3k4me3'] +
                sparse_losses['loss_h3k36me3'] +
                sparse_losses['loss_h3k9me3'] +
                sparse_losses['loss_h3k27me3'] +
                sparse_losses['loss_h2afz']
            ) / config.training.grad_accum_steps

            # === Calculate ETA (only on rank 0) ===
            if local_rank <= 0:
                elapsed_this_epoch = time.time() - epoch_start_time
                steps_per_second = (step + 1) / max(1.0, elapsed_this_epoch)
                remaining_steps_this_epoch = steps_per_epoch - (step + 1)
                current_ratio = curriculum_sampler.get_current_ratio(epoch)
                
                # Track timing for different curriculum phases
                curriculum_phase = "pre_curriculum" if epoch < config.training.curriculum_start else f"ratio_{current_ratio:.1f}"
                if curriculum_phase not in epoch_timing_history:
                    epoch_timing_history[curriculum_phase] = []
                epoch_timing_history[curriculum_phase].append(steps_per_second)
                
                # Calculate remaining steps accounting for curriculum changes
                remaining_steps_total = remaining_steps_this_epoch
                total_weighted_time = remaining_steps_this_epoch / steps_per_second
                
                # For each remaining epoch, calculate expected steps and time
                for future_epoch in range(epoch + 1, config.training.num_epochs):
                    # Determine curriculum ratio for future epoch
                    if future_epoch < config.training.curriculum_start:
                        future_ratio = 1.0
                        future_phase = "pre_curriculum"
                    else:
                        future_ratio = curriculum_sampler.get_current_ratio(future_epoch)
                        future_phase = f"ratio_{future_ratio:.1f}"
                    
                    # Calculate dataset size for this future epoch
                    total_dataset_size = len(multi_dataset)
                    train_data_fraction = config.training.train_data_fraction
                    base_train_size = int(total_dataset_size * train_data_fraction)
                    future_dataset_size = int(base_train_size * future_ratio)
                    future_steps_per_epoch = future_dataset_size // config.training.batch_size

                    # Estimate processing speed for this curriculum phase
                    if future_phase in epoch_timing_history and len(epoch_timing_history[future_phase]) > 0:
                        # Use historical average for this curriculum phase
                        estimated_speed = np.mean(epoch_timing_history[future_phase])
                    else:
                        # Fallback to current speed
                        estimated_speed = steps_per_second
                    
                    remaining_steps_total += future_steps_per_epoch
                    total_weighted_time += future_steps_per_epoch / estimated_speed
                
                # Calculate ETA using weighted time estimation
                eta_seconds = total_weighted_time
                eta_hours1 = eta_seconds / 3600.0
                eta_minutes = int(eta_seconds // 60)
                eta_seconds = int(eta_seconds % 60)
                eta_hours = int(eta_minutes // 60)
                eta_minutes = int(eta_minutes % 60)
                eta_string = f"{eta_hours:02d}:{eta_minutes:02d}:{eta_seconds:02d}"

            # == Extract outputs ==
            logits_token_snp = outputs['token_snp']
            logits_dna_snp = outputs['dna_snp']

            # === Extract all label types ===
            snp_label = batch['snp_label'].to(device).long()

            # === logits and labels for token-level tasks ===
            attention_mask = batch['attention_mask'].to(device)
            valid_token_mask = (attention_mask == 1) & (snp_label != -100)

            # === Precision monitoring ===
            with torch.no_grad():
                # Token-level (only on valid tokens)
                if valid_token_mask.any():
                    valid_snp_logits = logits_token_snp[valid_token_mask]
                    valid_snp_labels = snp_label[valid_token_mask]
                    
                    preds_token = valid_snp_logits.argmax(dim=1)
                    tp_token = ((preds_token == 1) & (valid_snp_labels == 1)).sum()
                    fp_token = ((preds_token == 1) & (valid_snp_labels == 0)).sum()
                    precision_token = tp_token.float() / (tp_token + fp_token + 1e-8)
                else:
                    precision_token = torch.tensor(0.0, device=device)
                            
                # DNA-level
                preds_dna = logits_dna_snp.argmax(dim=1)
                true_dna = batch['has_mutation'].to(device)
                tp_dna = ((preds_dna == 1) & (true_dna == 1)).sum()
                fp_dna = ((preds_dna == 1) & (true_dna == 0)).sum()
                precision_dna = tp_dna.float() / (tp_dna + fp_dna + 1e-8)
                
                # Log both raw and smoothed precision to wandb (only on rank 0)
                if local_rank <= 0:
                    wandb.log({
                        'precision/token': precision_token.item(),
                        'precision/dna': precision_dna.item(),
                        'training/step': global_step
                    })

                    # Update tqdm progress bar
                    progress_bar.set_postfix({
                        'token_snp': f'{dense_losses["loss_snp"].item():.3f}',
                        'token_conf': f'{dense_losses["loss_snp_conf"].item():.3f}',
                        'dna_snp': f'{dense_losses["loss_dna"].item():.3f}',
                        'dna_conf': f'{dense_losses["loss_dna_conf"].item():.3f}',
                        'PrecT': f'{precision_token.item():.3f}',
                        'PrecD': f'{precision_dna.item():.3f}',
                        'ETA': eta_string
                    })

            if local_rank <= 0:
                progress_bar.update(1)

                # Log to wandb
                wandb.log({
                    'loss/loss_snp': dense_losses['loss_snp'].detach().item(),
                    'loss/loss_dna': dense_losses['loss_dna'].detach().item(),
                    'loss/loss_snp_conf': dense_losses['loss_snp_conf'].detach().item(),
                    'loss/loss_dna_conf': dense_losses['loss_dna_conf'].detach().item(),
                    'loss/loss_consistency': dense_losses['loss_consistency'].detach().item(),
                    'loss/loss_snp_fp': dense_losses['loss_snp_fp'].detach().item(),
                    'loss/loss_dna_fp': dense_losses['loss_dna_fp'].detach().item(),
                    'loss/loss_snp_hnm': dense_losses['loss_snp_hnm'].detach().item(),
                    'loss/loss_dna_hnm': dense_losses['loss_dna_hnm'].detach().item(),
                    'loss/loss_cons': dense_losses['loss_cons'].detach().item(),
                    'loss/loss_rna': sparse_losses['loss_rna'].detach().item(),
                    'loss/loss_dnase': sparse_losses['loss_dnase'].detach().item(),
                    'loss/loss_gerp': sparse_losses['loss_gerp'].detach().item(),
                    'loss/loss_h3k27ac': sparse_losses['loss_h3k27ac'].detach().item(),
                    'loss/loss_h3k4me3': sparse_losses['loss_h3k4me3'].detach().item(),
                    'loss/loss_h3k36me3': sparse_losses['loss_h3k36me3'].detach().item(),
                    'loss/loss_h3k9me3': sparse_losses['loss_h3k9me3'].detach().item(),
                    'loss/loss_h3k27me3': sparse_losses['loss_h3k27me3'].detach().item(),
                    'loss/loss_h2afz': sparse_losses['loss_h2afz'].detach().item(),
                    'loss/loss_total': loss.detach().item() * config.training.grad_accum_steps,
                    'training/eta_hours': eta_hours1,
                    'training/learning_rate': optimizer.param_groups[0]['lr'],
                    'training/consistency_weight': consistency_weight,
                    'training/step': global_step,
                    'precision/token': precision_token.item(),
                    'precision/dna': precision_dna.item(),
                })

            # === BACKWARD PASS - DeepSpeed handles this ===
            model_engine.backward(loss)

            # === DeepSpeed handles gradient accumulation and optimizer step ===
            model_engine.step()
            
            # Track gradients (only on rank 0)
            if local_rank <= 0:
                # Get all major model components for gradient tracking

                if step == 0:  # Initialize once per epoch
                    gradient_history = {name: [] for name in [
                        "base_model", "dna_attention", "token_attention",
                        "dna_backbone", "snp_backbone", "epigenetic_backbone", 
                        "dna_detection", "dna_confidence", "snp_detection", 
                        "snp_confidence", "snp_consequence", "epi_rna", 
                        "epi_dnase", "epi_h3k27ac", "epi_h3k4me3", "epi_h3k36me3",
                        "epi_h3k9me3", "epi_h3k27me3", "epi_h2afz", "epi_gerp"
                    ]}

                all_modules = {
                    # Base model
                    "base_model": model_engine.module.base,
                    
                    # Attention modules
                    "dna_attention": model_engine.module.dna_attention,
                    "token_attention": model_engine.module.token_attention,
                    
                    # Backbone networks
                    "dna_backbone": model_engine.module.dna_backbone,
                    "snp_backbone": model_engine.module.snp_backbone,
                    "epigenetic_backbone": model_engine.module.epigenetic_backbone,
                    
                    # DNA heads
                    "dna_detection": model_engine.module.dna_heads['dna_detection'],
                    "dna_confidence": model_engine.module.dna_heads['dna_confidence'],
                    
                    # SNP heads
                    "snp_detection": model_engine.module.snp_heads['snp_detection'],
                    "snp_confidence": model_engine.module.snp_heads['snp_confidence'],
                    "snp_consequence": model_engine.module.snp_heads['snp_consequence'],
                    
                    # Epigenetic heads
                    "epi_rna": model_engine.module.epigenetic_heads['EncodetotalRNA_sum'],
                    "epi_dnase": model_engine.module.epigenetic_heads['EncodeDNase_sum'],
                    "epi_h3k27ac": model_engine.module.epigenetic_heads['EncodeH3K27ac_sum'],
                    "epi_h3k4me3": model_engine.module.epigenetic_heads['EncodeH3K4me3_sum'],
                    "epi_h3k36me3": model_engine.module.epigenetic_heads['EncodeH3K36me3_sum'],
                    "epi_h3k9me3": model_engine.module.epigenetic_heads['EncodeH3K9me3_sum'],
                    "epi_h3k27me3": model_engine.module.epigenetic_heads['EncodeH3K27me3_sum'],
                    "epi_h2afz": model_engine.module.epigenetic_heads['EncodeH2AFZ_sum'],
                    "epi_gerp": model_engine.module.epigenetic_heads['gerp'],
                }

                # Initialize gradient history for all modules (add this at the beginning of train_model)
                gradient_history = {name: [] for name in all_modules.keys()}

                # Track gradients for all modules
                for name, module in all_modules.items():
                    grads = [p.grad for p in module.parameters() if p.grad is not None]
                    if grads:
                        mean_grad = torch.stack([g.abs().mean() for g in grads]).mean().item()
                        if not np.isfinite(mean_grad):
                            mean_grad = 0.0
                    else:
                        mean_grad = 0.0
                    gradient_history[name].append(mean_grad)
                    
                    # Log gradient norms to wandb
                    wandb.log({f'gradients/{name}': mean_grad, 'training/step': global_step})

            global_step += 1
            total_loss += loss.detach().item()        
            
        # === Validation & evaluation ===
        eval_metrics = evaluate(model_engine.module, test_loader, device)
        
        # Log validation metrics to wandb (only on rank 0)
        if local_rank <= 0:
            wandb.log({
                'epoch/loss': eval_metrics['val_loss'],
                'epoch/token_accuracy': eval_metrics['token_metrics']['accuracy'],
                'epoch/token_precision': eval_metrics['token_metrics']['precision'],
                'epoch/token_recall': eval_metrics['token_metrics']['recall'],
                'epoch/token_f1': eval_metrics['token_metrics']['f1'],
                'epoch/dna_accuracy': eval_metrics['dna_metrics']['accuracy'],
                'epoch/dna_precision': eval_metrics['dna_metrics']['precision'],
                'epoch/dna_recall': eval_metrics['dna_metrics']['recall'],
                'epoch/dna_f1': eval_metrics['dna_metrics']['f1'],
                'epoch/epoch': epoch + 1
            })

        # Track metrics
        prec = eval_metrics['dna_metrics']['precision']
        rec = eval_metrics['dna_metrics']['recall']
        beta = 0.5  # Favor precision slightly
        current_metric = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec + 1e-8)

        # Store the current evaluation metrics for logging
        best_eval_metrics = eval_metrics
        
        if local_rank <= 0:
            progress_bar.close()

            # Print epoch summary
            avg_loss = total_loss / steps_per_epoch
            tqdm.write(f"Epoch {epoch+1}/{config.training.num_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {eval_metrics['val_loss']:.4f} | DNA F1: {eval_metrics['dna_metrics']['f1']:.4f} | Token F1: {eval_metrics['token_metrics']['f1']:.4f}")


    # Finish the training loop
    if local_rank <= 0:
        print("Training complete! Logging final results...")

        # Run a final evaluation after training is complete
        print("Running final evaluation to collect metrics...")
        final_eval_metrics = evaluate(model_engine.module, test_loader, device)
        best_eval_metrics = final_eval_metrics  # Use the final metrics

        # Log confusion matrices and confidence plots after training
        print("Logging confusion matrices and confidence plots...")
        token_cm = best_eval_metrics['token_metrics']['confusion_matrix']
        dna_cm = best_eval_metrics['dna_metrics']['confusion_matrix']
        token_class_names = ["Non-SNP", "SNP"]


        dna_preds = best_eval_metrics['dna_preds']
        dna_class_names = ["Non-Mutated", "Mutated"]

        # Create and log token confusion matrix
        token_cm_fig = plot_confusion_matrix(
            token_cm, 
            classes=token_class_names,
            title='Token-Level Confusion Matrix'
        )
        wandb.log({"final/token_confusion_matrix_plot": wandb.Image(token_cm_fig)})
        plt.close(token_cm_fig)

        # Create and log DNA confusion matrix
        dna_cm_fig = plot_confusion_matrix(
            dna_cm,
            classes=dna_class_names,
            title='DNA-Level Confusion Matrix'
        )
        wandb.log({"final/dna_confusion_matrix_plot": wandb.Image(dna_cm_fig)})
        plt.close(dna_cm_fig)

        # Prepare data for token-level plot
        token_conf_correct = [conf for conf, pred, label in zip(
            best_eval_metrics['token_conf'], 
            best_eval_metrics['token_preds'], 
            best_eval_metrics['token_labels']
        ) if pred == label]

        token_conf_incorrect = [conf for conf, pred, label in zip(
            best_eval_metrics['token_conf'], 
            best_eval_metrics['token_preds'], 
            best_eval_metrics['token_labels']
        ) if pred != label]

        # Prepare data for DNA-level confidence plot
        dna_conf_correct = [conf for conf, pred, label in zip(
            best_eval_metrics['dna_conf'], 
            best_eval_metrics['dna_preds'], 
            best_eval_metrics['dna_labels']
        ) if pred == label]

        dna_conf_incorrect = [conf for conf, pred, label in zip(
            best_eval_metrics['dna_conf'], 
            best_eval_metrics['dna_preds'], 
            best_eval_metrics['dna_labels']
        ) if pred != label]

        # Create and log confidence box plots
        conf_fig = create_confidence_boxplots(token_conf_correct, token_conf_incorrect, dna_conf_correct, dna_conf_incorrect)
        wandb.log({"final/confidence_boxplots": wandb.Image(conf_fig)})
        plt.close(conf_fig)

        # Create and log confidence histograms
        hist_fig = create_confidence_histograms(token_conf_correct, token_conf_incorrect, dna_conf_correct, dna_conf_incorrect)
        wandb.log({"final/confidence_histograms": wandb.Image(hist_fig)})
        plt.close(hist_fig)
        
        token_samples = create_token_highlighting(
            model_engine.module, 
            test_loader, 
            device, 
            dna_tokenizer, 
            max_samples=10,
            max_len=config.training.max_token_length
        )
        html_tables = create_html_token_tables(token_samples)
        wandb.log({"final/token_html_visualization": wandb.Html(html_tables)})

        # Save final model
        save_model(model_engine)        # Finish the wandb run
        print("All metrics logged successfully!")
        wandb.finish()
