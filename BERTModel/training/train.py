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
from BERTModel.data.dataset_loading import collate_fn
from BERTModel.training.evaluate import *
from BERTModel.training.setup_deepspeed import setup_training



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

    global_step = 0
    best_eval_metrics = None
    best_val_f1 = -float('inf') 
    total_steps = 0  

    if local_rank <= 0:
        print("Training Model with DeepSpeed...")
        global_start_time = time.time()
        steps_done = 0
        for epoch in range(config.training.num_epochs):
            epoch_indices = curriculum_sampler.get_indices_for_epoch(epoch)
            epoch_dataset = torch.utils.data.Subset(multi_dataset, epoch_indices)
            steps_per_epoch = len(epoch_dataset) // config.training.batch_size
            total_steps += steps_per_epoch


    
    for epoch in range(config.training.num_epochs):
        model_engine.train()
        total_loss = 0    
            
        
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
            
            print(f"\n Epoch {epoch} Curriculum Status:")
            print(f"   Expected ratio: {current_ratio:.3f}")
            print(f"   Actual ratio: {actual_ratio:.3f}")
            print(f"   Samples/Gpu: {pos_count:,} pos, {neg_count:,} neg")
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
      
            # === Compute dense and sparse losses ===
            outputs = model_engine(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            dense_losses = compute_dense_losses(outputs, batch, device, config)
            sparse_losses = compute_sparse_losses(outputs, batch, device, config)

            # === Joint Loss ===
            loss = (
                dense_losses['loss_snp'] +
                dense_losses['loss_snp_hnm'] +
                dense_losses['loss_snp_conf'] +
                dense_losses['loss_cons'] +
                dense_losses['loss_snp_precision'] +
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


            # == Extract outputs ==
            logits_token_snp = outputs['token_snp']

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

            # === Log Step Data ===
            if local_rank <= 0:
                steps_done += 1
                elapsed_global = time.time() - global_start_time
                avg_steps_per_sec_global = steps_done / elapsed_global if elapsed_global > 0 else 1
                steps_left_global = total_steps - steps_done
                eta_seconds_global = int(steps_left_global / avg_steps_per_sec_global) if avg_steps_per_sec_global > 0 else 0
                eta_seconds_global = eta_seconds_global ** 1.02
                eta_h = int(eta_seconds_global // 3600)
                eta_m = int((eta_seconds_global % 3600) // 60)
                eta_s = int(eta_seconds_global % 60)
                eta_string = f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
                eta_h = eta_seconds_global / 3600.0
              

                progress_bar.set_postfix({
                    'total_loss': f'{loss.item() * config.training.grad_accum_steps:.3f}',
                    'token_snp': f'{dense_losses["loss_snp"].item():.3f}',
                    'PrecT': f'{precision_token.item():.3f}',
                    'ETA': eta_string
                })

                progress_bar.update(1)

                # Log to wandb
                wandb.log({
                    'loss/loss_snp': dense_losses['loss_snp'].detach().item(),
                    'loss/loss_snp_fp': dense_losses['loss_snp_fp'].detach().item(),
                    'loss/loss_snp_hnm': dense_losses['loss_snp_hnm'].detach().item(),
                    'loss/loss_cons': dense_losses['loss_cons'].detach().item(),
                    'loss/loss_snp_precision': dense_losses['loss_snp_precision'].detach().item(),
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
                    'training/eta_hours': eta_h,
                    'training/learning_rate': optimizer.param_groups[0]['lr'],
                    'training/step': global_step,
                    'precision/token': precision_token.item(),
                })

            # === BACKWARD PASS ===
            model_engine.backward(loss)

            # === Model engine Step ===
            model_engine.step()
            global_step += 1
            total_loss += loss.detach().item()        
            

        # === Epoch Validation & Evaluation ===
        eval_metrics = evaluate(model_engine.module, test_loader, device)
        
        # Log validation metrics to wandb (only on rank 0)
        if local_rank <= 0:
            wandb.log({
                'epoch/loss': eval_metrics['val_loss'],
                'epoch/token_accuracy': eval_metrics['token_metrics']['accuracy'],
                'epoch/token_precision': eval_metrics['token_metrics']['precision'],
                'epoch/token_recall': eval_metrics['token_metrics']['recall'],
                'epoch/token_f1': eval_metrics['token_metrics']['f1'],
                'epoch/epoch': epoch + 1
            })

            # === Save best model checkpoint ===
            current_f1 = eval_metrics['token_metrics']['f1']
            current_loss = eval_metrics['val_loss']

            # === Save based on F1 ===
            if (epoch + 1) % 10 == 0 and current_f1 > best_val_f1:
                best_val_f1 = current_f1
                checkpoint_path = f"DeepSNP3D_epoch{epoch+1}_f1{current_f1:.4f}.pth"
                torch.save(model_engine.module.state_dict(), checkpoint_path)
                print(f"Saved new best model to {checkpoint_path} (F1: {current_f1:.4f} Loss: {current_loss:.4f})")


            progress_bar.close()

            # Print epoch summary
            avg_loss = total_loss / steps_per_epoch
            tqdm.write(f"Epoch {epoch+1}/{config.training.num_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {eval_metrics['val_loss']:.4f} | Token F1: {eval_metrics['token_metrics']['f1']:.4f}")


    # === Final Evaluation ===
    if local_rank <= 0:
        print("Training complete! Logging final results...")

        # Run a final evaluation after training is complete
        print("Running final evaluation to collect metrics...")
        final_eval_metrics = evaluate(model_engine.module, test_loader, device)
        best_eval_metrics = final_eval_metrics  # Use the final metrics

        # Log confusion matrices and confidence plots after training
        print("Logging confusion matrices and confidence plots...")
        token_cm = best_eval_metrics['token_metrics']['confusion_matrix']
        token_class_names = ["Non-SNP", "SNP"]


        # Create and log token confusion matrix
        token_cm_fig = plot_confusion_matrix(
            token_cm, 
            classes=token_class_names,
            title='Token-Level Confusion Matrix'
        )
        wandb.log({"final/token_confusion_matrix_plot": wandb.Image(token_cm_fig)})
        plt.close(token_cm_fig)


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


        # Create and log confidence box plots
        conf_fig = create_confidence_boxplots(token_conf_correct, token_conf_incorrect)
        wandb.log({"final/confidence_boxplots": wandb.Image(conf_fig)})
        plt.close(conf_fig)

        # Create and log confidence histograms
        hist_fig = create_confidence_histograms(token_conf_correct, token_conf_incorrect)
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
