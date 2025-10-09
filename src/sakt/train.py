import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import numpy as np
from pathlib import Path


def train(
    train_data, 
    valid_data, 
    model, 
    optimizer, 
    scheduler,
    num_epochs, 
    device, 
    grad_clip=10.0,
    print_every=10, 
    patience=5,
    save_path='best_model.pt',
    use_amp=True  # Mixed precision training
):
    criterion = nn.BCEWithLogitsLoss()
    best_val_auc = 0
    patience_counter = 0
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        train_loss, train_auc = train_epoch(
            model, train_data, optimizer, criterion, 
            device, grad_clip, scaler, print_every
        )
        
        print(f"\nTraining   - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")
        
        # Validation phase
        val_loss, val_auc = validate_epoch(model, valid_data, criterion, device)
        
        print(f"Validation - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Early stopping and model saving
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_auc': val_auc,
            }, save_path)
            print(f"Best model saved (AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
    
    print(f"\nTraining complete! Best validation AUC: {best_val_auc:.4f}")
    return best_val_auc


def train_epoch(model, train_data, optimizer, criterion, device, grad_clip, scaler, print_every):
    model.train()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for i, batch in enumerate(train_data):
        skill_ids, skill_inter_ids, answer = batch[0], batch[2], batch[4]
        skill_ids = skill_ids.to(device)
        skill_inter_ids = skill_inter_ids.to(device)
        answer = answer.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(skill_ids, skill_inter_ids)
                loss = compute_loss(preds, answer, criterion)
            
            # Backward with scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(skill_ids, skill_inter_ids)
            loss = compute_loss(preds, answer, criterion)
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        # Collect predictions for AUC (computed once per epoch)
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(preds).detach().cpu())
        all_labels.append(answer.cpu())
        
        if (i + 1) % print_every == 0:
            avg_loss = total_loss / (i + 1)
            print(f"  Batch {i + 1}/{len(train_data)}: Loss = {avg_loss:.4f}")
    
    # Compute epoch-level metrics
    avg_loss = total_loss / len(train_data)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Flatten for AUC computation
    all_preds = all_preds.reshape(-1)
    all_labels = all_labels.reshape(-1)
    
    # Remove padding (assuming 0 is padding)
    mask = all_labels >= 0
    auc = roc_auc_score(all_labels[mask], all_preds[mask])
    
    return avg_loss, auc


def validate_epoch(model, valid_data, criterion, device):
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in valid_data:
            skill_ids, skill_inter_ids, answer = batch[0], batch[2], batch[4]
            skill_ids = skill_ids.to(device)
            skill_inter_ids = skill_inter_ids.to(device)
            answer = answer.to(device)
            
            preds = model(skill_ids, skill_inter_ids)
            loss = compute_loss(preds, answer, criterion)
            
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(preds).cpu())
            all_labels.append(answer.cpu())
    
    # Compute metrics
    avg_loss = total_loss / len(valid_data)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Flatten and remove padding
    all_preds = all_preds.reshape(-1)
    all_labels = all_labels.reshape(-1)
    mask = all_labels >= 0
    
    auc = roc_auc_score(all_labels[mask], all_preds[mask])
    
    return avg_loss, auc


def compute_loss(preds, targets, criterion):
    """Compute loss ignoring padding positions"""
    mask = targets >= 0  # Assuming -1 or negative values are padding
    if mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device)
    
    return criterion(preds[mask], targets[mask].float())