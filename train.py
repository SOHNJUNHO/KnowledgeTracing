import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from utils import compute_auc, compute_loss
from sklearn.metrics import roc_auc_score, accuracy_score


def train(train_data, valid_data, model, optimizer, num_epochs, batch_size, seq_len, grad_clip, print_every=30, patience=5):
    criterion = nn.BCEWithLogitsLoss()
    step = 0
    best_val_auc = 0
    patience_counter = 0  # For early stopping

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_batches = train_data
        val_batches = valid_data
        
        # Training
        model.train()
        train_loss_total = 0
        train_auc_total = 0     
        for i, (problem_ids, interaction_inputs, labels) in enumerate(train_batches):     #prblem_ids, interaction_inputs, answer
            problem_ids, interaction_inputs, labels = problem_ids.cuda(), interaction_inputs.cuda(), labels.cuda()

            preds = model(problem_ids, interaction_inputs)
            loss = compute_loss(preds, labels, criterion)
            train_auc = compute_auc(torch.sigmoid(preds).detach().cpu(), labels.cpu())

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1

            train_loss_total += loss.item()
            train_auc_total += train_auc

            if (i + 1) % print_every == 0:
                print(f"  Batch {i + 1}: loss = {loss.item():.4f}, train AUC = {train_auc:.4f}")

        # Average training metrics for the epoch
        avg_train_loss = train_loss_total / len(train_batches)
        avg_train_auc = train_auc_total / len(train_batches)
        print(f"  Average Training Loss = {avg_train_loss:.4f}, Training AUC = {avg_train_auc:.4f}")

        # Validation
        model.eval()
        val_auc_total = 0
        num_val_batches = len(val_batches)
        with torch.no_grad():
            for problem_ids, interaction_inputs, labels in val_batches:
                problem_ids, interaction_inputs, labels = problem_ids.cuda(), interaction_inputs.cuda(), labels.cuda()
                preds = torch.sigmoid(model(problem_ids, interaction_inputs))
                val_auc = compute_auc(preds.cpu(), labels.cpu())
                val_auc_total += val_auc

        # Average validation AUC for the epoch
        val_auc_avg = val_auc_total / num_val_batches
        print(f"  Validation AUC = {val_auc_avg:.4f}")

        # Early stopping check
        if val_auc_avg > best_val_auc:
            best_val_auc = val_auc_avg
            patience_counter = 0  # Reset counter if validation improves
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
