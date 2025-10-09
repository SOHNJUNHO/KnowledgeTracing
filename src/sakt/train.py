import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import compute_auc, compute_loss
from sklearn.metrics import roc_auc_score, accuracy_score


def train(train_data, valid_data, model, optimizer, num_epochs, batch_size, grad_clip, device, print_every=10, patience=5):
    criterion = nn.BCEWithLogitsLoss()
    best_val_auc = 0
    patience_counter = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_batches = train_data
        val_batches = valid_data

        # Training
        model.train()
        train_loss_total = 0
        train_auc_total = 0
        for i, (skill_ids, _, skill_inter_ids, _, answer) in enumerate(train_batches):
            skill_ids, _, skill_inter_ids, _, answer = skill_ids.to(device), _, skill_inter_ids.to(device), _, answer.to(device)

            preds = model(skill_ids, skill_inter_ids)
            loss = compute_loss(preds, answer, criterion)

            train_auc = compute_auc(torch.sigmoid(preds).detach().cpu(), answer.cpu())

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss_total += loss.item()
            train_auc_total += train_auc

            if (i + 1) % print_every == 0:
                print(f"Batch {i + 1}: loss = {loss.item():.4f}, train AUC = {train_auc:.4f}")

        print(f"Epoch {epoch + 1} training complete.")

        # model.train()
        # train_loss_total = 0
        # train_auc_total = 0
        # for i, (skill_ids, problem_ids, skill_inter_ids, pro_inter_ids, answer) in enumerate(train_batches):
        #     skill_ids, problem_ids, skill_inter_ids, pro_inter_ids, answer = skill_ids.to(device), problem_ids.to(device), skill_inter_ids.to(device), pro_inter_ids.to(device), answer.to(device)

        #     preds = model(skill_ids, problem_ids, skill_inter_ids, pro_inter_ids, answer)
        #     loss = compute_loss(preds, answer, criterion)

        #     train_auc = compute_auc(torch.sigmoid(preds).detach().cpu(), answer.cpu())

        #     optimizer.zero_grad()
        #     loss.backward()
        #     clip_grad_norm_(model.parameters(), grad_clip)
        #     optimizer.step()

        #     train_loss_total += loss.item()
        #     train_auc_total += train_auc

        #     if (i + 1) % print_every == 0:
        #         print(f"Batch {i + 1}: loss = {loss.item():.4f}, train AUC = {train_auc:.4f}")


        # Average training metrics for the epoch
        avg_train_loss = train_loss_total / len(train_batches)
        avg_train_auc = train_auc_total / len(train_batches)

        print(f"Average Training Loss = {avg_train_loss:.4f}, Training AUC = {avg_train_auc:.4f}")

        # Validation
        model.eval()
        val_auc_total = 0
        num_val_batches = len(val_batches)
        with torch.no_grad():
            for skill_ids, _, skill_inter_ids, _, answer in val_batches:
                skill_ids, _, skill_inter_ids, _, answer = skill_ids.to(device), _, skill_inter_ids.to(device), _, answer.to(device)
                preds = torch.sigmoid(model(skill_ids,skill_inter_ids))
                val_auc = compute_auc(preds.cpu(), answer.cpu())
                val_auc_total += val_auc

        # Average validation AUC for the epoch
        val_auc_avg = val_auc_total / num_val_batches
        print(f"Validation AUC = {val_auc_avg:.4f}")
        scheduler.step(val_auc_avg)


        # Early stopping check
        if val_auc_avg > best_val_auc:
            best_val_auc = val_auc_avg
            patience_counter = 0  # Reset counter if validation improves
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
