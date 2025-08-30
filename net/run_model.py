import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, auc, accuracy_score, average_precision_score
from transformers import AutoModel
from net.model import CombinedModel
from datetime import datetime
from tqdm import tqdm

current_time = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

def run(device, epochs, train_loader, val_loader, train_count, val_count, learning_rate, fold=None, patience=5):
    # ----------------- Model -----------------
    model = CombinedModel()
    bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    model.to(device)
    bert_model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    os.makedirs(f"./logs/{current_time}", exist_ok=True)
    train_summary = SummaryWriter(log_dir=f"./logs/{current_time}/summary")
    log = open(f"./logs/{current_time}/training_log_fold-{fold+1}.txt", 'w')

    best_bac_acc = 0.0
    best_auc_val = 0.0
    best_epoch = 0
    patience_counter = 0  # Early stopping

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_true, train_pred_probs, train_pred_labels = [], [], []

        # ----------------- Training -----------------
        for img, texts, label, _ in tqdm(train_loader, leave=True, file=sys.stdout):
            img, label = img.to(device), label.to(device)
            texts = {k: v.squeeze(1).to(device) for k, v in texts.items()}
            
            
            texts['input_ids'] = texts['input_ids'].squeeze(1)
            texts['attention_mask'] = texts['attention_mask'].squeeze(1)
            texts['token_type_ids'] = texts['token_type_ids'].squeeze(1)
            text_features = bert_model(**texts)[0]

            outputs = model(img, text_features)
            loss = loss_func(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_true.extend(label.cpu().numpy())
            train_pred_probs.extend(F.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy())
            train_pred_labels.extend(outputs.argmax(dim=1).cpu().numpy())

        train_acc = accuracy_score(train_true, train_pred_labels)
        train_auc = auc(*roc_curve(train_true, train_pred_probs)[:2])
        train_loss /= train_count

        # ----------------- Validation -----------------
        model.eval()
        val_loss, val_correct = 0.0, 0
        TP = FP = TN = FN = 0
        val_true, val_pred_labels, val_pred_probs = [], [], []

        with torch.no_grad():
            for img, texts, label, _ in tqdm(val_loader, leave=True, file=sys.stdout):
                img, label = img.to(device), label.to(device)
                texts = {k: v.squeeze(1).to(device) for k, v in texts.items()}
                
                texts['input_ids'] = texts['input_ids'].squeeze(1)
                texts['attention_mask'] = texts['attention_mask'].squeeze(1)
                texts['token_type_ids'] = texts['token_type_ids'].squeeze(1)
                text_features = bert_model(**texts)[0]

                outputs = model(img, text_features)
                val_loss += loss_func(outputs, label).item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == label).sum().item()

                TP += ((label == 1) & (preds == 1)).sum().item()
                FP += ((label == 0) & (preds == 1)).sum().item()
                TN += ((label == 0) & (preds == 0)).sum().item()
                FN += ((label == 1) & (preds == 0)).sum().item()

                val_true.extend(label.cpu().numpy())
                val_pred_labels.extend(preds.cpu().numpy())
                val_pred_probs.extend(F.softmax(outputs, dim=1)[:, 1].cpu().numpy())

        # ----------------- Metrics -----------------
        numerator = TP * TN - FP * FN
        denominator = ((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)) ** 0.5
        mcc = numerator / denominator if denominator > 0 else 0

        fpr, tpr, _ = roc_curve(val_true, val_pred_probs)
        roc_auc_val = auc(fpr, tpr)
        average_precision_val = average_precision_score(val_true, val_pred_probs)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1_score_val = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        bac_val = (recall + specificity) / 2
        val_acc = val_correct / val_count
        val_loss /= val_count

        # ----------------- Early Stopping -----------------
        if val_acc > best_bac_acc or (val_acc == best_bac_acc and roc_auc_val > best_auc_val):
            best_bac_acc = val_acc
            best_auc_val = roc_auc_val
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), f"./logs/{current_time}/fold-{fold+1}_best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch+1}")
                break

        # ----------------- Logging -----------------
        train_summary.add_scalar("train_loss", train_loss, epoch)
        train_summary.add_scalar("val_acc", val_acc, epoch)
        train_summary.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)

        log_info = (
            f"Epoch {epoch+1} | Train Loss: {train_loss:.6f}, Acc: {train_acc*100:.2f}%, AUC: {train_auc:.6f}\n"
            f"Val Loss: {val_loss:.6f}, Acc: {val_acc*100:.2f}%, ROC AUC: {roc_auc_val:.6f}, F1: {f1_score_val:.6f}, BAC: {bac_val:.6f}, MCC: {mcc:.6f}\n"
            f"Confusion Matrix: TP:{TP}, FP:{FP}, TN:{TN}, FN:{FN}\n"
            f"Time: {time.time()-start_time:.2f}s | Best Epoch: {best_epoch+1}\n"
        )
        print(log_info)
        log.write(log_info)
