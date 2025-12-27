import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from model import OneDCNN
from generate import MFCCDataset
import os
TRAIN_NPY_DIR = "./train_npy_time"
TEST_NPY_DIR = "./test_a_npy_time"
SUBMIT_CSV_PATH = "./submit_time.csv"

# Prepare training dataset
train_dataset = MFCCDataset(data_dir=TRAIN_NPY_DIR,is_test= False)
test_dataset = MFCCDataset(data_dir=TEST_NPY_DIR,is_test= True)
label_to_index = {label: i for i, label in enumerate(sorted(set(label for _, label in train_dataset)))}
index_to_label = {i: label for label, i in label_to_index.items()}

def collate_fn(batch):
    X, y = zip(*batch)
    X = torch.stack(X)
    y = torch.tensor([label_to_index[label] for label in y])
    return X, y

def train_model():
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    val_accuracies = []
    test_predictions = np.zeros((len(test_dataset), len(label_to_index)))

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.arange(len(train_dataset)), [label for _, label in train_dataset])
    ):
        print(f"\n==== Fold {fold+1} / {n_splits} ====")

        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, collate_fn=collate_fn)

        model = OneDCNN(input_shape=(344, 40), num_classes=len(label_to_index))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        best_val_acc = 0.0
        for epoch in range(1, 101):
            model.train()
            train_loss = 0
            correct = 0
            total = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * y_batch.size(0)
                _, predicted = outputs.max(1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

            train_acc = correct / total
            print(f"Epoch {epoch} - Train Loss: {train_loss/total:.4f}, Train Acc: {train_acc:.4f}")

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    outputs = model(X_val)
                    _, predicted = outputs.max(1)
                    val_correct += (predicted == y_val).sum().item()
                    val_total += y_val.size(0)

            val_acc = val_correct / val_total
            print(f"Validation Accuracy: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"best_model_fold_{fold}.pth")

        val_accuracies.append(best_val_acc)

        # Test predictions aggregation
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        for i, (X_test_batch, _) in enumerate(test_loader):
            outputs = model(X_test_batch)
            preds = outputs.cpu().detach().numpy()
            test_predictions[i * 128:i * 128 + len(preds)] += preds / n_splits

    final_labels = [index_to_label[pred.argmax()] for pred in test_predictions]
    pd.DataFrame({"name": os.listdir(TEST_NPY_DIR), "label": final_labels}).to_csv(SUBMIT_CSV_PATH, index=False)

train_model()