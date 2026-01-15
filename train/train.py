import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import GroupKFold
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import Counter

from utils.config import HP
from utils.Interfere import FTMA, EarlyStopping
from utils.Process import process_emg_data, generate_spectrograms_per_window, merge_images_per_window
from utils.dataset import Dataset
from model.model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
HP.device = device

AUG_SEED = getattr(HP, 'augment_seed', 42)
masker = FTMA(block_width=300, num_blocks=3, drop_modes=['noise', 'zero', 'drift', 'spike'],
                                  seed=AUG_SEED)


def valid(valid_loader, model, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch_x, batch_y in valid_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)
            batch_size = batch_y.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(outputs.cpu())
            all_trues.append(batch_y.cpu())

    if total_samples == 0:
        return float('nan'), 0.0

    avg_loss = total_loss / total_samples

    preds_tensor = torch.cat(all_preds, dim=0)
    trues_tensor = torch.cat(all_trues, dim=0)

    probs = torch.softmax(preds_tensor, dim=1)
    preds_label = torch.argmax(probs, dim=1).numpy()
    accuracy = (preds_label == trues_tensor.numpy()).mean()

    return avg_loss, accuracy


if __name__ == '__main__':
    for sub_num in HP.sub_nums:
        subject_dir = os.path.join(HP.data_path, sub_num)
        print(f"=== Subject {sub_num}: {subject_dir} ===")

        proc_folders = [d for d in os.listdir(subject_dir)
                        if os.path.isdir(os.path.join(subject_dir, d)) and d.startswith('Processed1')]
        if not proc_folders:
            print(f"Skipping {sub_num}: Processed* folder not found")
            continue
        processed_dir = os.path.join(subject_dir, proc_folders[0])
        print(f"Using directory: {processed_dir}")

        emg_windows, labels, window_sources = process_emg_data(
            folder_path=processed_dir,
            emg_keys=HP.emg_keys1,
            win_size=HP.win_size,
            step_size=HP.step_size
        )
        if len(emg_windows) == 0:
            print(f"Skipping {sub_num}: No window data generated")
            continue

        gkf = GroupKFold(n_splits=HP.n_splits)
        for fold, (train_idx, val_idx) in enumerate(gkf.split(emg_windows, labels, window_sources), start=1):
            print(f"\n-- {sub_num} Fold {fold}/{HP.n_splits} --")

            fold_dir = os.path.join(HP.save_dir3, sub_num, f'fold{fold}')
            os.makedirs(fold_dir, exist_ok=True)

            val_sources = sorted(set(window_sources[i] for i in val_idx))
            print("Val CSV files:", val_sources)

            with open(os.path.join(fold_dir, "val_sources.txt"), "w") as f:
                for src in val_sources:
                    f.write(src + "\n")

            train_emg = np.array([emg_windows[i] for i in train_idx])
            train_labels = [labels[i] for i in train_idx]
            train_groups = [window_sources[i] for i in train_idx]
            val_emg = np.array([emg_windows[i] for i in val_idx])
            val_labels = [labels[i] for i in val_idx]
            val_groups = [window_sources[i] for i in val_idx]

            train_image_data = generate_spectrograms_per_window(train_emg, train_labels,
                                                                  masker=masker, augment_prob=HP.augment_prob1)
            train_merged = merge_images_per_window(train_image_data)

            val_image_data = generate_spectrograms_per_window(val_emg, val_labels, masker=masker,
                                                                augment_prob=HP.augment_prob1)
            val_merged = merge_images_per_window(val_image_data)

            save_npz = os.path.join(fold_dir, "val_merged.npz")
            _images = np.stack([item[0] for item in val_merged])
            _labels = np.array([item[1] for item in val_merged])
            np.savez_compressed(save_npz, images=_images, labels=_labels)
            print(f"Saved val_merged -> {save_npz}")

            temp_loader = DataLoader(Dataset(train_merged, transform=transforms.ToTensor()),
                                     batch_size=HP.batch_size, shuffle=True, drop_last=True)
            for images, targets in temp_loader:
                print("Batch Images Shape:", images.shape)
                print("Batch Labels Shape:", targets.shape)
                break

            print("Train CSV files:", sorted(set(train_groups)))
            print("Val   CSV files:", sorted(set(val_groups)))
            print("Grouping Correct" if not set(train_groups) & set(val_groups) else "Grouping Error")
            print("Train dist:", dict(Counter([x[1] for x in train_merged])))
            print("Val   dist:", dict(Counter([x[1] for x in val_merged])))

            transform_train = transforms.ToTensor()
            transform_val = transforms.ToTensor()
            train_ds = Dataset(train_merged, transform=transform_train)
            val_ds = Dataset(val_merged, transform=transform_val)
            train_loader = DataLoader(train_ds, batch_size=HP.batch_size, shuffle=True, pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=HP.batch_size, shuffle=False, pin_memory=True, drop_last=True)

            model = TFSNet(
                base_dim=HP.base_dim,
                drop_path_rate=HP.drop_path_rate,
                mlp_ratio=HP.mlp_ratio,
                num_classes=HP.class_num,
            ).to(device)

            y_train = [x[1] for x in train_merged]
            cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            weights = torch.tensor(cw, dtype=torch.float).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
            optimizer = optim.AdamW(model.parameters(), lr=HP.lr, weight_decay=HP.weight_decay)
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5,
                patience=HP.scheduler_patience, threshold=HP.scheduler_threshold,
                cooldown=HP.scheduler_cooldown, min_lr=HP.min_lr
            )
            early_stopping = EarlyStopping(patience=HP.early_patience, mode='max')

            save_dir = os.path.join(HP.save_dir3, sub_num)
            os.makedirs(save_dir, exist_ok=True)
            best_acc = 0.0
            for epoch in range(1, HP.epochs + 1):
                model.train()
                epoch_loss, correct = 0.0, 0
                for imgs, tgts in train_loader:
                    imgs, tgts = imgs.to(device), tgts.to(device)
                    optimizer.zero_grad()
                    outs = model(imgs)
                    loss = criterion(outs, tgts)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * imgs.size(0)
                    correct += (outs.argmax(1) == tgts).sum().item()
                train_acc = correct / len(train_ds)
                train_loss = epoch_loss / len(train_ds)
                val_loss, val_acc = valid(val_loader, model, device)
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f},"
                      f" val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
                scheduler.step(val_acc)
                if val_acc > best_acc:
                    best_acc = val_acc
                if early_stopping(val_acc):
                    print(f"Early stopping at epoch {epoch}, best_acc={best_acc:.4f}")
                    break

            model_path = os.path.join(fold_dir, f"{model.__class__.__name__}_fold{fold}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved: {model_path}")
    print("All training complete.")
