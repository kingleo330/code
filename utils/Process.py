import os
import re
import random
import numpy as np
import pandas as pd
import scipy.signal as signal
import cv2
import torch
from collections import defaultdict
from utils.config import HP


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('(\d+)', s)]


def process_emg_data(folder_path=HP.data_path,
                     emg_keys=HP.emg_keys,
                     win_size=HP.win_size,
                     step_size=HP.step_size):
    all_wins_emg = []
    all_wins_label = []
    all_window_sources = []

    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    files.sort(key=natural_sort_key)

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            continue

        pd_data = pd.read_csv(file_path)

        mode = pd_data['Mode'].values

        emg_data = pd_data[emg_keys].values

        wins_emg, wins_label, window_file_sources = [], [], []
        num_windows = max((len(mode) - win_size) // step_size + 1, 0)

        for i in range(num_windows):
            start_index = i * step_size
            end_index = start_index + win_size
            if end_index > len(mode):
                break

            is_unique = len(np.unique(mode[start_index:end_index])) == 1

            if is_unique:
                wins_emg.append(emg_data[start_index:end_index])
                wins_label.append(mode[start_index + (win_size // 2)])
                window_file_sources.append(filename)

        all_wins_emg.extend(wins_emg)
        all_wins_label.extend(wins_label)
        all_window_sources.extend(window_file_sources)

    if len(all_wins_emg) == 0:
        return [], [], []

    return (
        np.stack(all_wins_emg, axis=1),
        np.array(all_wins_label),
        np.array(all_window_sources)
    )


def generate_spectrograms_per_window(emg_data, mode_data, masker=None, augment_prob=0):
    num_windows, window_length, num_channels = emg_data.shape
    image_data_list = []

    for idx in range(num_windows):
        window = emg_data[idx]

        if masker is not None and random.random() < augment_prob:
            augmented = masker(window.T)
            if isinstance(augmented, torch.Tensor):
                augmented = augmented.cpu().numpy()
            window = augmented.T

        label = mode_data[idx]

        for channel in range(num_channels):
            signal_ch = window[:, channel]

            mean_val = np.mean(signal_ch)
            std_val = np.std(signal_ch)
            if std_val == 0:
                signal_norm = signal_ch - mean_val
            else:
                signal_norm = (signal_ch - mean_val) / std_val

            _, _, Sxx = signal.spectrogram(
                signal_norm, fs=HP.Fs, nperseg=HP.nperseg, noverlap=HP.noverlap, window='hamming'
            )

            Sxx_log = np.log10(Sxx + 1e-10)
            Sxx_min, Sxx_max = Sxx_log.min(), Sxx_log.max()

            if Sxx_max > Sxx_min:
                Sxx_norm = 255 * (Sxx_log - Sxx_min) / (Sxx_max - Sxx_min)
            else:
                Sxx_norm = np.full_like(Sxx_log, 127)

            Sxx_norm = Sxx_norm.astype(np.uint8)
            spectrogram_rgb = cv2.applyColorMap(Sxx_norm, cv2.COLORMAP_JET)
            spectrogram_rgb = cv2.cvtColor(spectrogram_rgb, cv2.COLOR_BGR2RGB)
            image_data_list.append((spectrogram_rgb, label, idx, channel))

    return image_data_list


def merge_images_per_window(image_data_list):
    window_dict = defaultdict(list)

    for img, label, idx, ch in image_data_list:
        window_dict[idx].append((ch, img, label))

    merged_images = []

    for idx, channel_imgs in window_dict.items():
        channel_imgs.sort(key=lambda x: x[0])
        images = [item[1] for item in channel_imgs]
        label = channel_imgs[1][2]

        merged_image = cv2.hconcat(images)
        merged_image = np.clip(merged_image, 0, 255).astype(np.uint8)
        merged_image = merged_image.astype(np.float32) / 255.0

        merged_images.append((merged_image, label, idx))

    return merged_images