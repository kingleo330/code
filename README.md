TFS-Net: An Interpretable Lower-Limb SEMG Recognition Network
This repository contains the official PyTorch implementation of the paper: "An Interpretable Lower-Limb SEMG Recognition Network Driven by Time-Frequency Masking and Multi-Scale CNN".

TFS-Net is a lightweight and robust deep learning framework designed for lower-limb motion recognition using surface electromyography (sEMG) signals. It converts multi-channel sEMG signals into time-frequency representations (STFT) and employs a novel architecture to achieve high accuracy and interpretability.

TFS_Net_Project/
│
├── model/                  # Model Definitions
│   ├── __init__.py
│   ├── layers.py           # Basic layers (ConvBN, SEBlock, BlurPool)
│   ├── blocks.py           # Core blocks (DirectionalBlock, SGBlock, SAFM)
│   ├── attention.py        # Attention modules (CHM)
│   └── model.py            # Main TFS-Net architecture
│
├── utils/                  # Utilities
│   ├── config.py           # Hyperparameters & Paths
│   ├── Process.py          # Data preprocessing & STFT generation
│   ├── Interfere.py        # FTMA Augmentation & EarlyStopping
│   ├── dataset.py          # PyTorch Dataset class
├── train/
│   └── train.py           # Script for robustness training (With interference)
└── README.md

This project uses the ENABL3S dataset. Please download the dataset from the following link:
https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2018.00014/full
