import torch
class Config:

    data_path = r"your data"
    save_dir = r"your save"

    sub_nums = ['AB156', 'AB185', 'AB186', 'AB188', 'AB189', 'AB190', 'AB191', 'AB192', 'AB193', 'AB194']


    emg_keys = [
        'Left_TA', 'Left_MG', 'Left_SOL', 'Left_BF', 'Left_ST', 'Left_VL', 'Left_RF',
        'Right_TA', 'Right_MG', 'Right_SOL', 'Right_BF', 'Right_ST', 'Right_VL', 'Right_RF'
    ]
    # Cross Validation
    n_splits = 5

    # Windowing
    win_size = 300
    step_size = 20

    # STFT
    Fs = 1000
    nperseg = 64
    noverlap = 32

    # Training Hyperparameters
    batch_size = 64
    lr = 3.75e-4
    weight_decay = 0.025
    epochs = 100
    test_size = 0.2
    random_state = 42
    normalize = True

    # Model Architecture
    base_dim = 12
    mlp_ratio = 2
    drop_path_rate = 0.0
    class_num = 7

    # Scheduler
    scheduler_patience = 3
    scheduler_threshold = 1e-4
    scheduler_cooldown = 1
    min_lr = 1e-6

    # Early Stopping
    early_patience = 10

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Augmentation Probabilities
    augment_prob = 0.25



HP = Config()
