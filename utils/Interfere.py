import random
import numpy as np
import torch

class FTMA:
    def __init__(self, block_width, num_blocks, drop_modes=None, seed=None):
        self.block_width = block_width
        self.num_blocks = num_blocks
        self.drop_modes = drop_modes if drop_modes is not None else ['zero']
        self.seed = None if seed is None else int(seed)

        self._py_rand = random.Random(self.seed)
        self._torch_gen = torch.Generator()
        if self.seed is not None:
            self._torch_gen.manual_seed(self.seed)

    def __call__(self, signal):
        if not isinstance(signal, torch.Tensor):
            signal = torch.from_numpy(np.asarray(signal)).float()

        if signal.ndim != 2:
            raise ValueError(f"Shape must be (channels, time_steps), got {signal.shape}")

        channels, time_steps = signal.shape
        if self.block_width > time_steps:
            raise ValueError(f"block_width ({self.block_width}) cannot be larger than time_steps ({time_steps})")
        if self.num_blocks > channels:
            raise ValueError(f"num_blocks ({self.num_blocks}) cannot be larger than channels ({channels})")

        augmented_signal = signal.clone()
        selected_channels = self._py_rand.sample(range(channels), self.num_blocks)

        for ch in selected_channels:
            start_max = time_steps - self.block_width
            start = self._py_rand.randint(0, start_max)
            end = start + self.block_width
            mode = self._py_rand.choice(self.drop_modes)

            if mode == 'zero':
                augmented_signal[ch, start:end] = 0
            elif mode == 'noise':
                std = torch.std(augmented_signal[ch])
                noise = torch.randn(self.block_width, generator=self._torch_gen, device=signal.device) * (std * self._py_rand.uniform(0.0, 0.9))
                augmented_signal[ch, start:end] = noise
            elif mode == 'drift':
                base = torch.mean(augmented_signal[ch, start:end])
                drift = torch.linspace(base - 0.3, base + 0.3, self.block_width, device=signal.device)
                augmented_signal[ch, start:end] = drift
            elif mode == 'spike':
                base = torch.mean(augmented_signal[ch, start:end])
                std = torch.std(augmented_signal[ch, start:end])
                spike = base + torch.randn(self.block_width, generator=self._torch_gen, device=signal.device) * std * self._py_rand.uniform(1, 5)
                augmented_signal[ch, start:end] = spike

        return augmented_signal

class EarlyStopping:
    def __init__(self, patience=10, delta=0, mode='max'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == 'max':
            if current_score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = current_score
                self.counter = 0
        else:
            if current_score > self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = current_score
                self.counter = 0

        return self.early_stop