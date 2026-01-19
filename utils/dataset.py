import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, merged_images, transform=None):
        self.merged_images = merged_images
        self.transform = transform

    def __len__(self):
        return len(self.merged_images)

    def __getitem__(self, idx):
        image, label, _ = self.merged_images[idx]

        if self.transform:
            image = image.copy()
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(0, 2, 1).float() / 255.0

        return image, torch.tensor(label).long()