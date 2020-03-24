from PIL import Image
import os

import numpy as np
import torch

from torch.utils.data import Dataset

TEST_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../data/processed/train')

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.y = torch.from_numpy(np.load(root + '/y.npy')).float()
        self.subject_id = np.load(root + '/subject_id.npy')

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(os.path.join(self.root, 'X', str(idx) + '.png'))

        if self.transform:
            image = self.transform(image)

        return image, self.y[idx]

    def getY(self):
        return np.array(self.y).flatten()

    def getGroups(self):
        return self.subject_id

