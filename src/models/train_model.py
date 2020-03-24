"""Malaria detection on segmented cell images.

Random search and training script for convolutional neural networks
(see models.py).

Dataset from https://ceb.nlm.nih.gov/proj/malaria/cell_images.zip.

Script runs N_ITER iterations of random search over the parameters defined in
GRID. Model score (i.e. accuracy) is cross-validated over N_SPLITS using a
sklearn's GroupKFold scheme whereby samples (here, cell images) from the
same group (here, from the same patient) are not shared between
validation splits.

Validation results are saved to malaria-detection/reports using a csv format.
"""
import warnings

# Ignore sklearn deprecation warnings caused by skorch
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=FutureWarning)

import os
import datetime

import pandas as pd
import numpy as np
import torch
from torchvision import transforms
import torch.optim as optim
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from skorch import NeuralNetBinaryClassifier
from skorch.helper import SliceDataset

from src.models.models import Tiny, PaperCNN
from src.data.CustomDataset import CustomDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../data/processed/test'
)

REPORT_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../reports'
)

# Training and random search
torch.manual_seed(123)
np.random.seed(123)

N_ITER = 5

GRID = dict(
    lr=np.logspace(start=-5, stop=-1, num=1000),
    optimizer=[optim.Adam, optim.SGD],
    batch_size=[32, 64, 128],
    max_epochs=np.logspace(start=0, stop=3, num=100, base=2, dtype='int'),
    module__dp_fc=np.linspace(start=0, stop=0.5, num=100),
    module__dp_conv=np.logspace(start=-5, stop=-1, num=50),
)

# Cross validation
N_SPLITS = 2

# Transforms
# Augmented
TRANSFORM_AUG = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomAffine(degrees=180, translate=(0.15, 0.15),
                            scale=(0.9, 1.05), shear=30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5298, 0.4235, 0.4512],
                         std=[0.3347, 0.2708, 0.2851])
])

# Vanilla
TRANSFORM = transforms.Compose([
    transforms.Resize(size=(96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5298, 0.4235, 0.4512],
                         std=[0.3347, 0.2708, 0.2851])
])

# SCRIPT
print('Searching...\n')

# Retrieve x, y and subject_id
dataset = CustomDataset(root=TRAIN_PATH, transform=TRANSFORM)
y = dataset.getY()
subject_id = dataset.getGroups()

# Wrap Pytorch Dataset in skorch class for sklearn compatibility
X = SliceDataset(dataset)

# GroupeKFold validation scheme meaning samples from a given group
# (here a subject) won't be shared between splits
group_kfold = GroupKFold(n_splits=N_SPLITS).get_n_splits(X, y, subject_id)


# Estimator and randomized search
net = NeuralNetBinaryClassifier(
    Tiny,
    device=DEVICE,
    criterion=torch.nn.BCELoss,
    iterator_train__shuffle=True,
    train_split=None  # RandomizedSearchCV handles validation
)

clf = RandomizedSearchCV(estimator=net,
                         param_distributions=GRID,
                         n_iter=N_ITER,
                         cv=group_kfold,
                         scoring='accuracy',
                         verbose=1,
                         refit=False)

search = clf.fit(X, y)

# Retrieve best results and save
df = pd.DataFrame.from_dict(search.cv_results_)
df = df.sort_values(by='rank_test_score')
df.to_csv(os.path.join(REPORT_PATH,
                       '{date:%Y-%m-%d_%H:%M:%S}.csv'.format(
                           date=datetime.datetime.now() )))
