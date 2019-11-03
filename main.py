"""Neural network for malaria detection.

Main experiment script.
Dataset from https://ceb.nlm.nih.gov/proj/malaria/cell_images.zip.
"""
from itertools import product

from train import train
import models
from utils import summarize

__author__ = 'Sacha Morin'

SESSION = 0

# Options
EPOCHS = 100
PATIENCE = 20  # Epochs with no improvements before early stopping
BATCH_SIZE = 128

# Model selection and hyper parameters
# Add more than one value to list to trigger grid search
MODELS = [models.CustomVGG]  # Models

LR = [0.001]
LR_DECAY = [1]
DP_CONV = [0]  # Dropout on conv layers
DP_FC = [0.5]  # Dropout on fully-connected layers

# Hyper parameters combinations for grid search
combinations = product(MODELS, LR, LR_DECAY, DP_CONV, DP_FC)

# Basic Grid Search
for i, c in enumerate(combinations, 1):
    print('\n\nExperiment no {}\n'.format(i))

    # Hyper parameters
    model, lr, lr_decay, dp_conv, dp_fc = c

    # Model initialization
    m = model(dp_conv=dp_conv, dp_fc=dp_fc)

    # Launch train session
    results = train(model=m, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    patience=PATIENCE, lr=lr, lr_decay=lr_decay, plot=False)

    # Save experiment to results.csv
    summarize(session=SESSION,
              id=results['id'],
              name=m.__class__.__name__,
              epochs=EPOCHS,
              patience=PATIENCE,
              epochs_completed=results['epochs_completed'],
              lr=lr,
              lr_decay=lr_decay,
              dp_conv=dp_conv,
              dp_fc=dp_fc,
              val_error=results['validation_error'],
              train_error=results['train_error'],
              val_loss=results['validation_loss'],
              train_loss=results['train_loss'])
