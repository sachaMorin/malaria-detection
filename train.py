"""Malaria detection neural network.

Train routine.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets

from utils import Board


def train(model, epochs, batch_size, patience, lr, lr_decay, plot):
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Transforms
    # Augmented
    transform_aug = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomAffine(degrees=180, translate=(0.15, 0.15),
                                scale=(0.9, 1.05), shear=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5295, 0.4240, 0.4530],
                             std=[0.3342, 0.2703, 0.2857])
    ])

    # Vanilla
    transform = transforms.Compose([
        transforms.Resize(size=(96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5295, 0.4240, 0.4530],
                             std=[0.3342, 0.2703, 0.2857])
    ])

    # Datasets
    # ImageFolder loads in folder order, 'Parasitized' is mapped to 0 and
    # 'Uninfected' is mapped to 1
    print('\n\nSetting up NIH Malaria Dataset...\n')
    train_aug = datasets.ImageFolder(root='./data/train',
                                        transform=transform_aug)
    train = datasets.ImageFolder(root='./data/train',
                                    transform=transform)
    validation = datasets.ImageFolder(root='./data/validation',
                                         transform=transform)

    # Loaders
    train_loader_aug = torch.utils.data.DataLoader(train_aug,
                                                   batch_size=batch_size,
                                                   shuffle=True, num_workers=4)
    train_loader = torch.utils.data.DataLoader(train,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(validation,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=4)

    # MODEL
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                            lr_lambda=lambda e: lr_decay ** e)

    # Custom class to track, print and save performance metrics
    board = Board(dataset_name='NIH Malaria Dataset',
                  epoch_total=epochs,
                  net=model,
                  criterion=criterion,
                  train_loader=train_loader_aug,
                  train_loader_eval=train_loader,
                  val_loader=validation_loader,
                  device=device,
                  transforms=transform_aug,
                  optimizer=optimizer,
                  train_sample=int(2000 / batch_size))

    print('Training...')
    print('Device : {}\n'.format(device))

    # Training
    board.eval(current_epoch=0)
    if plot:
        board.plot_error()

    for i in range(1, epochs + 1):
        # Epoch
        for j, data in enumerate(train_loader_aug, 1):
            optimizer.zero_grad()

            inputs, labels = data[0].to(device), data[1].float().to(device)

            outputs = model(inputs)

            loss = criterion(outputs.view(-1), labels)
            loss.backward()

            optimizer.step()

        # Add point and plot loss
        board.eval(current_epoch=i)

        if plot:
            board.plot_error()

        # Early stopping
        if board.counter() > patience:
            print('Early Stopping...')
            break

        # Update learning rate
        scheduler.step()

    # Load last checkpoint (lowest validation error)
    model.load_state_dict(torch.load('checkpoint.pt'))

    # Save results
    print('\nSaving results...')
    metrics = board.save_results(save=False)  # Save and return metrics
    metrics['epochs_completed'] = i  # Epochs completed

    return metrics
