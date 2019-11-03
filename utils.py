"""Utilities."""
from __future__ import print_function

import os
import time
import sys
import math
import shutil

import torch
import numpy as np
from matplotlib import pyplot as plt
from random import shuffle


def compute_loss_error(net, criterion, loader, device, n=None):
    """Helper function to compute loss and error of a model.

    n is to specify a number of batches. If set to None, will compute loss
    and error over all dataset.
    """
    net.eval()

    # Compute loss and error over {n} batches. If set to None,
    # will compute over all batches
    correct = 0
    total = 0
    loss = 0

    with torch.no_grad():
        for i, data in enumerate(loader):
            if n and i > n:
                break

            inputs, labels = data[0].to(device), data[1].float().to(device)
            outputs = net(inputs)
            loss += criterion(outputs.view(-1), labels).item()
            predictions = outputs >= 0.5
            correct += (predictions.view(-1) == labels.byte()).sum().item()
            total += labels.size(0)  # Batch size

        error = 1 - correct / total
        loss = loss / total

    net.train()

    return error, loss


class Board:
    """Keep track of error and cost while training.

    Utility class to track cost & error for error and validation sets
    respectively. Also provides basic early stopping functionnality.
    """

    def __init__(self, dataset_name, epoch_total, net, criterion, train_loader,
                 train_loader_eval, val_loader, device, transforms,
                 optimizer, train_sample=20):
        """Constructor.

        Args:
            dataset_name(str): Dataset name.
            epoch_total (int): Total training batches.
            net(Net): Model.
            criterion(nn): Criterion used for training.
            train_loader(DataLoader): Training set loader.
            val_loader(DataLoader): Validation set loader.
            device(device): Device used for training.
            train_sample(int): Number of batches to compute train cost
            and error.
            If set to None, will compute over all training set.
        """
        self._epochs = 0
        self._dataset_name = dataset_name
        self._epoch_total = epoch_total
        self._net = net
        self._criterion = criterion
        self._train_loader = train_loader
        self._train_loader_eval = train_loader_eval
        self._val_loader = val_loader
        self._device = device
        self._transforms = transforms
        self._optimizer = optimizer
        self._train_sample = train_sample
        self._batch_no = []
        self._train_loss = []
        self._val_loss = []
        self._train_error = []
        self._val_error = []
        self._min_val_error = None
        self._stop_counter = 0

    def eval(self, current_epoch):
        """Compute, save and print metrics.

        Should be called after every epoch."""
        self._epochs = current_epoch

        # Error and loss over training set
        train_error, train_loss = compute_loss_error(self._net,
                                                     self._criterion,
                                                     self._train_loader_eval,
                                                     self._device,
                                                     self._train_sample)

        # Error and loss over validation set
        val_error, val_loss = compute_loss_error(self._net,
                                                 self._criterion,
                                                 self._val_loader,
                                                 self._device)

        # Update min
        record_flag = ""
        if self._min_val_error is None or val_error < self._min_val_error:
            # If new min, update attributes and checkpoint model
            self._min_val_error = val_error
            self._stop_counter = 0
            record_flag = "| New record!"
            torch.save(self._net.state_dict(), 'checkpoint.pt')
        else:
            self._stop_counter += 1

        # Report
        print("{:6.2f} % | "
              "train error : {:7.4f} % | "
              "train loss : {:6.4f} | "
              "val error : {:7.4f} % | "
              "val loss : {:6.4f} "
              "{}"
              .format(100 * self._epochs / self._epoch_total,
                      100 * train_error,
                      train_loss, 100 * val_error, val_loss, record_flag))

        # Update all lists
        self._batch_no.append(self._epochs)
        self._train_loss.append(train_loss)
        self._val_loss.append(val_loss)
        self._train_error.append(train_error)
        self._val_error.append(val_error)

    def counter(self):
        """Return number of epochs since val error record has been broken.

        Returns:
            int: number of epochs since val error record has been broken.

        """

        return self._stop_counter

    def plot_loss(self, show=True, save_path=None):
        """Display loss plot."""

        plt.clf()
        plt.plot(self._batch_no, self._train_loss, label='Training')
        plt.plot(self._batch_no, self._val_loss, label='Validation')
        plt.title('Model loss on {}'.format(self._dataset_name))
        plt.xlabel('Batches')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.gca().set_ylim([0, 0.03])  # Fixed scale on y axis
        plt.gca().set_yticks(np.arange(0, 0.031, 0.003))

        if show:
            plt.draw()
            plt.pause(0.001)

        if save_path:
            plt.savefig(save_path)

    def plot_error(self, show=True, save_path=None):
        """Display error plot."""

        plt.clf()
        plt.plot(self._batch_no, self._train_error, label='Training')
        plt.plot(self._batch_no, self._val_error, label='Validation')
        plt.title('Model Error on {}'.format(self._dataset_name))
        plt.xlabel('Batches')
        plt.ylabel('Error (%)')
        plt.legend()
        plt.grid()
        plt.gca().set_ylim([0, 0.6])  # Fixed scale on y axis
        plt.gca().set_yticks(np.arange(0, 0.61, 0.05))

        if show:
            plt.draw()
            plt.pause(1)

        if save_path:
            plt.savefig(save_path)

    def save_results(self, save=False):
        """Save model, metrics and log of experiment. Return metrics."""

        # Use timestamp as ID
        timestamp = str(time.time()).replace('.', '_')

        metrics = compute_metrics(self._net,
                                  self._criterion,
                                  self._train_loader_eval,
                                  self._val_loader,
                                  self._device)

        # Save log and model if required
        if save:
            # Create directory if needed
            if not os.path.exists('./results'):
                os.mkdir('./results')

            os.mkdir('./results/{}'.format(timestamp))

            file = open('./results/{}/log.txt'.format(timestamp), 'w+')
            _print_log(file, timestamp, self._epochs, self._transforms,
                       self._net, self._optimizer,
                       metrics)
            file.close()

            # Save state and metrics
            save = {
                'metrics': metrics,
                'epochs': self._epochs,
            }
            torch.save(save, './results/{}/metrics.pt'.format(timestamp))

            m = {
                'model_state_dict': self._net.state_dict(),
                'optimizer_state_dict': self._optimizer.state_dict(),
            }
            torch.save(m, './results/{}/model.pt'.format(timestamp))

        metrics['id'] = timestamp

        return metrics

    def set_train_loader(self, loader):
        self._train_loader = loader


def show_tensor_img(t):
    """Show image.

    Args:
        t(tensor): Torch tensor.

    """
    n = t.numpy()
    n = n.transpose((1, 2, 0))
    plt.imshow(n)
    plt.axis("off")
    plt.show()


def _print_log(file, timestamp, epochs, transforms, model, optimizer,
               metrics):
    """Print human readable summary of experiment."""

    def print_metric_obj(obj):
        print("train error      : {:7.4f} % | train loss      : {:6.4f}\n"
              "validation error : {:7.4f} % | validation loss : {:6.4f}"
              .format(100 * obj['train_error'],
                      obj['train_loss'],
                      100 * obj['validation_error'],
                      obj['validation_loss']))

    # Log transforms, model, optimizer and metrics in human readable format
    original = sys.stdout
    sys.stdout = file
    print('#' * 70)
    print('\n')
    print(timestamp)
    print('\n')
    print(time.asctime(time.localtime(float(timestamp.replace('_', '.')))))
    print('\n')
    print('METRICS:\n')
    print_metric_obj(metrics)
    print('\n')
    print('EPOCHS: \n\n{}'.format(epochs))
    print('\n')
    print('TRANSFORMS:\n')
    print(transforms)
    print('\n')
    print('MODEL:\n')
    print(model)
    print('\n{} parameters'.format(sum(p.numel() for p in model.parameters())))
    print('\n')
    print('OPTIMIZER:\n')
    print(optimizer)
    print('\n')
    sys.stdout = original


def compute_metrics(model, criterion, train_loader_eval, val_loader, device):
    train_error, train_loss = compute_loss_error(model, criterion,
                                                 train_loader_eval, device)
    validation_error, validation_loss = compute_loss_error(model, criterion,
                                                           val_loader, device)

    return {
        'train_loss': train_loss,
        'train_error': train_error,
        'validation_loss': validation_loss,
        'validation_error': validation_error,
    }


def split(source, train_ratio, validation_ratio, test_ratio):
    """Split NIH malaria dataset.

    Shuffle images and split into train, test and validation sets following the
    given proportions.train, test and validation parameters should sum to 1.

    Save datasets under ./data/train, ./data/validation & ./data/test.

    Keep the /Parasitized /Uninfected structure for subdirectories.

    Update: MalariaDataset could load the full dataset and
    torch.utils.data.random_split() could be used instead.


    Args:
      source(str): Location of extracted dataset. Typically './cell_images'.
      train_ratio (float): Number from 0 to 1.
      validation_ratio(float): Number from 0 to 1.
      test_ratio(float): Number from 0 to 1.

    """
    print('Splitting NIH malaria dataset...')

    if 100 * train_ratio + 100 * test_ratio + 100 * validation_ratio != 100:
        # Multiplication by 100 to avoid numerical errors
        raise Exception('Split ratios should sum to 1.')

    # Create directories
    if os.path.exists('./data'):
        raise Exception('It seems a split already exists. Check the ./data '
                        'directory.')
    os.makedirs('./data/train/Parasitized')
    os.makedirs('./data/train/Uninfected')
    os.makedirs('./data/validation/Parasitized')
    os.makedirs('./data/validation/Uninfected')
    os.makedirs('./data/test/Parasitized')
    os.makedirs('./data/test/Uninfected')

    def distribute(cell_type):
        """List files, shuffle and split.

        Args:
            cell_type(str): 'Parasitized' or 'Uninfected'

        """
        print('Splitting {} cells...'.format(cell_type))

        # List files and shuffle
        file_list = os.listdir('{}/{}'.format(source, cell_type))
        shuffle(file_list)

        # Determine ratios
        files_n = len(file_list)
        train_n = math.ceil(train_ratio * files_n)
        test_n = math.floor(test_ratio * files_n)
        validation_n = math.floor(validation_ratio * files_n)

        if train_n + test_n + validation_n != files_n:
            raise Exception('Some files in {}/{} are ignored!'
                            .format(source, cell_type))
        train_files = file_list[0:train_n]
        validation_files = file_list[train_n:train_n + validation_n]
        test_files = file_list[train_n + validation_n:]

        def copy(set_name, f_list):
            # Copy files listed in f_list to ./data/{set_name}
            for f in f_list:
                shutil.copy2(
                    '{}/{}/{}'.format(source, cell_type, f),
                    './data/{}/{}/{}'.format(set_name, cell_type, f)
                )

        copy('train', train_files)
        copy('validation', validation_files)
        copy('test', test_files)

    distribute('Parasitized')
    distribute('Uninfected')

    print('Splitting successful!')


def current_top(n=5):
    # List timestamps
    timestamps = [name for name in os.listdir('./results')
                  if os.path.isdir(os.path.join('./results', name))]

    # List min validation errors
    min_error = []
    for ts in timestamps:
        checkpoint = torch.load('./results/{}/metrics.pt'.format(ts))
        min_error.append(checkpoint['metrics']['validation_error'])

    if os.path.exists('./results/current_top.txt'):
        os.remove('./results/current_top.txt')

    file = open('./results/current_top.txt', 'w+')
    for i, t in enumerate(sorted(zip(min_error, timestamps))):
        if n and i >= n:  # Summarize n results
            break
        log = open('./results/{}/log.txt'.format(t[1]), 'r')
        # Append log to summary
        file.write(log.read())
        file.write('\n')


def summarize(session, id, name, epochs, patience, epochs_completed, lr,
              lr_decay, dp_conv, dp_fc, val_error, train_error,
              val_loss, train_loss):
    """"Summarize experiments to a handy .csv file."""
    # Update results.csv
    if not os.path.exists('./results.csv'):
        newline = 'SESSION,ID,MODEL,EPOCHS,PATIENCE,EPOCHS_COMPLETED,LR,' \
                  'LR_DECAY,DP_CONV,DP_FC,VAL_ERROR,TRAIN_ERROR,' \
                  'VAL_LOSS,TRAIN_LOSS\n'
    else:
        newline = ''

    newline += '{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
        session,
        id,
        name,
        epochs,
        patience,
        epochs_completed,
        lr,
        lr_decay,
        dp_conv,
        dp_fc,
        val_error,
        train_error,
        val_loss,
        train_loss
    )
    file = open('./results.csv', 'a+')
    file.write(newline)
    file.close()
