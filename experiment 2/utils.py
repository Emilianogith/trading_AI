import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import pandas as pd
from io import StringIO
from torch.utils.data import random_split, TensorDataset


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def plot_train_loss(csv_data):
    # Read CSV data into DataFrame
    df = pd.read_csv(csv_data)

    # Drop rows with NaNs and keep one row per epoch per metric
    train_loss = df.dropna(subset=['train_loss']).groupby('epoch').mean()['train_loss']
    val_loss = df.dropna(subset=['val_loss']).groupby('epoch').mean()['val_loss']

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss.index, train_loss.values, label='Train Loss', marker='o')
    plt.plot(val_loss.index, val_loss.values, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_dataset(data_path, verbose = False):
    dataset_list_no_labels = []
    labels_list = []

    for filename in os.listdir(data_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(data_path, filename)
            array = np.load(file_path)
            if verbose:
                print('lenght of data: ',len(array))

            dataset_list_no_labels.append(array[:, :-1])
            labels_list.append(array[:, -1])

    X = np.concatenate(dataset_list_no_labels)
    Y = np.concatenate(labels_list)

    if verbose:
        print("size X:", X.shape)
        print("size Y:", Y.shape)

        positive_examples = np.sum(Y==1)
        print("positive examples:",positive_examples)
        print("negative examples:",len(Y)-positive_examples)

    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    return X, Y


def split_data(split_mode = 'random'):
    data_path='./data'
    X,Y = get_dataset(data_path)

    if split_mode == 'random':
        # Wrap into a dataset
        dataset = TensorDataset(X, Y)

        # Define lengths for the splits
        train_len = int(0.7 * len(dataset))
        val_len   = int(0.15 * len(dataset))
        test_len  = len(dataset) - train_len - val_len

        # Split the dataset randomly
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

        torch.save(train_dataset, './train_dataset.pt')
        torch.save(val_dataset, './val_dataset.pt')
        torch.save(test_dataset, './test_dataset.pt')

        print("Datasets splitted.")

    elif split_mode == 'split_with_criteria':
        # Identify positive and negative indices
        positive_indices = (Y == 1).nonzero(as_tuple=True)[0]
        negative_indices = (Y == 0).nonzero(as_tuple=True)[0]

        # Shuffle indices
        positive_indices = positive_indices[torch.randperm(len(positive_indices))]
        negative_indices = negative_indices[torch.randperm(len(negative_indices))]

        # Calculate desired split sizes
        total_len = len(Y)
        train_len = int(0.7 * total_len)
        val_len = int(0.15 * total_len)
        test_len = total_len - train_len - val_len

        # Ensure ~50% of positives go into the train set
        num_pos_train = len(positive_indices) // 2
        train_pos = positive_indices[:num_pos_train]

        # Fill train set with negatives
        remaining_train_len = train_len - len(train_pos)
        train_neg = negative_indices[:remaining_train_len]

        # Remaining indices for val and test
        remaining_pos = positive_indices[num_pos_train:]
        remaining_neg = negative_indices[remaining_train_len:]

        # Concatenate and shuffle val and test
        remaining_indices = torch.cat([remaining_pos, remaining_neg])
        remaining_indices = remaining_indices[torch.randperm(len(remaining_indices))]

        val_indices = remaining_indices[:val_len]
        test_indices = remaining_indices[val_len:]

        # Combine train indices
        train_indices = torch.cat([train_pos, train_neg])
        train_indices = train_indices[torch.randperm(len(train_indices))]

        # Build datasets
        train_dataset = TensorDataset(X[train_indices], Y[train_indices])
        val_dataset   = TensorDataset(X[val_indices], Y[val_indices])
        test_dataset  = TensorDataset(X[test_indices], Y[test_indices])

        # Save
        torch.save(train_dataset, './train_dataset.pt')
        torch.save(val_dataset, './val_dataset.pt')
        torch.save(test_dataset, './test_dataset.pt')

        print("Datasets split with at least 50% of positives in train.")


if __name__ == "__main__":
    data_path='./data'
    get_dataset(data_path, verbose=True)
    split_data(split_mode='split_with_criteria')