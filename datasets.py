import os
import torch
import ast
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MultiLabelImageDataset(Dataset):
    def __init__(self, metadata_file, img_dir, transform=None):
        """
        Args: 
            metadata_file (string): Path to the CSV files with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optonal transform to be applied on a sample.
        """
        self.metadata = pd.read_csv(metadata_file) # Load the csv metadata file
        self.img_dir = img_dir # Directory where images are stored
        self.transform = transform # Any image transformations like resizing, normalization, etc

    def __len__(self):
        """ Returns the total number of samples"""
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample.

        Returns:
             image (Tensor): The image data.
             labels (Tensor): The corresponding mulit-label target.
        """
        # Get the file name of the image
        img_name = os.path.join(self.img_dir, self.metadata.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB') # Ensure image is in RGB

        # Get the labels of the image (assuming the labels are in the second column as a stringified list)
        labels = self.metadata.iloc[idx, 1]
        labels = ast.literal_eval(labels) # Convert the string to list

        # Apply transformations, if any
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels, dtype=torch.float32)
    

transform = transforms.Compose([
    transforms.Resize((720, 1280)), # Resize the imag file as expected by the model
    transforms.ToTensor(), # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalizr the image using means and stds
])


def create_data_loaders(batch_size=1, shuffle=True, num_workers=4):
    train_metadata_file = os.path.join('..', 'data', 'metadata', 'imgs_train_oversampling.csv')
    val_metadata_file = os.path.join('..', 'data', 'metadata', 'imgs_val.csv')
    img_dir = os.path.join('..', 'data', 'images')

    train_dataset = MultiLabelImageDataset(metadata_file=train_metadata_file, img_dir=img_dir, transform=transform)
    val_dataset = MultiLabelImageDataset(metadata_file=val_metadata_file, img_dir=img_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


if __name__ == "__main__":
    # print("Current working directory: ", os.getcwd())
    train_loader, val_loader = create_data_loaders(batch_size=1)

    # Examples iterating through the DataLoader
    for images, labels in train_loader:
        print("Batch of images shape: ", images.shape)
        print("Batch of labels shape: ", labels.shape)
        break # Just to show the first batch