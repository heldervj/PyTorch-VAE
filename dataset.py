import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile

from skimage import io, transform
from PIL import Image
import PIL.ImageEnhance as ie
import PIL.Image as im
from torch.autograd import Variable

import numpy as np
import pandas as pd

# Add your custom dataset class here
class ImageDataset(Dataset): 
    
    def __init__(self, csv_file, data_path, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = data_path
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Id = File
        # Category = Category_Id
        img_name = os.path.join(self.root_dir, self.data_frame['File'][idx])         
        image = Image.open(img_name).convert('RGB')                               
        label = np.array(self.data_frame['Category_Id'][idx])                        
        if self.transform:            
            image = self.transform(image)                                         
        sample = (image, label) 
        return sample


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.csv_file_train = kwargs['csv_file_train']
        self.csv_file_valid = kwargs['csv_file_valid']

    def setup(self, stage: Optional[str] = None) -> None:
    
        train_transforms = transforms.Compose([transforms.Resize((self.patch_size, self.patch_size)),
                                              transforms.ToTensor(),])

        # train_transforms = None
        
        val_transforms = transforms.Compose([transforms.Resize((self.patch_size, self.patch_size)),
                                              transforms.ToTensor(),])
        
        self.train_dataset = ImageDataset(
            self.csv_file_train,
            self.data_dir,
            transform=train_transforms,
        )

        print(self.data_dir)

        print(len(self.train_dataset))
        
        # Replace CelebA with your dataset
        self.val_dataset = ImageDataset(
            self.csv_file_valid,
            self.data_dir,
            transform=val_transforms,
        )
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     