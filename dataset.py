
import sys
import cv2
sys.path.append('') # add path
import numpy as np
from PIL import Image
import torch
import pandas as pd
import albumentations
import random

from torch.utils.data import Dataset

class BBBC021_3ChannelSet(Dataset):
    def __init__(self, path10, local_crops_number):
        path0 = pd.read_csv(path10)

        self.X0 = path0[['Image_FileName_DAPI','Image_FileName_Tubulin', 'Image_FileName_Actin']]
        self.tag = 'Unique_Compounds'
        self.aug0 = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Normalize(mean=[0], std=[1], max_pixel_value=10000, always_apply=True),
            albumentations.augmentations.crops.transforms.RandomResizedCrop(224, 224, scale=(0.1, 0.2), ratio=(1, 1),
                                                                            interpolation=cv2.INTER_CUBIC,
                                                                            always_apply=True),
        ],
            additional_targets={'image1': 'image', 'image2': 'image'})

        self.aug1 = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Normalize(mean=[0], std=[1], max_pixel_value=10000, always_apply=True),
            albumentations.augmentations.crops.transforms.RandomResizedCrop(224, 224, scale=(0.1, 0.2), ratio=(1, 1),
                                                                            interpolation=cv2.INTER_CUBIC,
                                                                            always_apply=True),
        ],
            additional_targets={'image1': 'image', 'image2': 'image'})

        self.local_crops_number = local_crops_number

        self.aug_local_crop = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Normalize(mean=[0], std=[1], max_pixel_value=10000, always_apply=True),
            albumentations.augmentations.crops.transforms.RandomResizedCrop(96, 96, scale=(0.04, 0.08), ratio=(1, 1),
                                                                            interpolation=cv2.INTER_CUBIC,
                                                                            always_apply=True),
        ],
            additional_targets={'image1': 'image', 'image2': 'image'})

    def __len__(self):
        return (len(self.X0))

    def __getitem__(self, i):
        Label = self.tag[i]
        same_idx = self.tag[self.tag == Label].index.values

        firstImageSet = []
        secondImageSet = []
        for channel in ['Image_FileName_DAPI','Image_FileName_Tubulin', 'Image_FileName_Actin']:
            Aimage = Image.open(self.X0[i, channel])
            Aimage = np.array(Aimage)
            Aimage[Aimage > 10000] = 10000

            transformed0 = self.aug0(image=Aimage)
            image = transformed0['image']
            image_0 = image.astype(np.float32)

            transformed1 = self.aug1(image=Aimage)
            image = transformed1['image']
            image_1 = image.astype(np.float32)

            image_0 = np.expand_dims(image_0, 0)
            image_1 = np.expand_dims(image_1, 0)
            firstImageSet.append(image_0)
            secondImageSet.append(image_1)

        image_0 = torch.tensor(np.concatenate(firstImageSet, axis=0), dtype=torch.float)
        del firstImageSet
        image_1 = torch.tensor(np.concatenate(secondImageSet, axis=0), dtype=torch.float)
        del secondImageSet

        crops = []
        crops.append(image_0)
        crops.append(image_1)

        random_i = random.choice(same_idx)

        # Select and load second image from same compound

        rand_images = []
        for channel in ['Image_FileName_DAPI','Image_FileName_Tubulin', 'Image_FileName_Actin']:
            rand_Aimage = Image.open(self.X0[random_i, channel])
            rand_Aimage = np.array(rand_Aimage)
            rand_Aimage[rand_Aimage > 10000] = 10000
            rand_images.append(rand_Aimage)

        for _ in range(self.local_crops_number):
            rand_augmented_images = []
            for channeled_image in rand_images:
                transformed2 = self.aug_local_crop(image=channeled_image)
                image = transformed2['image']
                image = image.astype(np.float32)
                image = np.expand_dims(image, 0)
                rand_augmented_images.append(image)

            image_2 = np.concatenate(rand_augmented_images, axis=0)
            del rand_augmented_images
            image_2 = torch.tensor(image_2, dtype=torch.float)
            crops.append(image_2)

        return crops