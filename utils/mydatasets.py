import os
import random
import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

import utilities as UT


class MultiScaleProbabilistic(Dataset):
    """
    A PyTorch Dataset for loading multi-scale satellite images for probabilistic
    risk prediction.

    This dataset loads three multi-scale images, applies a single consistent random
    crop and data augmentation to all three, and calculates target Beta
    distribution parameters based on the crop's properties.

    Args:
        data_path (str): Base path to the dataset.
        pos_folder_17 (str): Folder for positive samples (year 17).
        neg_folder_17 (str): Folder for negative samples (year 17).
        pos_folder_18 (str): Folder for positive samples (year 18).
        neg_folder_18 (str): Folder for negative samples (year 18).
        pos_folder_19 (str): Folder for positive samples (year 19).
        neg_folder_19 (str): Folder for negative samples (year 19).
        pos_metadata (str): CSV for positive sample metadata.
        neg_metadata (str): CSV for negative sample metadata.
        min_crop_ratio (float): Minimum crop ratio.
        basic_transform (transforms.Compose): Basic transformations (resize, etc.).
        max_dist_proportion (float): Max distance influence for positive samples.
        base_certainty_K (float): Base certainty for Beta distribution.
        min_positive_risk_mean (float): Minimum mean risk for positive samples.
        min_concentration_positives (float): Minimum concentration for positives.
        weight_distance (float): Weight for distance influence on alpha.
        weight_crop_size (float): Weight for crop size influence on alpha.
        epsilon_beta (float): Small epsilon for numerical stability.
        test (bool): If True, use perfect Beta distribution for positive samples.
        large_crop_ratio (float): Probability of forcing a large crop.
    """
    def __init__(self,
                 data_path="./data",
                 pos_folder_17="Positive/Bing_Map_Positive_17",
                 neg_folder_17="Negative/Bing_Map_Negative_17",
                 pos_folder_18="Positive/Bing_Map_Positive_18",
                 neg_folder_18="Negative/Bing_Map_Negative_18",
                 pos_folder_19="Positive/Bing_Map_Positive_19",
                 neg_folder_19="Negative/Bing_Map_Negative_19",
                 pos_metadata="train_pos.csv",
                 neg_metadata="train_neg.csv",
                 min_crop_ratio=0.5,
                 basic_transform=None,
                 # Parameters for Beta distribution
                 max_dist_proportion=0.75,
                 base_certainty_K=1000.0,
                 min_positive_risk_mean=0.2,
                 min_concentration_positives=20.0,
                 weight_distance=0.7,
                 weight_crop_size=0.3,
                 epsilon_beta=0.00001,
                 test=False,
                 large_crop_ratio=0.7
                ):
        
        self.data_path = data_path
        self.pos_folder_17 = pos_folder_17
        self.neg_folder_17 = neg_folder_17
        self.pos_folder_18 = pos_folder_18
        self.neg_folder_18 = neg_folder_18
        self.pos_folder_19 = pos_folder_19
        self.neg_folder_19 = neg_folder_19 

        pos_meta =  UT.read_csv(os.path.join(self.data_path, pos_metadata))
        neg_meta =  UT.read_csv(os.path.join(self.data_path, neg_metadata))
        self.metadata = np.append(pos_meta, neg_meta, axis=0)
        
        self.basic_transform = basic_transform       

        self.min_crop_ratio = min_crop_ratio
        self.test = test

        self.max_dist_proportion = max_dist_proportion
        self.base_certainty_K = base_certainty_K
        self.min_positive_risk_mean = min_positive_risk_mean
        self.min_concentration_positives = min_concentration_positives
        self.weight_distance = weight_distance
        self.weight_crop_size = weight_crop_size
        self.epsilon_beta = epsilon_beta
        self.large_crop_ratio = large_crop_ratio

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        # Get file info and load original images
        filename = self.metadata[index][0]
        label = int(filename.split("_")[-1][0])
        if label > 1: label = 1

        if label == 0:
            img_path_17 = os.path.join(self.data_path, self.neg_folder_17, filename)
            img_path_18 = os.path.join(self.data_path, self.neg_folder_18, filename)
            img_path_19 = os.path.join(self.data_path, self.neg_folder_19, filename)
        else:
            img_path_17 = os.path.join(self.data_path, self.pos_folder_17, filename)
            img_path_18 = os.path.join(self.data_path, self.pos_folder_18, filename)
            img_path_19 = os.path.join(self.data_path, self.pos_folder_19, filename)
        
        img_path_list = [img_path_17, img_path_18, img_path_19]
                            
        img_original_17 = Image.open(img_path_17.replace("\ufeff", ""))
        img_original_18 = Image.open(img_path_18.replace("\ufeff", ""))
        img_original_19 = Image.open(img_path_19.replace("\ufeff", ""))

        # --- Probabilistic Target Calculation (based on a single, consistent crop) ---
        min_dim = min(img_original_17.size)
        
        if label == 1 and random.random() < self.large_crop_ratio:
            _crop_ratio = random.uniform(0.9, 1.0)
        else:
            _crop_ratio = random.uniform(self.min_crop_ratio, 1.0)
            
        _crop_size = int(min_dim * _crop_ratio)
        _crop_size = max(1, _crop_size)
        _crop_size = _crop_size if _crop_size % 2 == 0 else _crop_size + 1

        _max_x = img_original_17.size[0] - _crop_size
        _max_y = img_original_17.size[1] - _crop_size
        _crop_x = random.randint(0, max(0, _max_x))
        _crop_y = random.randint(0, max(0, _max_y))
        
        alpha_target, beta_target, normalized_dist = 0.0, 0.0, 0.0

        if label == 1:
            img_center_x, img_center_y = img_original_17.size[0] / 2.0, img_original_17.size[1] / 2.0
            crop_center_x, crop_center_y = _crop_x + _crop_size / 2.0, _crop_y + _crop_size / 2.0
            dist_pixels = np.sqrt((crop_center_x - img_center_x)**2 + (crop_center_y - img_center_y)**2)
            
            effective_max_dist = min(img_original_17.size) * self.max_dist_proportion / 2.0
            normalized_dist = min(dist_pixels / (effective_max_dist + 1e-6), 1.0)

            min_crop_norm = int(min(img_original_17.size) * self.min_crop_ratio)
            max_crop_norm = int(min(img_original_17.size) * 1.0)
            
            if max_crop_norm - min_crop_norm > 0:
                normalized_crop_size = (_crop_size - min_crop_norm) / (max_crop_norm - min_crop_norm)
            else:
                normalized_crop_size = 1.0
            normalized_crop_size = max(0.0, min(normalized_crop_size, 1.0))

            influence = ((1.0 - normalized_dist) * self.weight_distance +
                         normalized_crop_size * self.weight_crop_size)
            
            if self.test:
                influence = 1.0
                normalized_dist = 0.0
                
            target_mean = self.min_positive_risk_mean + (1.0 - self.min_positive_risk_mean) * influence
            target_mean = max(self.epsilon_beta, min(target_mean, 1.0 - self.epsilon_beta))
            
            concentration = self.min_concentration_positives + (self.base_certainty_K - self.min_concentration_positives) * influence
            alpha_target = target_mean * concentration
            beta_target = self.epsilon_beta
        else:
            alpha_target = self.epsilon_beta
            beta_target = self.base_certainty_K
        
        # --- Apply Consistent Augmentation ---
        img_aug_17 = F.crop(img_original_17, _crop_y, _crop_x, _crop_size, _crop_size)
        img_aug_18 = F.crop(img_original_18, _crop_y, _crop_x, _crop_size, _crop_size)
        img_aug_19 = F.crop(img_original_19, _crop_y, _crop_x, _crop_size, _crop_size)
        
        if torch.rand(1) < 0.5:
            img_aug_17, img_aug_18, img_aug_19 = F.hflip(img_aug_17), F.hflip(img_aug_18), F.hflip(img_aug_19)
            
        if torch.rand(1) < 0.5:
            img_aug_17, img_aug_18, img_aug_19 = F.vflip(img_aug_17), F.vflip(img_aug_18), F.vflip(img_aug_19)
            
        angle = transforms.RandomRotation.get_params([-90, 90])
        img_aug_17, img_aug_18, img_aug_19 = F.rotate(img_aug_17, angle), F.rotate(img_aug_18, angle), F.rotate(img_aug_19, angle)

        order, bright, cont, sat, hue = transforms.ColorJitter.get_params(
            brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4], hue=[0.0, 0.1])
        
        for img in [img_aug_17, img_aug_18, img_aug_19]:
            if bright is not None: img = F.adjust_brightness(img, bright)
            if cont is not None: img = F.adjust_contrast(img, cont)
            if sat is not None: img = F.adjust_saturation(img, sat)
            if hue is not None: img = F.adjust_hue(img, hue)
        
        # --- Final Transforms ---
        img_normal_17 = self.basic_transform(img_original_17)
        img_normal_18 = self.basic_transform(img_original_18)
        img_normal_19 = self.basic_transform(img_original_19)

        img_aug_17 = self.basic_transform(img_aug_17)
        img_aug_18 = self.basic_transform(img_aug_18)
        img_aug_19 = self.basic_transform(img_aug_19)

        return {
            "normal": [img_normal_17, img_normal_18, img_normal_19],
            "aug": [img_aug_17, img_aug_18, img_aug_19],
            "label": label,
            "img_path_list": img_path_list,
            "alpha_target": torch.tensor(alpha_target, dtype=torch.float32),
            "beta_target": torch.tensor(beta_target, dtype=torch.float32),
            "crop_ratio": torch.tensor(_crop_ratio, dtype=torch.float32),
            "normalized_dist": torch.tensor(normalized_dist, dtype=torch.float32)
        }
    
class Inferece_Dataset(Dataset):
    """
    A PyTorch Dataset for loading multi-scale satellite images for probabilistic
    risk prediction.

    This dataset loads three multi-scale images, applies a single consistent random
    crop and data augmentation to all three, and calculates target Beta
    distribution parameters based on the crop's properties.

    Args:
        data_path (str): Base path to the dataset.
        neg_folder_17 (str): Folder for negative samples (year 17).
        neg_folder_18 (str): Folder for negative samples (year 18).
        neg_folder_19 (str): Folder for negative samples (year 19).
        neg_metadata (str): CSV for negative sample metadata.
        basic_transform (transforms.Compose): Basic transformations (resize, etc.).
    """
    def __init__(self,
                 data_path="./data",
                 neg_folder_17="17",
                 neg_folder_18="18",
                 neg_folder_19="19",
                 neg_metadata="sa_river_walk.csv",
                 basic_transform=None,
                ):
        
        self.data_path = data_path
        self.neg_folder_17 = neg_folder_17
        self.neg_folder_18 = neg_folder_18
        self.neg_folder_19 = neg_folder_19 

        self.metadata =  UT.read_csv(os.path.join(self.data_path, neg_metadata))
        self.basic_transform = basic_transform               

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        # Get file info and load original images
        filename = self.metadata[index][0]
    
        img_path_17 = os.path.join(self.data_path, self.neg_folder_17, filename)
        img_path_18 = os.path.join(self.data_path, self.neg_folder_18, filename)
        img_path_19 = os.path.join(self.data_path, self.neg_folder_19, filename)
        
        img_path_list = [img_path_17, img_path_18, img_path_19]
                            
        # img_original_17 = Image.open(img_path_17.replace("\ufeff", ""))
        # img_original_18 = Image.open(img_path_18.replace("\ufeff", ""))
        # img_original_19 = Image.open(img_path_19.replace("\ufeff", ""))
        img_original_17 = Image.open(img_path_17)
        img_original_18 = Image.open(img_path_18)
        img_original_19 = Image.open(img_path_19)
                
        img_normal_17 = self.basic_transform(img_original_17)
        img_normal_18 = self.basic_transform(img_original_18)
        img_normal_19 = self.basic_transform(img_original_19)
        
        return {
            "normal": [img_normal_17, img_normal_18, img_normal_19],            
            "filename": filename
        }    