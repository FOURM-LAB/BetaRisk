import torch 
import torch.nn as nn
from torchvision import models
import numpy as np
import torch.nn.functional as F

class ResNet50_with_Feature(nn.Module):
    def __init__(self, feature_shape=(2048,24,24), 
                 pretrained=True, requires_grad=True,
                 num_classes=2):
        super(ResNet50_with_Feature, self).__init__()
        
        self.ft_img = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V2") if pretrained else models.resnet50()
        self.ft_img_modules = list(self.ft_img.children())[:-1]
        self.ft_img = nn.Sequential(*self.ft_img_modules)
        for p in self.ft_img.parameters():
            p.requires_grad = requires_grad

        # ConvLayer for image
        self.conv_img = nn.Sequential(
            nn.Conv2d(
                in_channels=feature_shape[0], # input height
                out_channels=feature_shape[0], # n_filters
                kernel_size=1, # filter size
            ),
            nn.BatchNorm2d(feature_shape[0]),
            nn.ReLU(),
        )
        self.gap_img = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.ft_img(x)        
        x = self.conv_img(x)           
        x = self.gap_img(x)

        ft = x.view(x.size(0), -1)
        x = self.fc(ft)
        return x, ft
    
class ResNet50_MultiScale_Probabilistic(nn.Module):
    """
    A multi-scale ResNet50 model that produces probabilistic outputs (alpha, beta
    for a Beta distribution) from multiple input modalities.

    This model is not a subclass of ResNet50_MultiScale but is built with similar
    principles. It uses a ResNet50_with_Feature instance as a backbone to extract
    features from each modality, fuses them, and then predicts beta distribution
    parameters and classification logits.

    Args:
        feature_shape (tuple): Shape of features from a single modality.
        modality (int): Number of input modalities.
        requires_grad (bool): Whether to train the backbone.
        num_classes (int): Number of classes for the classification head.
        pretrain_feature_extractor (dict, optional): A state_dict for the
            ResNet50_with_Feature backbone.
        dropout_rate (float): Dropout rate for regularization.
        epsilon (float): Small value to add for numerical stability.
    """
    def __init__(self, feature_shape=(2048,24,24), modality=3,
                 requires_grad=True, num_classes=2, pretrain_feature_extractor=None,
                 dropout_rate=0.7, epsilon=1e-5):
        super(ResNet50_MultiScale_Probabilistic, self).__init__()
        
        self.feature_shape = feature_shape
        self.modality = modality
        self.epsilon = epsilon
        
        # Create a temporary loader for the ResNet50_with_Feature weights
        feature_loader = ResNet50_with_Feature()
        if pretrain_feature_extractor is not None:
            feature_loader.load_state_dict(pretrain_feature_extractor)
        
        # Extract the ResNet backbone (ft_img) from the loader.
        # We take all layers except the last (avgpool) to get feature maps.
        self.ft_img = nn.Sequential(*list(feature_loader.ft_img.children())[:-1])
        
        for p in self.ft_img.parameters():
            p.requires_grad = requires_grad
            
        # Channel Fusion -- Apply 1X1 Conv kernels on the concatenated features
        self.conv_channel_fusion = nn.Sequential(
            nn.Conv2d(
                in_channels=modality * feature_shape[0], 
                out_channels=feature_shape[0],
                kernel_size=1,
            ),
            nn.BatchNorm2d(feature_shape[0]),
            nn.ReLU(),
        )        
            
        # ConvLayer for image feature refinement after fusion
        self.conv_img = nn.Sequential(
            nn.Conv2d(
                in_channels=feature_shape[0],
                out_channels=feature_shape[0],
                kernel_size=1,
            ),
            nn.BatchNorm2d(feature_shape[0]),
            nn.ReLU(),
        )
        
        self.gap_img = nn.AdaptiveAvgPool2d((1,1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Beta distribution parameters heads
        self.fc_alpha_raw = nn.Linear(2048, 1)
        self.fc_beta_raw = nn.Linear(2048, 1)
        
        # Initialize heads
        nn.init.normal_(self.fc_alpha_raw.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_alpha_raw.bias, 0.5)  
        nn.init.normal_(self.fc_beta_raw.weight, mean=0.0, std=0.01)  
        nn.init.constant_(self.fc_beta_raw.bias, 1.0)  
        
        # Classification head
        self.fc_classifier = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        # x is a list of tensors [img1, img2, img3]
        # Pass each image through the feature extractor
        ft1 = self.ft_img(x[0]) 
        ft2 = self.ft_img(x[1]) 
        ft3 = self.ft_img(x[2]) 
        
        # Concatenate the features along a new modality dimension
        # ft shape: [batch, channels, height, width]
        ft_cat = torch.stack([ft1, ft2, ft3], dim=1) # Shape: [batch, modality, C, H, W]
        
        # Reshape for channel fusion
        # From [batch, modality, C, H, W] to [batch, modality*C, H, W]
        ft_reshaped = ft_cat.view(ft_cat.size(0), -1, self.feature_shape[1], self.feature_shape[2])
        
        # Pass the concatenated feature through the channel fusion layer
        fused_ft = self.conv_channel_fusion(ft_reshaped)      
        
        # Pass the fused feature through a 1x1 conv for refinement
        fused_ft = self.conv_img(fused_ft)           
        
        fused_ft = self.gap_img(fused_ft)
        ft_flat = fused_ft.view(fused_ft.size(0), -1)
        
        # Apply dropout
        ft_dropout = self.dropout(ft_flat)
        
        # Beta distribution parameters
        alpha_raw = self.fc_alpha_raw(ft_dropout)
        beta_raw = self.fc_beta_raw(ft_dropout)
        
        # Using scaled softplus as in SingleScaleProbabilistic
        alpha = F.softplus(alpha_raw, beta=0.1) + self.epsilon
        beta = F.softplus(beta_raw, beta=0.1) + self.epsilon
        
        # Classification logits
        logits = self.fc_classifier(ft_dropout)
        
        # Risk score from Beta distribution mean
        risk_score = alpha / (alpha + beta)
        
        return {
            'features': ft_flat,
            'alpha': alpha.squeeze(-1),
            'beta': beta.squeeze(-1),
            'risk_score': risk_score.squeeze(-1),
            'logits': logits
        }

    def get_beta_distribution(self, x):
        """
        Returns the predicted Beta distribution for the given input.
        """
        output = self.forward(x)
        from torch.distributions import Beta
        return Beta(output['alpha'], output['beta'])
    
    def predict_risk(self, x):
        """
        Returns the risk score (mean of Beta distribution) for the given input.
        """
        with torch.no_grad():
            output = self.forward(x)
            return output['risk_score']