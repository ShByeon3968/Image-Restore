import torch.nn as nn
import torch
from torchvision.models import vgg16

# Perceptual Loss (VGG 기반)
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features[:16])).eval()  # Conv3_3까지 사용
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.l2_loss = nn.MSELoss()

    def forward(self, pred, target):
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        return self.l2_loss(pred_features, target_features)


# GAN Loss
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True):
        super(GANLoss, self).__init__()
        self.real_label = 1.0
        self.fake_label = 0.0
        if use_lsgan:
            self.loss = nn.MSELoss()  # Least Squares GAN Loss
        else:
            self.loss = nn.BCELoss()  # Binary Cross-Entropy Loss

    def get_target_tensor(self, prediction, target_is_real):
        target_tensor = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return target_tensor

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)