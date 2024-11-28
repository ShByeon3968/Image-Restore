import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetGenerator, self).__init__()

        def down_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        def up_block(in_feat, out_feat, dropout=0.0):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(out_feat),
                      nn.ReLU(inplace=True)]
            if dropout:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)
        
        self.down1 = down_block(in_channels, 64, normalize=False)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        self.down5 = down_block(512, 512)
        self.down6 = down_block(512, 512)
        self.down7 = down_block(512, 512)
        self.down8 = down_block(512, 512, normalize=False)

        self.up1 = up_block(512, 512, dropout=0.5)
        self.up2 = up_block(1024, 512, dropout=0.5)
        self.up3 = up_block(1024, 512, dropout=0.5)
        self.up4 = up_block(1024, 512)
        self.up5 = up_block(1024, 256)
        self.up6 = up_block(512, 128)
        self.up7 = up_block(256, 64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        u8 = self.up8(torch.cat([u7, d1], 1))

        return u8
    
# Contextual Attention Module
class ContextualAttentionModule(nn.Module):
    def __init__(self):
        super(ContextualAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Simple example of Contextual Attention
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.conv4(x)
    
# Edge Prediction Network
class EdgePredictor(nn.Module):
    def __init__(self):
        super(EdgePredictor, self).__init__()
        self.unet = UNetGenerator(in_channels=2, out_channels=1)  # Grayscale + Mask → Edge Map

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)
        return self.unet(x)


# Colorization Network
class ColorizationNetwork(nn.Module):
    def __init__(self):
        super(ColorizationNetwork, self).__init__()
        self.unet = UNetGenerator(in_channels=2, out_channels=3)  # Grayscale + Edge Map → RGB

    def forward(self, grayscale, edge_map):
        x = torch.cat([grayscale, edge_map], dim=1)
        return self.unet(x)


# Inpainting Network with Contextual Attention
class ContextualAttentionInpainting(nn.Module):
    def __init__(self):
        super(ContextualAttentionInpainting, self).__init__()
        self.attention = ContextualAttentionModule()  # 전역적 Attention 활용
        self.unet = UNetGenerator(in_channels=4, out_channels=3)  # 채색된 이미지 + Mask → 복구된 이미지

    def forward(self, colored_img, mask):
        x = torch.cat([colored_img, mask], dim=1)
        attention_out = self.attention(x)  # Contextual Attention 적용
        return self.unet(attention_out)
    
# 통합 파이프라인
class ColorizationAndInpaintingPipeline(nn.Module):
    def __init__(self):
        super(ColorizationAndInpaintingPipeline, self).__init__()
        self.edge_predictor = EdgePredictor()
        self.colorizer = ColorizationNetwork()
        self.inpainter = ContextualAttentionInpainting()

    def forward(self, grayscale_img, mask):
        # Step 1: Edge Prediction
        edge_map = self.edge_predictor(grayscale_img, mask)
        
        # Step 2: Colorization
        colored_img = self.colorizer(grayscale_img, edge_map)
        
        # Step 3: Inpainting
        final_img = self.inpainter(colored_img, mask)
        
        return final_img
    

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=4):
        super(PatchGANDiscriminator, self).__init__()   

        def block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.model = nn.Sequential(
            block(in_channels, 64, normalization=False),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, input_img, output_img):
        combined_img = torch.cat((input_img, output_img), 1)
        return self.model(combined_img)