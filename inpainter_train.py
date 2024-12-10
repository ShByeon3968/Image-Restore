import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms 
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from model import *
from LossFunc import *
from dataset import CustomImageDataset, TestImageDataset
import os
from torchvision.utils import save_image
import cv2

# Early Stopping Helper Class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        """
        Args:
            patience (int): Number of epochs to wait without improvement before stopping training.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

EPOCH = 30
BATCH_SIZE = 8

# 샘플 저장 디렉토리
sample_dir = "samples"
os.makedirs(sample_dir, exist_ok=True)

test_dir = "./open/test_input"
output_dir = "test_results"
os.makedirs(output_dir, exist_ok=True)

# 데이터 경로 설정
input_dir = "./open/train_input_masked"
input_unmask_dir = "./open/train_input_unmasked"
gt_dir = "./open/train_gt"
mask_dir = "./open/train_mask_alpha"

# Transform 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Mask Map Generator
pretrain = torch.load('EdgeMapModel.pth',map_location="cuda")
edge_predictor = Unet(
    backbone='convnext_base', # backbone network name
    in_channels=3,            # input channels (1 for gray-scale images, 3 for RGB, etc.)
    num_classes=1,            # output channels (number of classes in your dataset)
).cuda()
edge_predictor.load_state_dict(pretrain)
edge_predictor.eval()


# Dataset 초기화
train_dataset = CustomImageDataset(input_dir,input_unmask_dir, gt_dir, mask_dir, transform=transform,mode='train')
train_loader = DataLoader(train_dataset,BATCH_SIZE,shuffle=True)
val_dataset = CustomImageDataset(input_dir,input_unmask_dir,gt_dir, mask_dir, transform=transform,mode='val')
val_loader = DataLoader(val_dataset,BATCH_SIZE,shuffle=True)

model = ColorizationAndInpaintingPipeline(True).cuda()
discriminator = PatchGANDiscriminator().cuda()

# 손실 함수 초기화
criterion_color = PerceptualLoss().cuda()
criterion_inpaint_l1 = nn.L1Loss()
criterion_gan = GANLoss()

# 옵티마이저
optimizer_gen = optim.Adam(model.parameters(), lr= 1e-5)
optimizer_disc = optim.Adam(discriminator.parameters(), lr=1e-4)
scheduler_gen = optim.lr_scheduler.ReduceLROnPlateau(optimizer_gen, mode='min', factor=0.5, patience=5, verbose=True)
scheduler_disc = optim.lr_scheduler.ReduceLROnPlateau(optimizer_disc, mode='min', factor=0.5, patience=5, verbose=True)

# Early Stopping 초기화
early_stopping = EarlyStopping(patience=10, min_delta=0.001)

# 훈련 루프
best_val_loss = np.inf
for epoch in range(EPOCH):
    model.train()
    discriminator.train()

    total_gen_loss = 0
    total_disc_loss = 0

    # 훈련 단계
    train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCH} [Training]")
    for batch_idx, (input_image, input_unmask_image,gt_rgb_image, gt_gray_image, mask_image) in enumerate(train_loader):
        input_image,input_unmask_image ,gt_rgb_image,gt_gray_image ,mask_image = input_image.cuda(), input_unmask_image.cuda() ,gt_rgb_image.cuda(), gt_gray_image.cuda(),mask_image.cuda()

        # Forward pass
        with torch.no_grad():
            EdgeMap = edge_predictor(input_image)
        inpainted_image = model.inpainter(input_image, EdgeMap)
        colored_image = model.colorizer(inpainted_image)

        # --- Inpainting Loss (L1 Loss) ---
        inpaint_l1_loss = criterion_inpaint_l1(inpainted_image, gt_gray_image)
        # --- Colorization Loss ---
        color_loss = criterion_color(colored_image, gt_rgb_image)

        # --- GAN Loss ---
        # Train Discriminator
        real_output = discriminator(gt_rgb_image, input_image)
        fake_output = discriminator(colored_image.detach(), input_image)
        disc_loss_real = criterion_gan(real_output, True)
        disc_loss_fake = criterion_gan(fake_output, False)
        disc_loss = (disc_loss_real + disc_loss_fake) * 0.5

        optimizer_disc.zero_grad()
        disc_loss.backward()
        optimizer_disc.step()

        # Train Generator
        fake_output = discriminator(colored_image, input_image)
        gan_loss = criterion_gan(fake_output, True)

        gen_loss = color_loss + inpaint_l1_loss + 0.2 * gan_loss

        optimizer_gen.zero_grad()
        gen_loss.backward()
        optimizer_gen.step()

        # Accumulate loss for monitoring
        total_gen_loss += gen_loss.item()
        total_disc_loss += disc_loss.item()

        train_loader.set_postfix({
            "Inpaint_L1_Loss": inpaint_l1_loss.item(),
            "Color_Loss": color_loss.item(),
            "GAN_Loss": gan_loss.item(),
            "Disc_Loss": disc_loss.item(),
        })

    # Validation 단계
    model.eval()
    val_gen_loss = 0
    with torch.no_grad():
        val_loader = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCH} [Validation]")
        for input_image, input_unmask_image, gt_rgb_image,gt_gray_image ,mask_image in val_loader:
            input_image, input_unmask_image, gt_rgb_image,gt_gray_image, mask_image = input_image.cuda(),input_unmask_image.cuda() ,gt_rgb_image.cuda(), gt_gray_image.cuda(),mask_image.cuda()

            EdgeMap = edge_predictor(input_image)
            inpainted_image = model.inpainter(input_unmask_image,EdgeMap)
            colored_image = model.colorizer(inpainted_image)
        
            inpaint_l1_loss = criterion_inpaint_l1(inpainted_image, gt_gray_image)
            color_loss = criterion_color(colored_image, gt_rgb_image)
            fake_output = discriminator(colored_image, input_image)
            gan_loss = criterion_gan(fake_output, True)

            gen_loss = color_loss + inpaint_l1_loss + 0.1 * gan_loss
            val_gen_loss += gen_loss.item()

    # 평균 Loss 계산
    avg_train_gen_loss = total_gen_loss / len(train_loader)
    avg_train_disc_loss = total_disc_loss / len(train_loader)
    avg_val_gen_loss = val_gen_loss / len(val_loader)

    print(f"Epoch [{epoch + 1}/{EPOCH}] - Train Gen Loss: {avg_train_gen_loss:.4f}, "
          f"Train Disc Loss: {avg_train_disc_loss:.4f}, Val Gen Loss: {avg_val_gen_loss:.4f}")

    # Scheduler 업데이트
    scheduler_gen.step(avg_val_gen_loss)
    scheduler_disc.step(avg_val_gen_loss)

    # Early Stopping 체크
    early_stopping(avg_val_gen_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered. Training stopped.")
        break
    
    test_dataset = TestImageDataset(test_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 모델 추론
    with torch.no_grad():
        for input_image, image_name in tqdm(test_loader, desc="Testing"):
            input_image = input_image.cuda()

            # 모델 추론
            EdgeMap = edge_predictor(input_image)
            inpainted_image = model.inpainter(input_image, EdgeMap)
            colored_image = model.colorizer(inpainted_image)

            # 결과 저장
            for i in range(input_image.size(0)):
                save_image(
                    torch.cat([
                        input_image[i].repeat(3, 1, 1).cpu(),  # Grayscale to RGB
                        EdgeMap[i].repeat(3, 1, 1).cpu(),     # Edge Map
                        inpainted_image[i].repeat(3, 1, 1).cpu(), # Inpainted Image
                        colored_image[i].cpu()                # Colored Image
                    ], dim=2),
                    os.path.join(output_dir, f"{image_name[i]}")
                )

    print(f"Test results saved in {output_dir}")

    # 최적의 모델 저장
    if avg_val_gen_loss < best_val_loss:
        best_val_loss = avg_val_gen_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Model saved with Val Gen Loss: {best_val_loss:.4f}")