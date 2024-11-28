import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms 
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from model import *
from LossFunc import *
from dataset import CustomImageDataset

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

EPOCH = 20
BATCH_SIZE = 16


# 데이터 경로 설정
input_dir = "./open/train_input_masked"
gt_dir = "./open/train_gt"
mask_dir = "./open/train_mask"

# Transform 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Dataset 초기화
train_dataset = CustomImageDataset(input_dir, gt_dir, mask_dir, transform=transform,mode='train')
train_loader = DataLoader(train_dataset,BATCH_SIZE,shuffle=True)
val_dataset = CustomImageDataset(input_dir, gt_dir, mask_dir, transform=transform,mode='val')
val_loader = DataLoader(train_dataset,BATCH_SIZE,shuffle=True)


model = ColorizationAndInpaintingPipeline().cuda()
discriminator = PatchGANDiscriminator().cuda()

# 손실 함수 초기화
criterion_edge = nn.BCEWithLogitsLoss()
criterion_color = PerceptualLoss().cuda()
criterion_inpaint_l1 = nn.L1Loss()
criterion_gan = GANLoss()

# 옵티마이저
optimizer_gen = optim.Adam(model.parameters(), lr=1e-4)
optimizer_disc = optim.Adam(discriminator.parameters(), lr=4e-4)
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
    for batch_idx, (input_image, gt_image, mask_image) in enumerate(train_loader):
        input_image, gt_image, mask_image = input_image.cuda(), gt_image.cuda(), mask_image.cuda()

        # Forward pass
        edge_pred = model.edge_predictor(input_image, mask_image)
        colored_image = model.colorizer(input_image, edge_pred)
        inpainted_image = model.inpainter(colored_image, mask_image)

        # --- Edge Prediction Loss ---
        edge_loss = criterion_edge(edge_pred, mask_image)

        # --- Colorization Loss ---
        color_loss = criterion_color(colored_image, gt_image)

        # --- Inpainting Loss (L1 Loss) ---
        inpaint_l1_loss = criterion_inpaint_l1(inpainted_image, gt_image)

        # --- GAN Loss ---
        # Train Discriminator
        real_output = discriminator(gt_image, input_image)
        fake_output = discriminator(inpainted_image.detach(), input_image)
        disc_loss_real = criterion_gan(real_output, True)
        disc_loss_fake = criterion_gan(fake_output, False)
        disc_loss = (disc_loss_real + disc_loss_fake) * 0.5

        optimizer_disc.zero_grad()
        disc_loss.backward()
        optimizer_disc.step()

        # Train Generator
        fake_output = discriminator(inpainted_image, input_image)
        gan_loss = criterion_gan(fake_output, True)

        gen_loss = edge_loss + color_loss + inpaint_l1_loss + 0.1 * gan_loss

        optimizer_gen.zero_grad()
        gen_loss.backward()
        optimizer_gen.step()

        # Accumulate loss for monitoring
        total_gen_loss += gen_loss.item()
        total_disc_loss += disc_loss.item()

        train_loader.set_postfix({
            "Edge_Loss": edge_loss.item(),
            "Color_Loss": color_loss.item(),
            "Inpaint_L1_Loss": inpaint_l1_loss.item(),
            "GAN_Loss": gan_loss.item(),
            "Disc_Loss": disc_loss.item(),
        })

    # Validation 단계
    model.eval()
    val_gen_loss = 0
    with torch.no_grad():
        val_loader = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCH} [Validation]")
        for input_image, gt_image, mask_image in val_loader:
            input_image, gt_image, mask_image = input_image.cuda(), gt_image.cuda(), mask_image.cuda()

            edge_pred = model.edge_predictor(input_image, mask_image)
            colored_image = model.colorizer(input_image, edge_pred)
            inpainted_image = model.inpainter(colored_image, mask_image)

            edge_loss = criterion_edge(edge_pred, mask_image)
            color_loss = criterion_color(colored_image, gt_image)
            inpaint_l1_loss = criterion_inpaint_l1(inpainted_image, gt_image)
            fake_output = discriminator(inpainted_image, input_image)
            gan_loss = criterion_gan(fake_output, True)

            gen_loss = edge_loss + color_loss + inpaint_l1_loss + 0.1 * gan_loss
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

    # 최적의 모델 저장
    if avg_val_gen_loss < best_val_loss:
        best_val_loss = avg_val_gen_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Model saved with Val Gen Loss: {best_val_loss:.4f}")