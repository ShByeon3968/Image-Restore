import matplotlib.pyplot as plt
from backbones_unet.model.unet import Unet
import torchvision.transforms as transforms
import cv2
from colorization.colorizers import *
from src.dataset import ColorizationImageDataset, TestImageDataset
from src.model import InpaintGenerator


# 데이터 경로 설정
input_dir = "./open/test_input"
input_unmask_dir = "./open/train_input_unmasked"
gt_dir = "./open/train_gt_mask"
mask_dir = "./open/train_mask_alpha"

# Transform 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
## 1. Edge Map 추출
test_dataset = TestImageDataset(input_dir,transform)
image, _ = test_dataset[0]
image = image.unsqueeze(0)

mask_model = Unet(
    backbone='convnext_base', # backbone network name
    in_channels=3,            # input channels (1 for gray-scale images, 3 for RGB, etc.)
    num_classes=1,            # output channels (number of classes in your dataset)
)
mask_pretrain = torch.load('./EdgeMapModel.pth')
mask_model.load_state_dict(mask_pretrain)
mask_model.eval()
edge_map = mask_model(image)
# edge_map = edge_map.squeeze(0).permute(1,2,0).detach().numpy()

## 2. 흑백이미지 컬러화
color_dataset = ColorizationImageDataset(input_dir,input_unmask_dir,gt_dir,mask_dir,transform)

# load colorizers
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

tensor_tuple = color_dataset[0]

color_output = util.postprocess_tens(tensor_tuple[0],colorizer_siggraph17(tensor_tuple[1]).cpu())
color_output = cv2.resize(color_output,(256,256))
color_output = torch.from_numpy(color_output)
color_output = color_output.permute(2,0,1).unsqueeze(0)


## Inpainter
inpaint_pretrain = torch.load('./src/pretrain/G0000000.pt')
inpaint_model = InpaintGenerator()
inpaint_model.load_state_dict(inpaint_pretrain)
inpaint_model.eval()

mask = cv2.imread('./otsu_transparent_mask.png',cv2.IMREAD_GRAYSCALE)
mask = torch.from_numpy(mask)
mask = mask.unsqueeze(0).unsqueeze(0)

inpaint_image = inpaint_model(color_output,mask)
inpaint_image = inpaint_image.squeeze(0).permute(1,2,0).detach().numpy()

cv2.imwrite('output.png',inpaint_image)
