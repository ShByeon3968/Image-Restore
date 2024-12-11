import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import cv2
from backbones_unet.model.unet import Unet
from colorization.colorizers import *
from src.dataset import ColorizationImageDataset, TestImageDataset
import numpy as np
import os

# Postprocess function
def postprocess(image):
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image

# Function to generate edge map and colorize image
def process_images():
    global input_dir, output_dir
    if not input_dir:
        messagebox.showerror("Error", "Please select an image directory first!")
        return
    if not output_dir:
        messagebox.showerror("Error", "Please select an output directory first!")
        return

    # Transform 정의
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    ## 1. Edge Map 추출
    test_dataset = TestImageDataset(input_dir, transform)
    original_image, _ = test_dataset[0]  # Get the original image
    image = original_image.unsqueeze(0)

    mask_model = Unet(
        backbone='convnext_base',
        in_channels=3,
        num_classes=1
    )
    mask_pretrain = torch.load('./EdgeMapModel.pth')
    mask_model.load_state_dict(mask_pretrain)
    mask_model.eval()
    edge_map = mask_model(image)
    edge_map = edge_map.squeeze(0).permute(1, 2, 0).detach().numpy().astype(np.uint8)
    edge_map = cv2.resize(edge_map, (512, 512))

    # Save edge map
    edge_map_path = os.path.join(output_dir, 'edge_map.png')
    cv2.imwrite(edge_map_path, edge_map)

    ## 2. 흑백이미지 컬러화
    color_dataset = ColorizationImageDataset(input_dir, transform)
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    tensor_tuple = color_dataset[0]
    color_output = util.postprocess_tens(
        tensor_tuple[0], colorizer_siggraph17(tensor_tuple[1]).cpu()
    )
    color_output = cv2.resize(color_output, (512, 512))
    color_output_path = os.path.join(output_dir, 'color_output.png')
    plt.imsave(color_output_path, color_output)

    # Prepare original image for display
    original_image_np = postprocess(original_image)

    # Show results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(edge_map, cmap='gray')
    axes[1].set_title("Edge Map")
    axes[1].axis('off')

    axes[2].imshow(color_output)
    axes[2].set_title("Color Output")
    axes[2].axis('off')

    plt.show()
    messagebox.showinfo("Success", f"Images processed and saved in:\n{output_dir}")

# Function to set input directory
def select_input_directory():
    global input_dir
    input_dir = filedialog.askdirectory()
    if input_dir:
        input_dir_label.config(text=f"Input Directory: {input_dir}")

# Function to set output directory
def select_output_directory():
    global output_dir
    output_dir = filedialog.askdirectory()
    if output_dir:
        output_dir_label.config(text=f"Output Directory: {output_dir}")

# Initialize Tkinter
root = tk.Tk()
root.title("Image Processing UI")
root.geometry("500x400")

# Initialize global variables
input_dir = None
output_dir = None

# UI Elements
frame = tk.Frame(root, padx=20, pady=20)
frame.pack(expand=True)

title_label = tk.Label(frame, text="Edge Map and Colorization", font=("Helvetica", 16))
title_label.pack(pady=10)

input_button = tk.Button(frame, text="Select Input Directory", command=select_input_directory, width=30)
input_button.pack(pady=5)

input_dir_label = tk.Label(frame, text="No input directory selected", font=("Helvetica", 10))
input_dir_label.pack(pady=5)

output_button = tk.Button(frame, text="Select Output Directory", command=select_output_directory, width=30)
output_button.pack(pady=5)

output_dir_label = tk.Label(frame, text="No output directory selected", font=("Helvetica", 10))
output_dir_label.pack(pady=5)

process_button = tk.Button(frame, text="Process Images", command=process_images, width=30)
process_button.pack(pady=20)

# Run Tkinter main loop
root.mainloop()
