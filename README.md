# Image-Restore
This repository has been created for submission to the DACON Image Restoration Competition. The goal is to restore RGB images from damaged grayscale images using a deep learning-based approach.

## 1. Summarize
손상된 GRAYSCALE 영상을 입력으로 받아 Mask Map을 추출 -> Colorization -> Inpainting을 수행하는 Application 프로젝트

## 2. Architecture
MaskMapGenerator (UNet) -> Colorization Network -> Inpaint Network

## 3. Application
<img src="mdsrc/App1.png">
<img src="mdsrc/Plot.png">
