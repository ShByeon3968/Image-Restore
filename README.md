# **Image-Restore**  
🌟 **Revolutionizing Image Restoration through Deep Learning** 🌟  
This repository is dedicated to the **DACON Image Restoration Competition**, aiming to restore vibrant **RGB images** from **damaged grayscale inputs** using state-of-the-art **deep learning techniques**.

---

## **1. Project Overview**  
Transform damaged grayscale images into fully restored and realistic RGB outputs through a cutting-edge application pipeline. The process involves:  
1. Extracting a **Mask Map** from the damaged input.  
2. **Colorizing** the grayscale image.  
3. **Inpainting** missing or damaged regions.  

---

## **2. System Architecture**  
Our restoration pipeline employs a carefully designed **three-stage neural network architecture**:  

1. **Mask Map Generator**  
   - Model: UNet  
   - Task: Identifies regions of damage in the grayscale input.  

2. **Colorization Network**  
   - Task: Converts grayscale images into vibrant RGB outputs.  

3. **Inpaint Network**  
   - Task: Refines the image by filling in missing or damaged regions.  

<p align="center">
  <img src="mdsrc/Architecture.png" alt="System Architecture" width="75%">
</p>

---

## **3. Loss Functions**  
To achieve high-quality restoration, custom loss functions have been designed for each component:  

- **Mask Map Generator (UNet)**:  
  - **L2 Loss**: Measures pixel-wise reconstruction accuracy.  
  - **Perceptual Loss**: Encourages structural consistency.  

- **Colorization Network**:  
  - **Perceptual Loss**: Focuses on visual realism and content preservation.  

- **Inpaint Network**:  
  - **L1 Loss**: Ensures pixel accuracy in reconstructions.  
  - **GAN Loss**: Adds adversarial learning to improve naturalness.  

---

## **4. Application in Action**  
Below are visualizations showcasing the restoration pipeline's output and performance:  

<p align="center">
  <img src="mdsrc/App1.png" alt="Application Preview" width="80%">
</p>  

<p align="center">
  <img src="mdsrc/Plot.png" alt="Plot Result" width="80%">
</p>  

---

## **5. To-Do List**  
🔧 **Upcoming Enhancements:**  
1. Upload the **Image Inpainting Code**.  
2. Integrate advanced metrics for qualitative and quantitative evaluation.  
3. Optimize the networks for faster inference.  
4. Add pre-trained model weights and a demo script for users.  

---

