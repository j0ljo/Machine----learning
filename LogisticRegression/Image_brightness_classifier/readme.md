# 🖼️ Image Brightness Classifier

A tiny computer vision + machine learning project that decides if an image is **Bright (1)** or **Dark (0)**.

We take the **average RGB values** of an image, feed them into a **sigmoid neuron**, and let it learn the decision boundary — no hardcoded thresholds.

---

## How It Works
1. **Feature Extraction**  
   - Load the image using OpenCV  
   - Convert BGR → RGB  
   - Calculate the average R, G, B values across all pixels

2. **Model**  
   - A single **sigmoid neuron**:  
     \[
     \hat{y} = \sigma(w_R R + w_G G + w_B B + b)
     \]
   - Trained with **gradient descent** on labeled examples

3. **Prediction**  
   - If output > 0.5 → Bright (1)  
   - Else → Dark (0)

---
