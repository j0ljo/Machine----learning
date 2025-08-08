import cv2
import numpy as np

image = cv2.imread('Dark.png') # Reads in BGR format

# Convert BGR -> RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Calculate avergage RGB values
avg_rgb = np.mean(image_rgb, axis=(0,1)) # (R_avg, G_avg, B_avg)

# Convert RGB to brightness
brightness = 0.299*avg_rgb[0] + 0.587*avg_rgb[1] + 0.114*avg_rgb[2]

# Classify

threshold = 127 # Midpoint of 0-255 scale
if brightness > threshold:
    label = 1 # Bright
else:
    label = 0 # Dark

print("Classification: ", "Bright" if label == 1 else "Dark")
