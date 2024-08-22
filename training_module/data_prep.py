# This is where we will data prep the images if necessary
import cv2
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def preprocess_image(image, target_size=(256, 256)):
    # Resize image
    image = cv2.resize(image, target_size)
    # Normalize the image
    image = image.astype('float32') / 255.0
    return image

def preprocess_images(images, target_size=(256, 256)):
    return [preprocess_image(img, target_size) for img in images]

# Example usage:
# images = load_images_from_folder('path/to/your/folder')
# preprocessed_images = preprocess_images(images)
