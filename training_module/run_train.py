import os
import numpy as np
import tensorflow as tf
from skimage import io
from model_arch import build_unsupervised_rcnn, custom_loss
from ulearning import preprocess_image, segment_image, load_user_segmented_cells

def save_model(model, model_dir='/model_storage', model_name='unsupervised_rcnn.h5'):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)
    print(f"Model saved to {model_path}")

def train_model(images, model_dir='model_storage', epochs=50, batch_size=8, user_segmented_cells=None):
    # Preprocess all images
    preprocessed_images = np.array([preprocess_image(img) for img in images])
    input_shape = preprocessed_images[0].shape
    
    model = build_unsupervised_rcnn(input_shape)
    model.compile(optimizer='adam', loss=custom_loss)
    
    for epoch in range(epochs):
        for i in range(0, len(preprocessed_images), batch_size):
            batch = preprocessed_images[i:i+batch_size]
            loss = model.train_on_batch(batch, batch)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
        
        # Save intermediate segmentation results
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs
            test_image = images[0]  # Use the first image as a test
            segmentation = segment_image(model, test_image, user_segmented_cells)
            np.save(f"{model_dir}/segmentation_epoch_{epoch+1}.npy", segmentation)
    
    save_model(model, model_dir)

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.tiff', '.tif', '.png', '.jpg', '.jpeg')):
            img = io.imread(os.path.join(folder, filename))
            images.append(img)
    return images

# Example usage
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_folder = os.path.join(script_dir, 'data')
    images = load_images(image_folder)
    
    # Check if running in GUI mode (you'll need to implement this check)
    is_gui_mode = False  # Set this based on your GUI implementation
    
    user_segmented_cells = None
    if is_gui_mode:
        # In GUI mode, prompt user to segment 1-5 cells
        user_segmented_files = []  # This should be populated by your GUI
        user_segmented_cells = load_user_segmented_cells(user_segmented_files)
    
    train_model(images, user_segmented_cells=user_segmented_cells)
    
    # After training, to segment a new image:
    model = tf.keras.models.load_model('model_storage/unsupervised_rcnn.h5', custom_objects={'custom_loss': custom_loss})
    new_image = io.imread('/data/20230930_161325_546__WellA1_ChannelCamera - Blue,Camera - Transmission,Camera - Green,Camera - Gold_Seq0000_A1_0005_Camera - Blue.tif')
    segmentation = segment_image(model, new_image, user_segmented_cells)
    np.save('segmented_output.npy', segmentation)
    print("Segmentation saved as 'segmented_output.npy'")