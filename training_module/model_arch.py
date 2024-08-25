import tensorflow as tf
from tensorflow.keras import layers, models

def build_unsupervised_rcnn(input_shape):
    # Encoder (Feature Extractor)
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2))(x)

    # Decoder (for reconstruction loss)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    # Combined output: reconstruction + segmentation
    output = layers.Conv2D(input_shape[-1] + 1, (3, 3), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs, output)
    return model

def custom_loss(y_true, y_pred):
    # Split the prediction into reconstruction and segmentation
    y_pred_recon = y_pred[..., :-1]
    y_pred_seg = y_pred[..., -1:]

    # Reconstruction loss (Mean Squared Error)
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred_recon))
    
    # Segmentation loss (encourage separation)
    seg_loss = tf.reduce_mean(tf.image.total_variation(y_pred_seg))
    
    # Combine losses
    total_loss = mse_loss + 0.1 * seg_loss
    return total_loss