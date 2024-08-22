import tensorflow as tf
from tensorflow.keras import layers, models

def build_unsupervised_rcnn(input_shape=(256, 256, 4)):  # 4 channels to include HOG features
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
    decoded = layers.Conv2D(4, (3, 3), activation='sigmoid', padding='same')(x)

    # Segmentation head
    seg_output = layers.Conv2D(1, (1, 1), activation='sigmoid')(encoded)

    model = models.Model(inputs, [decoded, seg_output])
    return model

def custom_loss(y_true, y_pred):
    # Reconstruction loss
    mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred[0])
    
    # Segmentation loss (encourage separation)
    seg_loss = -tf.reduce_mean(tf.image.total_variation(y_pred[1]))
    
    # Combine losses
    total_loss = mse_loss + 0.1 * seg_loss
    return total_loss