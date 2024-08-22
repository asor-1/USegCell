import numpy as np
from sklearn.cluster import KMeans
from skimage import color, feature, filters, segmentation
from skimage.feature import hog
from skimage.measure import find_contours
from scipy.ndimage import gaussian_filter

def apply_kmeans_clustering(feature_map, num_clusters=2):
    flat_features = feature_map.reshape((-1, feature_map.shape[-1]))
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(flat_features)
    clustered = kmeans.cluster_centers_[kmeans.labels_]
    return clustered.reshape(feature_map.shape)

def preprocess_image(image):
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Extract HOG features
    hog_features = hog(color.rgb2gray(image), pixels_per_cell=(16, 16),
                       cells_per_block=(1, 1), visualize=False)
    
    # Reshape HOG features to match image dimensions
    hog_image = hog_features.reshape((image.shape[0] // 16, image.shape[1] // 16, -1))
    hog_image = np.repeat(np.repeat(hog_image, 16, axis=0), 16, axis=1)
    
    # Combine original image with HOG features
    combined_image = np.concatenate([image, hog_image[..., np.newaxis]], axis=-1)
    return combined_image

def segment_image(model, image, user_segmented_cells=None):
    preprocessed = preprocess_image(image)
    _, segmentation = model.predict(preprocessed[np.newaxis, ...])
    segmentation = segmentation[0, ..., 0]
    
    # Apply contour tracing and spatial gradient
    segmentation = apply_advanced_segmentation(segmentation)
    
    # If user-segmented cells are provided, use them for refinement
    if user_segmented_cells is not None:
        segmentation = refine_segmentation(segmentation, user_segmented_cells)
    
    return segmentation

def apply_advanced_segmentation(segmentation):
    # Apply Gaussian filter to reduce noise
    smoothed = gaussian_filter(segmentation, sigma=1)
    
    # Compute gradient magnitude
    gradient = filters.sobel(smoothed)
    
    # Use watershed algorithm with gradient
    markers = filters.rank.gradient(smoothed, segmentation.disk(5)) > 0.02
    markers = segmentation.label(markers)
    watershed_seg = segmentation.watershed(gradient, markers)
    
    # Find contours
    contours = find_contours(watershed_seg, 0.5)
    
    # Create final segmentation mask
    final_seg = np.zeros_like(segmentation)
    for contour in contours:
        final_seg = segmentation.polygon2mask(segmentation.shape, contour)
    
    return final_seg

def refine_segmentation(segmentation, user_segmented_cells):
    # Convert user segmented cells to binary masks
    user_masks = [cell > 0 for cell in user_segmented_cells]
    
    # Extract features from user-segmented cells
    user_features = [extract_cell_features(mask) for mask in user_masks]
    
    # Refine the segmentation based on user-segmented cell features
    refined_seg = np.zeros_like(segmentation)
    for label in np.unique(segmentation):
        if label == 0:  # background
            continue
        mask = segmentation == label
        features = extract_cell_features(mask)
        
        # Compare features with user-segmented cells
        if is_similar_to_user_cells(features, user_features):
            refined_seg[mask] = 1
    
    return refined_seg

def extract_cell_features(mask):
    # Extract relevant features from a cell mask
    area = np.sum(mask)
    perimeter = len(find_contours(mask, 0.5)[0])
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return [area, perimeter, circularity]

def is_similar_to_user_cells(features, user_features, threshold=0.2):
    # Compare features with user-segmented cells
    for user_feat in user_features:
        diff = np.abs(np.array(features) - np.array(user_feat)) / np.array(user_feat)
        if np.all(diff < threshold):
            return True
    return False

# Function to load user-segmented cells (if using GUI version)
def load_user_segmented_cells(file_paths):
    return [np.load(file_path) for file_path in file_paths]