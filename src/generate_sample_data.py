import numpy as np
import cv2
import os
import random
from typing import Tuple, List

def create_circle_pattern(size: Tuple[int, int], radius: int = 20) -> np.ndarray:
    """Create an image with circle pattern."""
    img = np.zeros(size, dtype=np.uint8)
    center = (size[1] // 2, size[0] // 2)
    cv2.circle(img, center, radius, 255, -1)
    return img

def create_textured_pattern(size: Tuple[int, int], scale: int = 4) -> np.ndarray:
    """Create a textured pattern using OpenCV's noise functions."""
    # Create base noise
    noise = np.random.normal(0, 1, (size[0]//scale, size[1]//scale))
    # Smooth the noise
    noise = cv2.GaussianBlur(noise, (scale*2+1, scale*2+1), 0)
    # Resize to original size
    noise = cv2.resize(noise, (size[1], size[0]))
    # Normalize to 0-255
    noise = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return noise

def create_linear_gradient(size: Tuple[int, int], direction: str = 'horizontal') -> np.ndarray:
    """Create a linear gradient pattern."""
    if direction == 'horizontal':
        gradient = np.linspace(0, 255, size[1]).astype(np.uint8)
        gradient = np.tile(gradient, (size[0], 1))
    else:  # vertical
        gradient = np.linspace(0, 255, size[0]).astype(np.uint8)
        gradient = np.tile(gradient.reshape(-1, 1), (1, size[1]))
    return gradient

def create_checkered_pattern(size: Tuple[int, int], checker_size: int = 8) -> np.ndarray:
    """Create a checkered pattern."""
    img = np.zeros(size, dtype=np.uint8)
    for i in range(0, size[0], checker_size):
        for j in range(0, size[1], checker_size):
            if (i//checker_size + j//checker_size) % 2 == 0:
                img[i:i+checker_size, j:j+checker_size] = 255
    return img

def create_wave_pattern(size: Tuple[int, int], frequency: float = 0.1) -> np.ndarray:
    """Create a wave pattern."""
    img = np.zeros(size, dtype=np.uint8)
    for i in range(size[0]):
        wave = int(size[1]/2 + size[1]/3 * np.sin(frequency * i))
        img[i, :wave] = 255
    return img

def apply_random_transformations(img: np.ndarray) -> np.ndarray:
    """Apply random transformations to an image."""
    # Apply random rotation
    angle = random.randint(0, 360)
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img = cv2.warpAffine(img, M, (cols, rows))
    
    # Apply random affine transformation
    if random.random() > 0.5:
        pts1 = np.float32([[0, 0], [cols, 0], [0, rows]])
        pts2 = np.float32([
            [random.randint(0, cols//8), random.randint(0, rows//8)],
            [cols - random.randint(0, cols//8), random.randint(0, rows//8)],
            [random.randint(0, cols//8), rows - random.randint(0, rows//8)]
        ])
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M, (cols, rows))
    
    return img

def add_noise(img: np.ndarray, noise_level: float = 0.2) -> np.ndarray:
    """Add random noise to an image."""
    noise = np.random.normal(0, noise_level * 255, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    return np.clip(img, 0, 255).astype(np.uint8)

def create_colored_image(pattern: np.ndarray, color_style: str = 'random') -> np.ndarray:
    """Convert grayscale pattern to color image."""
    # Create color image
    color_img = np.zeros((pattern.shape[0], pattern.shape[1], 3), dtype=np.uint8)
    
    if color_style == 'random':
        # Random colors for foreground and background
        fg_color = np.random.randint(0, 255, 3).tolist()
        bg_color = np.random.randint(0, 255, 3).tolist()
    elif color_style == 'complementary':
        # Complementary colors
        fg_color = np.random.randint(0, 255, 3).tolist()
        bg_color = [255 - c for c in fg_color]
    elif color_style == 'grayscale':
        # Just convert to 3-channel grayscale
        return cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
    else:  # color_style == 'channel'
        # Apply pattern to one random channel
        channel = random.randint(0, 2)
        color_img[:, :, channel] = pattern
        return color_img
    
    # Apply colors based on pattern
    mask = pattern > 127
    color_img[mask] = fg_color
    color_img[~mask] = bg_color
    
    return color_img

def generate_sample_images(
    num_classes: int = 3, 
    images_per_class: int = 10, 
    size: Tuple[int, int] = (64, 64),
    apply_color: bool = True
):
    """
    Generate realistic sample images for testing.
    
    Args:
        num_classes: Number of classes to generate
        images_per_class: Number of images per class
        size: Size of the generated images
        apply_color: Whether to generate color images
        
    """
    # Create data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Pattern generators for each class
    pattern_generators = [
        lambda: create_circle_pattern(size),
        lambda: create_textured_pattern(size),
        lambda: create_linear_gradient(size, direction='horizontal'),
        lambda: create_linear_gradient(size, direction='vertical'),
        lambda: create_checkered_pattern(size),
        lambda: create_wave_pattern(size)
    ]
    
    # Color styles
    color_styles = ['random', 'complementary', 'grayscale', 'channel']
    
    # Generate images for each class
    for class_idx in range(num_classes):
        class_dir = os.path.join(data_dir, str(class_idx))
        os.makedirs(class_dir, exist_ok=True)
        
        # Select pattern generator for this class
        # If we have more classes than patterns, some classes will share pattern types
        pattern_gen = pattern_generators[class_idx % len(pattern_generators)]
        
        for img_idx in range(images_per_class):
            # Generate base pattern
            pattern = pattern_gen()
            
            # Apply transformations to create variation
            pattern = apply_random_transformations(pattern)
            
            # Add noise
            noise_level = random.uniform(0.05, 0.3)
            pattern = add_noise(pattern, noise_level)
            
            # Convert to color if requested
            if apply_color:
                color_style = random.choice(color_styles)
                img = create_colored_image(pattern, color_style)
            else:
                img = pattern
            
            # Save image
            img_path = os.path.join(class_dir, f"sample_{img_idx}.png")
            cv2.imwrite(img_path, img)
    
    print(f"Sample data generated successfully! Created {num_classes} classes with {images_per_class} images each.")
    print(f"Images saved to {os.path.abspath(data_dir)}")

if __name__ == "__main__":
    generate_sample_images(
        num_classes=3,
        images_per_class=100,  # Increased from 50 to 100 images per class
        size=(64, 64),
        apply_color=True
    )
    print("Sample data generation complete with 100 images per class!") 