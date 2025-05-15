import numpy as np
import cv2
from collections import Counter
from typing import List, Tuple, Dict, Optional
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time


@dataclass
class BPEStats:
    """Statistics about BPE encoding performance."""
    original_size: int
    encoded_size: int
    compression_ratio: float
    unique_tokens: int
    merge_operations: int
    encoding_time: float
    decoding_time: float

class BPEImageEncoder:
    def __init__(self, vocab_size: int = 1000, sequence_length: int = 1024):
        """Initialize the BPE image encoder.
        
        Args:
            vocab_size: Size of the BPE vocabulary
            sequence_length: Fixed length for encoded sequences
            
        Raises:
            ValueError: If vocab_size or sequence_length are invalid
        """
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
            
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.merges: Dict[Tuple[int, int], int] = {}
        self.merge_order: List[Tuple[int, int]] = []
        self.byte_to_token: Dict[int, int] = {}
        self.token_to_byte: Dict[int, int] = {}
        self.stats: Optional[BPEStats] = None
        
    def _get_byte_pairs(self, data: List[int]) -> Counter:
        """
        Get frequency of byte pairs in the data.
        
        Args:
            data: List of bytes
            
        Returns:
            Counter object with byte pair frequencies
        """
        pairs = Counter()
        for i in range(len(data) - 1):
            pairs[(data[i], data[i + 1])] += 1
        return pairs
    
    def _merge_bytes(self, data: List[int], pair: Tuple[int, int], new_token: int) -> List[int]:
        """Merge a byte pair into a new token.
        
        Args:
            data: List of bytes
            pair: Byte pair to merge
            new_token: New token to use for the merged pair
            
        Returns:
            List of bytes with merged pairs
        """
        result = []
        i = 0
        while i < len(data) - 1:
            if (data[i], data[i + 1]) == pair:
                result.append(new_token)
                i += 2
            else:
                result.append(data[i])
                i += 1
                
        if i == len(data) - 1:
            result.append(data[i])
            
        return result
    
    def _validate_image(self, image: np.ndarray) -> None:
        """Validate input image.
        
        Args:
            image: Input image to validate
            
        Raises:
            ValueError: If image is invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if image.dtype != np.uint8:
            raise ValueError("Image must be of type uint8")
        if len(image.shape) not in [2, 3]:
            raise ValueError("Image must be 2D (grayscale) or 3D (color)")
        if len(image.shape) == 3 and image.shape[2] != 3:
            raise ValueError("Color images must have 3 channels")
            
    def _preprocess_image(self, image: np.ndarray, preserve_color: bool = False) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """Preprocess image for encoding.
        
        Args:
            image: Input image
            preserve_color: Whether to preserve color information
            
        Returns:
            Tuple of (processed image, original shape)
        """
        self._validate_image(image)
        original_shape = image.shape
        
        if len(image.shape) == 3 and not preserve_color:
            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        return image, original_shape
        
    def fit(self, images: List[np.ndarray], preserve_color: bool = False) -> None:
        """Fit the BPE encoder on a list of images.
        
        Args:
            images: List of input images
            preserve_color: Whether to preserve color information
            
        Raises:
            ValueError: If no valid images are provided
            RuntimeError: If fitting fails
        """
        if not images:
            raise ValueError("No images provided for fitting")
            
        try:
            start_time = time.time()
            
            # Initialize byte-to-token mapping
            self.byte_to_token = {i: i for i in range(256)}
            self.token_to_byte = {i: i for i in range(256)}
            
            # Process images in batches to save memory
            batch_size = min(10, len(images))  # Process 10 images at a time
            all_bytes = []
            
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                batch_bytes = []
                
                for j, img in enumerate(batch):
                    try:
                        processed_img, _ = self._preprocess_image(img, preserve_color)
                        batch_bytes.extend(processed_img.flatten().tolist())
                    except Exception:
                        continue
                
                if batch_bytes:
                    all_bytes.extend(batch_bytes)
            
            if not all_bytes:
                raise ValueError("No valid bytes found in images")
                
            # Perform BPE merges
            current_vocab_size = 256
            while current_vocab_size < self.vocab_size:
                pairs = self._get_byte_pairs(all_bytes)
                if not pairs:
                    break
                    
                best_pair = max(pairs.items(), key=lambda x: x[1])[0]
                new_token = current_vocab_size
                
                self.merges[best_pair] = new_token
                self.merge_order.append(best_pair)
                self.byte_to_token[new_token] = new_token
                self.token_to_byte[new_token] = new_token
                
                # Apply merge to all bytes
                all_bytes = self._merge_bytes(all_bytes, best_pair, new_token)
                current_vocab_size += 1
            
        except Exception as e:
            raise RuntimeError(f"BPE fitting failed: {str(e)}")
            
    def transform(self, image: np.ndarray, preserve_color: bool = False) -> Tuple[List[int], Tuple[int, ...]]:
        """Transform an image using learned BPE encoding.
        
        Args:
            image: Input image
            preserve_color: Whether to preserve color information
            
        Returns:
            Tuple of (encoded data, original shape)
            
        Raises:
            RuntimeError: If transformation fails
        """
        try:
            start_time = time.time()
            
            # Preprocess image
            processed_img, original_shape = self._preprocess_image(image, preserve_color)
            data = processed_img.flatten().tolist()
            
            # Apply BPE merges
            for pair, new_token in self.merges.items():
                data = self._merge_bytes(data, pair, new_token)
                
            # Pad or truncate to sequence length
            if len(data) < self.sequence_length:
                data.extend([0] * (self.sequence_length - len(data)))
            else:
                data = data[:self.sequence_length]
                
            # Record statistics
            self.stats = BPEStats(
                original_size=processed_img.size,
                encoded_size=len(data),
                compression_ratio=processed_img.size / len(data),
                unique_tokens=len(set(data)),
                merge_operations=len(self.merges),
                encoding_time=time.time() - start_time,
                decoding_time=0.0  # Will be updated in inverse_transform
            )
            
            return data, original_shape
            
        except Exception as e:
            raise RuntimeError(f"Image transformation failed: {str(e)}")
            
    def inverse_transform(self, encoded_data: List[int], original_shape: Tuple[int, ...]) -> np.ndarray:
        """Inverse transform encoded data back to image format.
        
        Args:
            encoded_data: Encoded data to transform
            original_shape: Original image shape
            
        Returns:
            Reconstructed image
            
        Raises:
            ValueError: If input data is invalid or cannot be reshaped to original shape
            RuntimeError: If inverse transformation fails for other reasons
        """
        if len(encoded_data) < 2:
            raise ValueError("Encoded data is too short for reconstruction")
            
        try:
            start_time = time.time()
            
            # Calculate target size based on original shape
            target_size = np.prod(original_shape)
            
            # Precompute reverse token mapping for efficiency
            reverse_tokens = {}
            for pair, token in self.merges.items():
                reverse_tokens[token] = pair
            
            # Process in a more efficient way using recursion
            result = []
            for token in encoded_data:
                # Expand token recursively
                expanded = self._expand_token(token, reverse_tokens)
                result.extend(expanded)
            
            # Ensure we have the right number of bytes for reshaping
            if len(result) > target_size:
                result = result[:target_size]
            elif len(result) < target_size:
                result.extend([0] * (target_size - len(result)))
            
            print("Decoded pixel range:", min(result), max(result))
            print("Decoded length:", len(result))
            print("Expected size (from original shape):", np.prod(original_shape))


            # Reshape to original dimensions
            if len(original_shape) == 3 and original_shape[2] == 3:
        # Color image - make sure we have enough pixels
                total_pixels = original_shape[0] * original_shape[1] * original_shape[2]
                if len(result) < total_pixels:
                    result.extend([0] * (total_pixels - len(result)))
                reconstructed = np.array(result, dtype=np.uint8).reshape(original_shape)
            else:
                # Grayscale image
                height, width = original_shape[:2]
                total_pixels = height * width
                if len(result) < total_pixels:
                    result.extend([0] * (total_pixels - len(result)))
                reconstructed = np.array(result, dtype=np.uint8).reshape((height, width))


            
            # Update decoding time in stats
            if self.stats:
                self.stats.decoding_time = time.time() - start_time
            
            return reconstructed
            
        except ValueError as e:
            raise ValueError(f"Cannot reshape data to original shape: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Inverse transformation failed: {str(e)}")
    
    def _expand_token(self, token: int, reverse_tokens: Dict[int, Tuple[int, int]]) -> List[int]:
        """Recursively expand a token into its constituent bytes.
        
        Args:
            token: Token to expand
            reverse_tokens: Mapping from tokens to byte pairs
            
        Returns:
            List of expanded bytes
        """
        if token in reverse_tokens:
            a, b = reverse_tokens[token]
            return self._expand_token(a, reverse_tokens) + self._expand_token(b, reverse_tokens)
        else:
            return [token]
            
    def encode_decode_image(self, image: np.ndarray, preserve_color: bool = False) -> Tuple[List[int], np.ndarray]:
        """Encode and decode an image to check reconstruction quality.
        
        Args:
            image: Input image
            preserve_color: Whether to preserve color information
            
        Returns:
            Tuple of (encoded data, reconstructed image)
        """
        encoded_data, original_shape = self.transform(image, preserve_color)
        reconstructed = self.inverse_transform(encoded_data, original_shape)
        return encoded_data, reconstructed
        
    def visualize_compression(self, 
                            original: np.ndarray, 
                            encoded_data: List[int], 
                            reconstructed: np.ndarray,
                            save_path: Optional[str] = None) -> None:
        """Visualize compression results.
        
        Args:
            original: Original image
            encoded_data: Encoded data
            reconstructed: Reconstructed image
            save_path: Optional path to save visualization
            
        Raises:
            RuntimeError: If visualization fails
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot original image
            axes[0].imshow(original, cmap='gray' if len(original.shape) == 2 else None)
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            # Plot encoded data as histogram
            axes[1].hist(encoded_data, bins=50)
            axes[1].set_title('Encoded Data Distribution')
            axes[1].set_xlabel('Token Value')
            axes[1].set_ylabel('Frequency')
            
            # Plot reconstructed image
            axes[2].imshow(reconstructed, cmap='gray' if len(reconstructed.shape) == 2 else None , vmin=0, vmax=255)
            axes[2].set_title('Reconstructed')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save or show
            if save_path:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            raise RuntimeError(f"Visualization failed: {str(e)}")
            
    def get_compression_stats(self) -> Optional[BPEStats]:
        """Get statistics about the last compression operation.
        
        Returns:
            BPEStats object if available, None otherwise
        """
        return self.stats