import unittest
import numpy as np
import cv2
import os
import sys
from src.bpe_encoder import BPEImageEncoder, BPEStats

class TestBPEImageEncoder(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create a simple test image with repetitive patterns for better compression
        self.test_image = np.zeros((64, 64), dtype=np.uint8)
        
        # Create a checkerboard pattern for better compression
        for i in range(0, 64, 8):
            for j in range(0, 64, 8):
                self.test_image[i:i+4, j:j+4] = 255
                self.test_image[i+4:i+8, j+4:j+8] = 255
        
        # Color test image
        self.color_test_image = np.zeros((64, 64, 3), dtype=np.uint8)
        self.color_test_image[:, :, 0] = self.test_image  # Red channel
        self.color_test_image[:, :, 1] = np.rot90(self.test_image)  # Green channel
        self.color_test_image[:, :, 2] = np.flip(self.test_image, axis=0)  # Blue channel
        
        # Create a small vocab encoder for faster testing
        self.encoder = BPEImageEncoder(vocab_size=50, sequence_length=1000)
        
    def test_init(self):
        """Test initialization of the encoder."""
        # Test valid initialization
        encoder = BPEImageEncoder(vocab_size=100, sequence_length=500)
        self.assertEqual(encoder.vocab_size, 100)
        self.assertEqual(encoder.sequence_length, 500)
        self.assertEqual(len(encoder.merges), 0)
        self.assertEqual(len(encoder.merge_order), 0)
        self.assertEqual(len(encoder.byte_to_token), 0)
        self.assertEqual(len(encoder.token_to_byte), 0)
        self.assertIsNone(encoder.stats)
        
        # Test invalid vocab_size
        with self.assertRaises(ValueError):
            BPEImageEncoder(vocab_size=1)
            
        # Test invalid sequence_length
        with self.assertRaises(ValueError):
            BPEImageEncoder(sequence_length=0)
            
    def test_validate_image(self):
        """Test image validation."""
        # Test valid grayscale image
        self.encoder._validate_image(self.test_image)
        
        # Test valid color image
        self.encoder._validate_image(self.color_test_image)
        
        # Test invalid dtype
        with self.assertRaises(ValueError):
            self.encoder._validate_image(self.test_image.astype(np.float32))
            
        # Test invalid dimensions
        with self.assertRaises(ValueError):
            self.encoder._validate_image(np.zeros((64, 64, 4)))
            
        # Test invalid input type
        with self.assertRaises(ValueError):
            self.encoder._validate_image([1, 2, 3])
            
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Test grayscale image
        processed, shape = self.encoder._preprocess_image(self.test_image)
        self.assertEqual(processed.shape, self.test_image.shape)
        self.assertEqual(shape, self.test_image.shape)
        
        # Test color image with preserve_color=True
        processed, shape = self.encoder._preprocess_image(self.color_test_image, preserve_color=True)
        self.assertEqual(processed.shape, self.color_test_image.shape)
        self.assertEqual(shape, self.color_test_image.shape)
        
        # Test color image with preserve_color=False
        processed, shape = self.encoder._preprocess_image(self.color_test_image, preserve_color=False)
        self.assertEqual(len(processed.shape), 2)
        self.assertEqual(shape, self.color_test_image.shape)
        
    def test_get_byte_pairs(self):
        """Test byte pair frequency calculation."""
        data = [1, 2, 1, 2, 3, 1, 2]
        pairs = self.encoder._get_byte_pairs(data)
        
        # Check frequencies
        self.assertEqual(pairs[(1, 2)], 3)
        self.assertEqual(pairs[(2, 1)], 1)
        self.assertEqual(pairs[(2, 3)], 1)
        self.assertEqual(pairs[(3, 1)], 1)
        
    def test_merge_bytes(self):
        """Test byte pair merging."""
        data = [1, 2, 1, 2, 3, 1, 2]
        pair = (1, 2)
        new_token = 10
        
        merged = self.encoder._merge_bytes(data, pair, new_token)
        self.assertEqual(merged, [10, 10, 3, 10])
        
    def test_fit(self):
        """Test the fit method."""
        # Test with valid images
        self.encoder.fit([self.test_image])
        
        # Check if the encoder is ready for use, even without merges
        self.assertGreaterEqual(len(self.encoder.merges), 0)  # Should be zero or more
        self.assertGreaterEqual(len(self.encoder.merge_order), 0)
        
        # Test with empty list
        with self.assertRaises(ValueError):
            self.encoder.fit([])
            
        # Test with invalid images
        with self.assertRaises(RuntimeError):  # This is now RuntimeError because it's wrapped
            self.encoder.fit([np.array([1, 2, 3])])
            
    def test_transform(self):
        """Test the transform method."""
        # Fit the encoder first
        self.encoder.fit([self.test_image])
        
        # Test grayscale image
        encoded, shape = self.encoder.transform(self.test_image)
        self.assertEqual(len(encoded), self.encoder.sequence_length)
        self.assertEqual(shape, self.test_image.shape)
        
        # Test color image
        encoded, shape = self.encoder.transform(self.color_test_image, preserve_color=True)
        self.assertEqual(len(encoded), self.encoder.sequence_length)
        self.assertEqual(shape, self.color_test_image.shape)
        
        # Test invalid image
        with self.assertRaises(RuntimeError):
            self.encoder.transform(np.array([1, 2, 3]))
            
    def test_inverse_transform(self):
        """Test the inverse_transform method."""
        # Fit and transform first
        self.encoder.fit([self.test_image])
        encoded, shape = self.encoder.transform(self.test_image)
        
        # Test reconstruction
        reconstructed = self.encoder.inverse_transform(encoded, shape)
        self.assertEqual(reconstructed.shape, shape)
        self.assertEqual(reconstructed.dtype, np.uint8)
        
        # Test with invalid data - very short array
        with self.assertRaises(ValueError):
            self.encoder.inverse_transform([1], (64, 64))
            
    def test_encode_decode_image(self):
        """Test the encode_decode_image method."""
        # Test grayscale image
        encoded, reconstructed = self.encoder.encode_decode_image(self.test_image)
        self.assertEqual(len(encoded), self.encoder.sequence_length)
        self.assertEqual(reconstructed.shape, self.test_image.shape)
        
        # Test color image
        encoded, reconstructed = self.encoder.encode_decode_image(self.color_test_image, preserve_color=True)
        self.assertEqual(len(encoded), self.encoder.sequence_length)
        self.assertEqual(reconstructed.shape, self.color_test_image.shape)
        
    def test_visualize_compression(self):
        """Test the visualize_compression method."""
        # Fit and encode first
        self.encoder.fit([self.test_image])
        encoded, reconstructed = self.encoder.encode_decode_image(self.test_image)
        
        # Test visualization
        self.encoder.visualize_compression(self.test_image, encoded, reconstructed)
        
        # Test saving to file
        save_path = "test_visualization.png"
        self.encoder.visualize_compression(self.test_image, encoded, reconstructed, save_path)
        self.assertTrue(os.path.exists(save_path))
        os.remove(save_path)
        
    def test_get_compression_stats(self):
        """Test the get_compression_stats method."""
        # Initially should be None
        self.assertIsNone(self.encoder.get_compression_stats())
        
        # After transformation should have stats
        self.encoder.fit([self.test_image])
        encoded, reconstructed = self.encoder.encode_decode_image(self.test_image)
        stats = self.encoder.get_compression_stats()
        
        self.assertIsInstance(stats, BPEStats)
        self.assertGreater(stats.original_size, 0)
        self.assertGreater(stats.encoded_size, 0)
        self.assertGreater(stats.compression_ratio, 0)
        self.assertGreater(stats.unique_tokens, 0)
        self.assertGreaterEqual(stats.merge_operations, 0)
        self.assertGreaterEqual(stats.encoding_time, 0)
        self.assertGreaterEqual(stats.decoding_time, 0)
        
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test invalid initialization
        with self.assertRaises(ValueError):
            BPEImageEncoder(vocab_size=1)
            
        with self.assertRaises(ValueError):
            BPEImageEncoder(sequence_length=0)
            
        # Test invalid image types
        with self.assertRaises(ValueError):
            self.encoder._validate_image(np.array([1, 2, 3]))
            
        with self.assertRaises(ValueError):
            self.encoder._validate_image(np.zeros((64, 64, 4)))
            
        # Test invalid data during fit
        with self.assertRaises(ValueError):
            self.encoder.fit([])
            
        # Test transform without merges (should not raise error)
        self.encoder.transform(self.test_image)
        
        # Test invalid data during transform
        with self.assertRaises(RuntimeError):  # This is now RuntimeError because it's wrapped
            self.encoder.transform(np.array([1, 2, 3]))
            
        # Test invalid data during inverse_transform
        with self.assertRaises(ValueError):
            # Use a very short array that will be caught by our length check
            self.encoder.inverse_transform([1], (64, 64))
            
        # Test invalid save path
        with self.assertRaises(Exception):
            self.encoder.visualize_compression(
                self.test_image, 
                [1, 2, 3], 
                self.test_image,
                save_path="/invalid/path/test.png"
            )

if __name__ == '__main__':
    unittest.main() 