import unittest
import os
import shutil
import numpy as np
import cv2
from src.comparative_analysis import ImageClassificationAnalysis

class TestImageClassificationAnalysis(unittest.TestCase):
    def setUp(self):
        # Create a test directory with sample data
        self.num_test_images = 5
        os.makedirs("test_data/0", exist_ok=True)
        os.makedirs("test_data/1", exist_ok=True)
        
        # Create test images
        for i in range(self.num_test_images):
            # Class 0: diagonal pattern
            img0 = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.line(img0, (0, 0), (63, 63), (255, 255, 255), 2)
            cv2.imwrite(f"test_data/0/img_{i}.png", img0)
            
            # Class 1: horizontal pattern
            img1 = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.line(img1, (0, 32), (63, 32), (255, 255, 255), 2)
            cv2.imwrite(f"test_data/1/img_{i}.png", img1)
            
        # Initialize analysis
        self.analysis = ImageClassificationAnalysis(
            data_dir="test_data",
            vocab_size=100,
            sequence_length=512,
            optimize_hyperparams=False
        )
        
    def tearDown(self):
        # Clean up test directory
        if os.path.exists("test_data"):
            shutil.rmtree("test_data")
            
    def test_init(self):
        """Test initialization of the analysis class."""
        self.assertEqual(self.analysis.data_dir, "test_data")
        self.assertEqual(self.analysis.vocab_size, 100)
        self.assertEqual(self.analysis.sequence_length, 512)
        self.assertFalse(self.analysis.preserve_color)
        self.assertFalse(self.analysis.optimize_hyperparams)
        
    def test_load_data(self):
        """Test data loading."""
        images, labels = self.analysis.load_data()
        self.assertEqual(len(images), 2 * self.num_test_images)
        self.assertEqual(len(labels), 2 * self.num_test_images)
        self.assertEqual(set(labels), {0, 1})
        
    def test_prepare_data(self):
        """Test data preparation."""
        images, labels = self.analysis.load_data()
        X, y = self.analysis.prepare_data(images, labels)
        self.assertEqual(X.shape, (2 * self.num_test_images, 64 * 64 * 3))
        self.assertEqual(y.shape, (2 * self.num_test_images,))
        
    def test_prepare_bpe_data(self):
        """Test BPE data preparation."""
        images, labels = self.analysis.load_data()
        X_bpe, y = self.analysis.prepare_bpe_data(images, labels)
        self.assertEqual(X_bpe.shape, (2 * self.num_test_images, 512))
        self.assertEqual(y.shape, (2 * self.num_test_images,))
        
    def test_run_comparative_analysis(self):
        """Test the full comparative analysis."""
        results = self.analysis.run_comparative_analysis()
        self.assertIsNotNone(results)
        
        # Check that we have results for all classifiers
        self.assertIn('classifier', results.columns)
        classifier_names = set(results['classifier'].unique())
        self.assertIn('Random Forest', classifier_names)
        self.assertIn('SVM', classifier_names)
        self.assertIn('Neural Network', classifier_names)
        
        # Verify encodings
        self.assertIn('encoding', results.columns)
        encoding_names = set(results['encoding'].unique())
        self.assertIn('Original', encoding_names)
        self.assertIn('BPE', encoding_names)
        
if __name__ == '__main__':
    unittest.main() 