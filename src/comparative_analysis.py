import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from bpe_encoder import BPEImageEncoder
import time
from typing import List, Tuple, Dict, Any
import pandas as pd


class ImageClassificationAnalysis:
    def __init__(self, 
                data_dir: str, 
                vocab_size: int = 1000, 
                sequence_length: int = 1024, 
                preserve_color: bool = False,
                optimize_hyperparams: bool = False):
        """Initialize the image classification analysis.
        
        Args:
            data_dir: Directory containing the image data
            vocab_size: Size of the BPE vocabulary
            sequence_length: Fixed length for encoded sequences
            preserve_color: Whether to preserve color information in BPE encoding
            optimize_hyperparams: Whether to optimize hyperparameters of classifiers
        """
        self.data_dir = data_dir
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.preserve_color = preserve_color
        self.optimize_hyperparams = optimize_hyperparams
        self.bpe_encoder = BPEImageEncoder(vocab_size=vocab_size, sequence_length=sequence_length)
        
        # Define classifiers with default parameters
        self.classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        
        # Define hyperparameter grids for optimization
        self.param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1]
            },
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
    def load_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Load images and their labels from the data directory.
        
        Returns:
            Tuple containing list of images and list of labels
            
        Raises:
            FileNotFoundError: If data directory doesn't exist
            ValueError: If no valid images are found
        """
        images = []
        labels = []
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            error_msg = f"Data directory {self.data_dir} does not exist"
            raise FileNotFoundError(error_msg)
        
        try:
            for class_name in os.listdir(self.data_dir):
                class_dir = os.path.join(self.data_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                    
                for img_name in os.listdir(class_dir):
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                        
                    img_path = os.path.join(class_dir, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Resize images to a standard size
                        img = cv2.resize(img, (64, 64))
                        images.append(img)
                        try:
                            labels.append(int(class_name))
                        except ValueError:
                            continue
        except Exception:
            raise
        
        if len(images) == 0:
            error_msg = "No valid images found in the data directory"
            raise ValueError(error_msg)
            
        return images, labels
    
    def prepare_data(self, images: List[np.ndarray], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training by flattening images.
        
        Args:
            images: List of input images
            labels: List of corresponding labels
            
        Returns:
            Tuple of features and labels as numpy arrays
        """
        try:
            X = np.array([img.flatten() for img in images])
            y = np.array(labels)
            return X, y
        except Exception:
            raise
    
    def prepare_bpe_data(self, images: List[np.ndarray], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data using BPE encoding.
        
        Args:
            images: List of input images
            labels: List of corresponding labels
            
        Returns:
            Tuple of encoded features and labels as numpy arrays
        """
        try:
            # Fit BPE encoder on training images
            self.bpe_encoder.fit(images, preserve_color=self.preserve_color)
            
            # Transform all images
            encoded_data = []
            for i, img in enumerate(images):
                try:
                    # The transform method now returns encoded data and original shape
                    encoded, _ = self.bpe_encoder.transform(img, preserve_color=self.preserve_color)
                    encoded_data.append(encoded)
                except Exception:
                    continue
            
            X_bpe = np.array(encoded_data)
            y = np.array(labels)
            return X_bpe, y
        except Exception:
            raise
    
    def optimize_classifier(self, 
                          clf_name: str, 
                          X_train: np.ndarray, 
                          y_train: np.ndarray) -> Any:
        """Optimize classifier hyperparameters using grid search.
        
        Args:
            clf_name: Name of the classifier to optimize
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Optimized classifier
        """
        try:
            base_clf = self.classifiers[clf_name]
            param_grid = self.param_grids[clf_name]
            
            # Use subset of data for faster optimization if dataset is large
            if X_train.shape[0] > 1000:
                sample_indices = np.random.choice(X_train.shape[0], 1000, replace=False)
                X_sample = X_train[sample_indices]
                y_sample = y_train[sample_indices]
            else:
                X_sample = X_train
                y_sample = y_train
                
            # Set up grid search with cross-validation
            grid_search = GridSearchCV(base_clf, param_grid, cv=3, n_jobs=-1)
            grid_search.fit(X_sample, y_sample)
            
            # Get best estimator and parameters
            best_params = grid_search.best_params_
            best_estimator = grid_search.best_estimator_
            
            return best_estimator
        except Exception:
            raise
    
    def evaluate_classifier(self, 
                          clf, 
                          X_train: np.ndarray, 
                          X_test: np.ndarray, 
                          y_train: np.ndarray, 
                          y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate a classifier and return metrics.
        
        Args:
            clf: Classifier to evaluate
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Train classifier
            train_start = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - train_start
            
            # Make predictions
            pred_start = time.time()
            y_pred = clf.predict(X_test)
            pred_time = time.time() - pred_start
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'train_time': train_time,
                'prediction_time': pred_time
            }
            
            return metrics
        except Exception as e:
            # Return default metrics
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'train_time': 0.0,
                'prediction_time': 0.0
            }
    
    def run_comparative_analysis(self) -> pd.DataFrame:
        """Run comparative analysis between original and BPE-encoded data.
        
        Returns:
            DataFrame containing analysis results
            
        Raises:
            RuntimeError: If analysis fails
        """
        results = []
        try:
            # Load data
            images, labels = self.load_data()
            
            # Prepare original data
            X_original, y = self.prepare_data(images, labels)
            X_train_orig, X_test_orig, y_train, y_test = train_test_split(
                X_original, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Prepare BPE data
            X_bpe, y = self.prepare_bpe_data(images, labels)
            # Use same train/test split as original data
            X_train_bpe, X_test_bpe, _, _ = train_test_split(
                X_bpe, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Run analysis for each classifier
            for clf_name, base_clf in self.classifiers.items():
                
                # Optimize hyperparameters if enabled
                if self.optimize_hyperparams:
                    # Optimize for original data
                    orig_clf = self.optimize_classifier(clf_name, X_train_orig, y_train)
                    
                    # Optimize for BPE data
                    bpe_clf = self.optimize_classifier(clf_name, X_train_bpe, y_train)
                else:
                    # Use default classifiers
                    orig_clf = base_clf
                    bpe_clf = base_clf
                
                # Evaluate on original data
                orig_metrics = self.evaluate_classifier(orig_clf, X_train_orig, X_test_orig, y_train, y_test)
                orig_metrics['encoding'] = 'Original'
                orig_metrics['classifier'] = clf_name
                results.append(orig_metrics)
                
                # Evaluate on BPE data
                bpe_metrics = self.evaluate_classifier(bpe_clf, X_train_bpe, X_test_bpe, y_train, y_test)
                bpe_metrics['encoding'] = 'BPE'
                bpe_metrics['classifier'] = clf_name
                results.append(bpe_metrics)
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results)
            
            # Create results directory if it doesn't exist
            os.makedirs('results', exist_ok=True)
            
            # Save results
            results_path = 'results/comparative_analysis_results.csv'
            results_df.to_csv(results_path, index=False)
            
            # Visualize results
            self.visualize_results(results_df)
            
            return results_df
        
        except Exception as e:
            if results:
                # Try to save partial results if available
                partial_df = pd.DataFrame(results)
                partial_df.to_csv('results/partial_results.csv', index=False)
            raise RuntimeError(f"Analysis failed: {str(e)}")
    
    def visualize_results(self, results_df: pd.DataFrame):
        """Create visualizations of the comparative analysis results.
        
        Args:
            results_df: DataFrame containing analysis results
        """
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 12))
            
            # Plot 1: Accuracy comparison
            plt.subplot(2, 2, 1)
            sns.barplot(data=results_df, x='classifier', y='accuracy', hue='encoding', palette='Set2')
            plt.title('Accuracy Comparison')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)  # Accuracy is in [0,1]
            
            # Plot 2: Training time comparison
            plt.subplot(2, 2, 2)
            sns.barplot(data=results_df, x='classifier', y='train_time', hue='encoding', palette='Set2')
            plt.title('Training Time Comparison (s)')
            plt.xticks(rotation=45)
            
            # Plot 3: Prediction time comparison
            plt.subplot(2, 2, 3)
            sns.barplot(data=results_df, x='classifier', y='prediction_time', hue='encoding', palette='Set2')
            plt.title('Prediction Time Comparison (s)')
            plt.xticks(rotation=45)
            
            # Plot 4: F1 score comparison
            plt.subplot(2, 2, 4)
            sns.barplot(data=results_df, x='classifier', y='f1', hue='encoding', palette='Set2')
            plt.title('F1 Score Comparison')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)  # F1 is in [0,1]
            
            plt.tight_layout()
            plots_path = 'results/comparative_analysis_plots.png'
            plt.savefig(plots_path)
            plt.close()
            
            # Create additional detailed comparison plots
            self._create_detailed_plots(results_df)
            
        except Exception as e:
            raise
            
    def _create_detailed_plots(self, results_df: pd.DataFrame):
        """Create additional detailed comparison plots.
        
        Args:
            results_df: DataFrame containing analysis results
        """
        try:
            # Create precision/recall comparison
            plt.figure(figsize=(12, 6))
            
            # Prepare data in long format for seaborn
            metrics = ['precision', 'recall']
            long_df = pd.melt(
                results_df, 
                id_vars=['classifier', 'encoding'], 
                value_vars=metrics,
                var_name='metric', 
                value_name='score'
            )
            
            # Plot precision/recall comparison
            sns.barplot(data=long_df, x='classifier', y='score', hue='encoding', palette='Set2', alpha=0.7)
            plt.title('Precision and Recall Comparison')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            
            # Add metric markers
            for i, metric in enumerate(metrics):
                plt.axhline(y=i/len(metrics), color='gray', linestyle='--', alpha=0.3)
                plt.text(
                    results_df.shape[0] - 0.5, 
                    i/len(metrics) + 0.05,
                    metric.capitalize(),
                    fontsize=10
                )
            
            plt.tight_layout()
            plt.savefig('results/precision_recall_comparison.png')
            plt.close()
            
            # Create time efficiency plot
            plt.figure(figsize=(10, 6))
            
            # Calculate efficiency (accuracy / training time)
            efficiency_df = results_df.copy()
            efficiency_df['efficiency'] = efficiency_df['accuracy'] / (efficiency_df['train_time'] + 0.001)  # Avoid division by zero
            
            # Normalize for better visualization
            max_efficiency = efficiency_df['efficiency'].max()
            efficiency_df['efficiency'] = efficiency_df['efficiency'] / max_efficiency
            
            # Plot efficiency comparison
            sns.barplot(data=efficiency_df, x='classifier', y='efficiency', hue='encoding', palette='Set2')
            plt.title('Time Efficiency (Accuracy per Second)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('results/efficiency_comparison.png')
            plt.close()
            
        except Exception as e:
            raise
        
    def visualize_sample_encoding(self, num_samples: int = 3):
        """Visualize sample encodings to demonstrate BPE compression.
        
        Args:
            num_samples: Number of samples to visualize
        """
        try:
            # Load data
            images, _ = self.load_data()
            
            # Select a few random samples
            if len(images) < num_samples:
                num_samples = len(images)
                
            sample_indices = np.random.choice(len(images), num_samples, replace=False)
            
            # Create a directory for sample visualizations
            os.makedirs('results/samples', exist_ok=True)
            
            # Process each sample
            for i, idx in enumerate(sample_indices):
                img = images[idx]
                
                # Get encoded data and reconstructed image
                encoded_data, reconstructed = self.bpe_encoder.encode_decode_image(
                    img, preserve_color=self.preserve_color
                )
                
                # Visualize and save
                self.bpe_encoder.visualize_compression(
                    img, encoded_data, reconstructed,
                    save_path=f'results/samples/sample_{i+1}.png'
                )
                
        except Exception as e:
            raise

def main():
    """Main function to run the analysis."""
    try:
        # Initialize analysis with data directory
        analysis = ImageClassificationAnalysis(
            data_dir='../data',  # Updated path
            vocab_size=1000,
            sequence_length=1024,
            preserve_color=False,
            optimize_hyperparams=True
        )
        
        # Run analysis
        results_df = analysis.run_comparative_analysis()
        
        # Visualize results
        analysis.visualize_results(results_df)
        
        # Visualize sample encodings
        analysis.visualize_sample_encoding(num_samples=3)
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()