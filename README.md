# BPE-Based Image Classification

This project implements Byte Pair Encoding (BPE) for image classification, demonstrating how BPE can be used to efficiently encode and classify images while maintaining classification accuracy.

## Overview

The project implements a novel approach to image classification by applying Byte Pair Encoding (BPE) to image data. BPE, traditionally used in text compression and NLP tasks, is adapted here for image processing, offering several advantages:

- Efficient data compression
- Reduced memory usage
- Faster training and inference times
- Maintained classification accuracy

## Features

- BPE-based image encoding and decoding
- Support for both grayscale and color images
- Multiple classifier implementations (Random Forest, SVM, Neural Network)
- Comprehensive performance analysis and visualization
- Sample image encoding demonstrations

## Results

### Performance Comparison

![Performance Metrics](src/results/comparative_analysis_plots.png)

The above plot shows the comparison of different performance metrics across various classifiers and encoding methods:
- Accuracy comparison
- Training time comparison
- Prediction time comparison
- F1 score comparison

### Precision and Recall Analysis

![Precision and Recall](src/results/precision_recall_comparison.png)

This visualization demonstrates the precision and recall metrics for different classifiers using both original and BPE-encoded data.

### Efficiency Analysis

![Efficiency Comparison](src/results/efficiency_comparison.png)

The efficiency comparison shows the trade-off between accuracy and computational resources for different approaches.

### Sample Encodings

The project includes sample visualizations demonstrating the BPE encoding process:

![Sample 1](src/results/samples/sample_1.png)
![Sample 2](src/results/samples/sample_2.png)
![Sample 3](src/results/samples/sample_3.png)

These samples show:
- Original images
- BPE-encoded representations
- Reconstructed images

## Setup and Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
python src/comparative_analysis.py
```

## Project Structure

```
.
├── src/
│   ├── bpe_encoder.py          # BPE implementation
│   ├── comparative_analysis.py # Analysis and visualization
│   └── results/               # Generated plots and results
├── data/                      # Image dataset
└── app.py                     # Streamlit visualization app
```

## Usage

1. Prepare your image dataset in the `data` directory, organized by class:
```
data/
├── 0/
│   ├── image1.jpg
│   └── image2.jpg
├── 1/
│   ├── image3.jpg
│   └── image4.jpg
└── 2/
    ├── image5.jpg
    └── image6.jpg
```

2. Run the analysis:
```bash
python src/comparative_analysis.py
```

3. View the interactive visualizations:
```bash
streamlit run app.py
```

## Key Findings

1. **Compression Efficiency**: BPE encoding achieves significant compression ratios while maintaining image quality.
2. **Classification Performance**: BPE-encoded images maintain comparable or better classification accuracy compared to original images.
3. **Computational Efficiency**: Reduced training and inference times due to compressed representation.
4. **Memory Usage**: Lower memory requirements for storing and processing encoded images.

## Future Work

- Integration with deep learning architectures
- Support for larger image datasets
- Optimization for real-time applications
- Extension to video processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 