import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image


from src.bpe_encoder import BPEImageEncoder
from src.comparative_analysis import ImageClassificationAnalysis

# Set page config
st.set_page_config(
    page_title="BPE Image Classification Explorer",
    page_icon="🧩",
    layout="wide"
)

# Title and description
st.title("BPE Image Classification Explorer")
st.markdown("""
This interactive app demonstrates Byte Pair Encoding (BPE) for image classification.
Explore the dataset, visualize the encoding process, and compare classification results.
""")

# Functions to load data
@st.cache_data
def load_dataset(data_dir="data"):
    """Load a sample of images from each class"""
    classes = []
    images_by_class = {}
    
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        try:
            class_num = int(class_name)
            classes.append(class_num)
            
            # Load up to 9 images per class
            images = []
            for img_name in os.listdir(class_dir)[:9]:
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    img = cv2.resize(img, (64, 64))  # Standardize size
                    images.append(img)
            
            if images:
                images_by_class[class_num] = images
                
        except ValueError:
            continue
    
    return sorted(classes), images_by_class

@st.cache_resource
def load_results():
    """Load precomputed results from results/comparative_analysis_results.csv"""
    try:
        results_df = pd.read_csv("results/comparative_analysis_results.csv")
        return results_df
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return pd.DataFrame()

# Initialize session state for parameters if not exists
if 'vocab_size' not in st.session_state:
    st.session_state['vocab_size'] = 1000
if 'sequence_length' not in st.session_state:
    st.session_state['sequence_length'] = 1024
if 'classifier_type' not in st.session_state:
    st.session_state['classifier_type'] = "Random Forest"

# Load data
with st.spinner("Loading dataset..."):
    classes, images_by_class = load_dataset()
    results_df = load_results()

# Main interface tabs
tab1, tab2, tab3 = st.tabs(["Dataset Explorer", "BPE Visualization", "Classification Results"])

# Tab 1: Dataset Explorer
with tab1:
    st.header("Dataset Explorer")
    
    if not classes:
        st.warning("No classes found in the dataset. Make sure your data directory contains class subdirectories.")
    else:
        selected_class = st.selectbox("Select Class", classes)
        
        if selected_class in images_by_class and images_by_class[selected_class]:
            st.write(f"Showing sample images from class {selected_class}")
            
            # Display grid of images
            images = images_by_class[selected_class]
            cols = st.columns(3)
            for i, img in enumerate(images):
                cols[i % 3].image(img, caption=f"Image {i+1}", width=150)
            
            # Select image for detailed analysis
            selected_img_idx = st.number_input(
                "Select Image for Analysis", 
                min_value=1, 
                max_value=len(images), 
                value=1
            ) - 1  # Convert to 0-indexed
            
            st.session_state["selected_class"] = selected_class
            st.session_state["selected_img_idx"] = selected_img_idx
            st.session_state["selected_img"] = images[selected_img_idx]
        else:
            st.warning(f"No images found for class {selected_class}")

# Tab 2: BPE Visualization
with tab2:
    st.header("BPE Encoding Visualization")
    
    # BPE Parameters in this tab
    st.subheader("BPE Parameters")
    col1, col2 = st.columns(2)
    with col1:
        vocab_size = st.slider("Vocabulary Size", 256, 2000, st.session_state['vocab_size'])
        st.session_state['vocab_size'] = vocab_size
    with col2:
        sequence_length = st.slider("Sequence Length", 256, 2048, st.session_state['sequence_length'])
        st.session_state['sequence_length'] = sequence_length
    
    if "selected_img" in st.session_state:
        # Initialize BPE encoder with current parameters
        encoder = BPEImageEncoder(vocab_size=vocab_size, sequence_length=sequence_length)
        
        # Display original image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(st.session_state["selected_img"], caption="Original", width=200)
        
        # Encode and visualize when button is clicked
        if st.button("Apply BPE Encoding"):
            with st.spinner("Encoding image..."):
                try:
                    # Fit encoder on just this image for demonstration
                    encoder.fit([st.session_state["selected_img"]])
                    
                    # Encode and decode
                    encoded_data, original_shape = encoder.transform(st.session_state["selected_img"])
                    reconstructed = encoder.inverse_transform(encoded_data, original_shape)
                    
                    # Get stats
                    stats = encoder.get_compression_stats()
                    
                    # Display reconstructed image
                    from PIL import Image

                    with col2:
                        st.subheader("Reconstructed Image")

                        # Flatten result if needed
                        is_color = len(original_shape) == 3 and original_shape[2] == 3

                        # Only use the first height × width values for grayscale
                        try:
                            if is_color:
                                # Color image reconstruction
                                height, width, channels = original_shape
                                color_image = np.array(reconstructed).reshape((height, width, channels)).astype(np.uint8)
                                st.image(color_image, caption="After BPE", width=200)
                            else:
                                # Grayscale image reconstruction
                                height, width = original_shape[:2]
                                gray_image = np.array(reconstructed).reshape((height, width)).astype(np.uint8)
                                pil_image = Image.fromarray(gray_image, mode="L")
                                st.image(pil_image, caption="After BPE", width=200)
                        except Exception as e:
                            st.error(f"Failed to reshape reconstructed image: {e}")


                    
                    # Display compression metrics
                    st.subheader("Compression Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Compression Ratio", f"{stats.compression_ratio:.2f}x")
                    col2.metric("Unique Tokens", stats.unique_tokens)
                    col3.metric("Encoding Time", f"{stats.encoding_time*1000:.1f} ms")
                    
                    # Token distribution
                    st.subheader("Token Distribution")
                    fig, ax = plt.subplots(figsize=(7, 3))
                    ax.hist(encoded_data, bins=min(50, len(set(encoded_data))), alpha=0.7)
                    ax.set_xlabel("Token Value")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Distribution of BPE Tokens")
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error during encoding: {str(e)}")
    else:
        st.info("Select an image from the Dataset Explorer tab first")

# Tab 3: Classification Results
with tab3:
    st.header("Classification Results")
    
    # Classifier selection in this tab
    classifier_type = st.selectbox(
        "Select Classifier", 
        ["SVM", "Random Forest", "Neural Network"],
        index=["SVM", "Random Forest", "Neural Network"].index(st.session_state['classifier_type'])
    )
    st.session_state['classifier_type'] = classifier_type
    
    if not results_df.empty:
        # Filter results for the selected classifier
        classifier_results = results_df[results_df["classifier"] == classifier_type]
        
        if not classifier_results.empty:
            # Comparison of original vs BPE encoding
            st.subheader(f"Performance Comparison for {classifier_type}")
            
            # Create performance comparison chart
            fig, ax = plt.subplots(figsize=(7, 4))
            
            # Bar chart comparing metrics
            metrics = ["accuracy", "precision", "recall", "f1"]
            x = np.arange(len(metrics))
            width = 0.35
            
            # Get values for original and BPE encoding
            original_vals = classifier_results[classifier_results["encoding"] == "Original"][metrics].values[0]
            bpe_vals = classifier_results[classifier_results["encoding"] == "BPE"][metrics].values[0]
            
            # Plot bars
            ax.bar(x - width/2, original_vals, width, label="Original")
            ax.bar(x + width/2, bpe_vals, width, label="BPE")
            
            # Add labels and formatting
            ax.set_ylabel("Score")
            ax.set_title(f"{classifier_type} Performance")
            ax.set_xticks(x)
            ax.set_xticklabels([m.capitalize() for m in metrics])
            ax.legend()
            ax.set_ylim(0, 1)
            
            # Display the figure
            st.pyplot(fig)
            
            # Efficiency metrics
            st.subheader("Efficiency Comparison")
            
            # Create a DataFrame for time comparison
            time_df = classifier_results[["encoding", "train_time", "prediction_time"]]
            time_df = time_df.rename(columns={
                "train_time": "Training Time (s)",
                "prediction_time": "Prediction Time (s)"
            })
            
            # Create bar chart for efficiency metrics instead of table
            fig, ax = plt.subplots(figsize=(7, 4))
            
            # Prepare data for plotting
            metrics = ["Training Time (s)", "Prediction Time (s)"]
            x = np.arange(len(metrics))
            width = 0.35
            
            # Get values for original and BPE encoding
            orig_row = time_df[time_df["encoding"] == "Original"]
            bpe_row = time_df[time_df["encoding"] == "BPE"]
            
            # Using log scale for better visualization of potentially large differences
            orig_vals = [orig_row["Training Time (s)"].values[0], orig_row["Prediction Time (s)"].values[0]]
            bpe_vals = [bpe_row["Training Time (s)"].values[0], bpe_row["Prediction Time (s)"].values[0]]
            
            # Plot bars
            ax.bar(x - width/2, orig_vals, width, label="Original")
            ax.bar(x + width/2, bpe_vals, width, label="BPE")
            
            # Add labels and formatting
            ax.set_ylabel("Time (seconds)")
            ax.set_title(f"{classifier_type} Efficiency")
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            
            # Use log scale if the values have large differences
            if max(orig_vals + bpe_vals) / min([v for v in orig_vals + bpe_vals if v > 0]) > 10:
                ax.set_yscale('log')
                
            # Display the figure
            st.pyplot(fig)
            
            # Display speedup metrics as text
            train_speedup = orig_row["Training Time (s)"].values[0] / bpe_row["Training Time (s)"].values[0]
            prediction_speedup = orig_row["Prediction Time (s)"].values[0] / bpe_row["Prediction Time (s)"].values[0]
            
            st.write(f"💡 BPE encoding provides a **{train_speedup:.2f}x** speedup in training time and a **{prediction_speedup:.2f}x** speedup in prediction time for {classifier_type}.")
            
            # Accuracy difference
            orig_acc = classifier_results[classifier_results["encoding"] == "Original"]["accuracy"].values[0]
            bpe_acc = classifier_results[classifier_results["encoding"] == "BPE"]["accuracy"].values[0]
            
            
            # Calculate and display the difference
            acc_diff = bpe_acc - orig_acc
            if acc_diff > 0:
                st.write(f"🔼 BPE encoding improves accuracy by **{acc_diff*100:.1f}** percentage points for {classifier_type}.")
            else:
                st.write(f"🔽 BPE encoding reduces accuracy by **{abs(acc_diff)*100:.1f}** percentage points for {classifier_type}.")
                    
        else:
            st.warning(f"No results found for {classifier_type}")
    else:
        st.warning("No classification results available")

# Footer
st.markdown("---")
st.markdown("BPE Image Classification Project - Developed with Streamlit") 