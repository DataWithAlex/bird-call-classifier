# Bird Call Classifier Project

## Overview
This project focuses on developing a machine learning model to classify bird species based on their calls. The primary goal is to transform raw audio data into structured representations, extract meaningful features using the VGGish model, and train various classifiers, including Random Forest, Support Vector Machine (SVM), and Neural Networks. The performance of these models is evaluated using metrics such as accuracy, precision, recall, and F1-score, and the results are visualized to compare their effectiveness.

### Full Research Report Linked Below

https://drive.google.com/file/d/1IcBMQZPly0PNGvMnhvHOJL0oIaQFgQ-s/view?usp=sharing 

## Introduction
Bird call classification is a challenging task that involves identifying bird species based on their vocalizations. This capability has significant implications for ecological research, wildlife conservation, and environmental monitoring. Humans can learn to recognize bird species by their calls through experience; however, automating this process using machine learning requires converting audio signals into numerical representations that machines can interpret. This project aims to develop and evaluate a machine learning-based bird call classifier using audio recordings from three bird species: Blue Jay, Cardinal, and Red-bellied Woodpecker.

## Installation and Setup

### Requirements
- Python 3.7+
- Jupyter Notebook
- TensorFlow 1.x (for VGGish model)
- PyTorch
- librosa
- noisereduce
- scikit-learn
- seaborn
- matplotlib

### Directory Structure
Ensure the following directory structure in your working directory:

├── /data
├── /vggish
├── /models
└── ├── BirdCallClassifier.ipynb

### Google Drive Links for Data
- **Models Folder:** [Google Drive Link](https://drive.google.com/drive/folders/1221pOi4EO8e-Lc1FVUl4_M2t0aGDgfpY?usp=sharing)
- **VGGish Folder:** [Google Drive Link](https://drive.google.com/drive/folders/1E6aBAhtlAhxzegRgBomb6D9EpRHqombT?usp=sharing)
- **Data Folder:** [Google Drive Link](https://drive.google.com/drive/folders/1-jl6V391PYS8xcrOtAPan6rBnPV25RXk?usp=sharing)

### Steps to Download Required Files
Use the following code to download the required files into the respective directories:
```python
import gdown

# Download models folder
gdown.download_folder('https://drive.google.com/drive/folders/1221pOi4EO8e-Lc1FVUl4_M2t0aGDgfpY?usp=sharing', output='models')

# Download VGGish folder
gdown.download_folder('https://drive.google.com/drive/folders/1E6aBAhtlAhxzegRgBomb6D9EpRHqombT?usp=sharing', output='vggish')

# Download data folder
gdown.download_folder('https://drive.google.com/drive/folders/1-jl6V391PYS8xcrOtAPan6rBnPV25RXk?usp=sharing', output='data')

```

Open the `BirdCallClassifier.ipynb` notebook and follow the instructions within the notebook to load and preprocess the data, train the classifiers, and evaluate their performance.


## Data Processing

### Data Collection
The data was collected from xeno-canto, a website containing wildlife audio recordings often used by researchers. Recordings were sorted by highest quality, and only those without interfering bird calls, other species, or background noise were downloaded. All recordings underwent extensive data cleaning for model training.

### Data Preprocessing
- **Loading:** I used the librosa library to load audio files at a sample rate of 16,000 Hz. This standardization ensures that all audio data has a consistent sample rate.
- **Filtering:** Audio processed through a high-pass filter removes low-frequency noise and rumble below 100 Hz, designed using a Butterworth filter. Noise reduction using the noisereduce library further cleans the audio signal while preserving the integrity of the bird calls.
- **Segmentation:** The audio signal is segmented into fixed-length segments of 2 seconds each. Energy-based filtering ensures that only segments with significant audio content are retained, reducing irrelevant or noisy data.

### Feature Extraction
- **Log Mel Spectrograms:** The segmented audio is converted into log-mel spectrograms using the wave_to_examples function from the VGGish library. This conversion involves transforming the raw audio waveform into a time-frequency representation, capturing the intensity of different frequency components over time.
- **Embedding Extraction:** The log-mel spectrograms are fed into the VGGish model to generate embeddings. This step involves projecting the input spectrograms through a series of convolutional layers, extracting hierarchical audio features. These features are condensed into a fixed-size embedding vector.
- **Embedding Post-Processing:** The raw embeddings are post-processed to ensure consistency and improve their usability. This step applies Principal Component Analysis (PCA) to the embeddings, reducing their dimensionality and normalizing the feature distribution.

## Hypotheses
- **Distinct Clustering Hypothesis:** The embeddings for each bird species form distinct clusters in the PCA space, indicating that the feature extraction process successfully captures unique characteristics of each species' calls.
- **Species-Specific Accuracy Hypothesis:** The distinct clusters observed in the PCA plot suggest that certain species may exhibit higher classification accuracy compared to others due to more distinct vocalization patterns.

## Model Training and Evaluation

### Training and Testing Split
The dataset was divided into training and testing sets to evaluate the performance of various classifiers. The split ensured an unbiased evaluation by retaining a significant portion of the data for testing.

### Training Summary
The log-mel spectrograms and embeddings were used to train the classifiers. The VGGish model processes the input spectrograms through a series of convolutional layers, generating hierarchical audio features. These features were then condensed into embeddings used for classification tasks.

### Supervised Learning Results
The performance of the classifiers was evaluated using various metrics, including precision, recall, F1-score, and accuracy. The results provide a comprehensive view of the classifiers' effectiveness in distinguishing between bird species based on their calls.

### Unsupervised Learning Results
The performance of various clustering models was evaluated using silhouette scores, adjusted Rand Index (ARI), and clustering accuracy. These metrics provide insights into the clustering models' ability to group bird call embeddings effectively.

## Conclusion
The study highlights the importance of selecting appropriate models for bird call classification tasks. Supervised learning models, particularly CNNs and Random Forests, excel at this task when trained with well-processed embeddings. Unsupervised learning approaches can benefit from further refinement to improve clustering accuracy and better capture the underlying structure of bird call data. These insights contribute to the broader understanding of machine learning applications in ecological and environmental research.

