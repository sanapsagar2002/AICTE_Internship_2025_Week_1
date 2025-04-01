# AICTE_Internship_2025_Week_1

# Forest Fire Detection using Deep Learning

## Overview
This project focuses on detecting forest fires using a Deep Learning model trained on the **Wildfire Dataset** from Kaggle. The model is implemented using TensorFlow and Keras and runs in **Google Colab**.

## Dataset
The dataset is sourced from Kaggle:
- **Name**: [The Wildfire Dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset)
- **Categories**: Images classified as fire and non-fire
- **Usage**: Training, validation, and testing of the deep learning model

## Features
- Uses **Convolutional Neural Networks (CNNs)** for fire detection
- **Data Augmentation** to improve generalization
- **Evaluation Metrics**: Accuracy, loss, and visualizations
- **Visualization**: Displays dataset samples
- **Colab Compatible**: Designed to run on Google Colab with GPU support

## Setup and Installation
### Prerequisites
Ensure you have a Google Colab environment and install necessary dependencies:

```python
!pip install kagglehub
```

### Download Dataset
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("elmadafri/the-wildfire-dataset")
print("Path to dataset files:", path)
```

## Project Structure
```
├── forest_fire_detection.ipynb  # Google Colab
├── README.md                     # Project documentation
├── dataset/                       # Contains dataset images
│   ├── train/
│   ├── val/
│   ├── test/
```

## Implementation
### Import Libraries
```python
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
```

### Load and Explore Dataset
```python
train_dir = '/root/.cache/kagglehub/datasets/elmadafri/the-wildfire-dataset/versions/3/the_wildfire_dataset_2n_version/train'
val_dir = '/root/.cache/kagglehub/datasets/elmadafri/the-wildfire-dataset/versions/3/the_wildfire_dataset_2n_version/val'
test_dir = '/root/.cache/kagglehub/datasets/elmadafri/the-wildfire-dataset/versions/3/the_wildfire_dataset_2n_version/test'

# List all classes
classes = os.listdir(train_dir)
num_classes = len(classes)
print(f'Number of Classes: {num_classes}')
print(f'Classes: {classes}')
```

### Visualizing Dataset
```python
plt.figure(figsize=(12, 10))
for i in range(5):
    class_path = os.path.join(train_dir, classes[0])
    img_name = os.listdir(class_path)[i]
    img_path = os.path.join(class_path, img_name)
    img = plt.imread(img_path)
    
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.title(f'{classes[0]} \n shape: {img.shape}')
    plt.axis('off')
plt.show()
```

## Training the Model
The project includes a **CNN model** with layers like **Conv2D, MaxPooling, Flatten, Dense, and Dropout** for effective feature extraction and classification.

## Results & Evaluation
The model is evaluated using accuracy and loss plots, confusion matrix, and test predictions.

## Future Improvements
- Experiment with different architectures (ResNet, VGG16, EfficientNet)
- Implement **Transfer Learning** for better performance
- Deploy the model using Flask or Streamlit.

## License
This project is licensed under the MIT License.

