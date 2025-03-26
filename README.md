# Pneumonia Detection Using CNN

## Introduction
This project trains a Convolutional Neural Network (CNN) to classify chest X-ray images as either normal or indicating pneumonia. The dataset used for training and testing is the [Chest X-ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle.

## Dataset
The dataset consists of chest X-ray images categorized into:
- **Normal**: Healthy lung X-rays
- **Pneumonia**: X-rays showing signs of pneumonia

The dataset is divided into training and testing sets. A validation set is created by splitting a portion of the training data.

## Model Architecture
The model is a CNN with the following layers:
1. **Convolutional layers**: Extract features from images using filters.
2. **MaxPooling layers**: Reduce spatial dimensions and computational complexity.
3. **Flatten layer**: Converts the 2D feature maps into a 1D feature vector.
4. **Dense layers**: Fully connected layers for classification.
5. **Softmax activation**: Outputs probability scores for normal vs. pneumonia classes.

## Training
The model is compiled using:
- **Optimizer**: Adam
- **Loss function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy

An early stopping mechanism is implemented to prevent overfitting.

## Results
The model achieved an accuracy of **98%**, demonstrating strong performance in classifying pneumonia from chest X-rays.

## Usage
1. Download the dataset from Kaggle and extract it.
2. Ensure the dataset paths are correctly set in the script.
3. Run the Python script to train the model.
4. The trained model can be used for further evaluation or deployment.

## Visualization
Plots of loss and accuracy trends are generated, along with a Grad-CAM visualization to highlight important areas in X-ray images used for classification.

## References
- [Kaggle Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- TensorFlow and Keras documentation

