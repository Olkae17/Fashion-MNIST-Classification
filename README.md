
## Fashion MNIST Classification Project ##

## Overview

This project demonstrates how to build a Convolutional Neural Network (CNN) using both "Python (Keras)" and "R (Keras with TensorFlow backend)" to classify images from the "Fashion MNIST dataset". It simulates a real-world task at Microsoft AI, where accurate classification of fashion-related images supports targeted marketing.

## Contents

- "Fashion MNIST Classification.ipynb" — Python Jupyter Notebook implementing the CNN and making predictions.
- "Fashion MNIST Classification.R" — R script that performs the same tasks using the `keras` and `tensorflow` packages.
- "README.md" — Project explanation and usage instructions.

## Requirements

1. Python
- Python 3.10 or lower (TensorFlow does not yet support 3.13)
- "tensorflow"
- "keras"
- "matplotlib"
- "numpy"

To install dependencies:

2. R
- R 4.4.x
- "keras", "tensorflow", "ggplot2", "tidyr"


## Project Tasks

## CNN Implementation (Python & R) ##

A 6-layer CNN model was built with the following architecture:

1. Conv2D (32 filters)
2. MaxPooling2D
3. Conv2D (64 filters)
4. MaxPooling2D
5. Flatten
6. Dense (128 units)
7. Dense (10 units with softmax)

This model is trained for 15 epochs with a batch size of 128 and 10% validation split.

## Preprocessing

- Pixel values were normalized by dividing by 255.
- Input shape reshaped to `(28, 28, 1)` to fit Keras expectations.
- Output labels were converted to one-hot encoding using "to_categorical()".


## 🧾 How to Use

### Python

Run "Fashion MNIST Classification.ipynb" in Jupyter Lab or Notebook.

## R

Open "Fashion MNIST Classification.R" in RStudio and execute line by line.
