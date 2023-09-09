# Malaria Detection using Convolutional Neural Networks

![Malaria](https://img.shields.io/badge/Malaria-Detection-brightgreen.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a Malaria Detection project that uses Convolutional Neural Networks (CNNs) implemented in TensorFlow/Keras to classify whether a given image contains malaria-infected (P) or uninfected (U) cells. The model is trained on a dataset of malaria cell images and achieves high accuracy in detection.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Malaria is a life-threatening disease that affects millions of people globally. Early and accurate detection of malaria-infected cells is critical for effective treatment. This project utilizes deep learning techniques to automate the process of malaria cell detection from microscopic images.

## Features

- Malaria-infected (P) or uninfected (U) cell classification.
- Utilizes Convolutional Neural Networks (CNNs) for image analysis.
- Achieves high accuracy in detecting malaria-infected cells.

## Dataset

The Malaria dataset used for this project can be found [here](https://link-to-your-dataset-source.com). You should download the dataset and organize it before running the code.

## Data Preprocessing

The dataset is preprocessed as follows:

- Images are resized to a standard size (IM_SIZE x IM_SIZE) and rescaled to values between 0 and 1.
- Data augmentation techniques (e.g., random rotation, horizontal flip) can be added for improved model generalization.

## Model Architecture

The model architecture is based on a modified LeNet-5 CNN:

- Input Layer (IM_SIZE x IM_SIZE x 3)
- Convolutional Layer 1 (6 filters, kernel size 3x3, ReLU activation)
- Batch Normalization
- MaxPooling Layer 1
- Convolutional Layer 2 (16 filters, kernel size 3x3, ReLU activation)
- Batch Normalization
- MaxPooling Layer 2
- Flatten Layer
- Dense Layer 1 (100 units, ReLU activation)
- Batch Normalization
- Dense Layer 2 (10 units, ReLU activation)
- Batch Normalization
- Output Layer (1 unit, sigmoid activation)

## Training

The model is trained using the Adam optimizer with a binary cross-entropy loss function. Training can be executed using the provided training dataset, and validation is performed with the validation dataset. Training history, including loss and accuracy metrics, is recorded.

## Evaluation

The model is evaluated on the test dataset to assess its performance. Evaluation metrics include loss and accuracy. Additionally, sample test images are used to demonstrate the model's predictions.

## Results

The trained model achieved high accuracy on the test dataset, with a loss of approximately 0.30 and an accuracy of 93.14%.

## Usage

To use the Malaria Detection model for your own images, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/malaria-detection.git
   cd malaria-detection
