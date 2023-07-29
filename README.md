# Flower Species Recognition - AI Application

![Flowers](assets/Flowers.png)

## Introduction

I have developed an AI application for flower species recognition. The application uses deep learning to train an image classifier capable of recognizing different species of flowers. When provided with an image, the trained classifier can predict the species of the flower. This technology can be integrated into a smartphone app, allowing users to identify flowers through their phone's camera.

## Dataset

For this project, I used the [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). The dataset contains images of 102 different flower categories.

## Project Overview

The project was divided into several key steps:

1. Data Loading and Preprocessing: I utilized the `torchvision` library to load and preprocess the dataset. The data was split into three sets: training, validation, and testing.

2. Building and Training the Classifier: A pre-trained deep learning model from `torchvision.models` served as the feature extractor. I built a new, untrained feed-forward classifier on top of it and trained it on the flower images using backpropagation.

3. Testing the Trained Network: I evaluated the performance of the trained model on a separate test set to assess how well it generalized to new, unseen data.

4. Saving and Loading the Model: I saved the trained model and other necessary information to a checkpoint file. This enables loading the model later for making predictions without retraining.

5. Inference for Classification: I implemented a function to use the trained network for making predictions. This function takes an image as input and returns the top K most probable flower species along with their probabilities.

6. Sanity Checking: To visually validate the model's predictions, I plotted the probabilities for the top K classes as a bar graph along with the input image.

## Getting Started

If you wish to use this application or explore the code, you can run the provided Jupyter Notebook. Remember to add any necessary imports at the beginning of the code if required.

## Preprocessing the Data

Before training the model, I preprocessed the images. This included transformations like random scaling, cropping, and flipping for the training set, and resizing and center cropping for the validation and testing sets. Additionally, I normalized the images using the mean and standard deviations of the color channels.

## Building and Training the Classifier

I built and trained the classifier using a pre-trained network from `torchvision.models` as the feature extractor. I defined a new feed-forward classifier on top of it and trained it on the training set. Throughout the process, I tracked the loss and accuracy on the validation set to fine-tune hyperparameters and saved the best model.

## Testing the Network

To assess the classifier's performance, I evaluated it on the test set, providing an estimate of its capability to handle new, unseen images.

## Making Predictions

I implemented a function to predict the flower species from an input image. The function takes the image file path and the trained model as inputs, returning the top K most probable flower species along with their probabilities.

## Visualization

To visualize the results, I used `matplotlib`. The output displayed the input image alongside the top K predicted flower species and their probabilities in a bar graph.

## Conclusion

With this project, I have successfully built an application capable of recognizing different species of flowers from images. The knowledge gained from this endeavor can be applied to create applications that classify various objects, animals, or any other subject of interest.

Feel free to explore the application further and leverage your newfound skills to build even more exciting projects! Happy coding! 🌸🌺🌼
