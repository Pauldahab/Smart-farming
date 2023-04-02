# Smart Farming Seed Classification Model using K-Nearest Neighbors

Model Overview

This project aims to build a KNN (K-Nearest Neighbors) model to predict the type of seed that should be grown based on environmental factors such as temperature, humidity , etc. The dataset used for this project is the "Crop_recommendation" which contains information about 22 different seed types.

Dataset:

The crop recommendation Dataset used for this project contains 2200 examples with 7 attributes:

Nitrogen

Phosphorus

Potassium

Temperature

Humidity

pH

Rainfall


The dataset is preprocessed and normalized before being used for training the KNN model.

Steps:

1. Load and preprocess the dataset
The Crop Recommendation Dataset is loaded using Pandas and preprocessed by normalizing the data to a standard scale. This ensures that each attribute has the same impact on the KNN model.

2. Split the dataset into training and testing sets
The preprocessed dataset is split into a training set and a testing set. The training set is used to train the KNN model, while the testing set is used to evaluate the accuracy of the model.

3. Train the KNN model with the training set
The KNN model is trained using the training set. The number of neighbors to be considered is set to a value that provides the best accuracy for the model (in our case k=1).

4. Evaluate the accuracy of the KNN model with the testing set
The accuracy of the KNN model is evaluated by comparing the predicted seed types with the actual seed types from the testing set. The accuracy of the model is calculated using the accuracy_score function from the Scikit-Learn library.

5. Use the KNN model to predict the seed type based on new environmental factors
The trained KNN model can be used to predict the seed type based on new environmental factors such as temperature, humidity, etc.
