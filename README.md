# Smart Farming Seed Classification Model using K-Nearest Neighbors

This is a machine learning model designed to help farmers decide which type of seeds to grow based on the environmental conditions of their farm. The model uses the K-Nearest Neighbors (KNN) algorithm to classify the seeds based on the similarity of their environmental conditions to those in the training dataset.

How to Use:
Go to this link to view the app : 
https://pauldahab-smart-farming-main-gllj04.streamlit.app/

Model Details
The KNN algorithm used in this model is a simple yet effective way to classify data based on the similarity of their feature values to those in the training dataset. The algorithm works by finding the K nearest samples in the training dataset to the input sample and using the majority label among those samples as the predicted label for the input sample.

The KNN algorithm used in this model is implemented using the Scikit-learn library. The algorithm takes a single parameter, K, which represents the number of nearest neighbors to consider when making predictions. The optimal value of K can be determined using cross-validation techniques.

The dataset used to train the model consists of seed types and their corresponding environmental conditions. The model uses the Euclidean distance metric to compute the similarity between samples. The resulting classification model can be used to make predictions on new environmental conditions to determine the most suitable seed type to grow.
