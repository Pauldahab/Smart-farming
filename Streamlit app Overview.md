# Streamlit App Deployment for KNN Machine Learning Model


Link for the app:

https://pauldahab-smart-farming-main-gllj04.streamlit.app/


Overview

This project aims to deploy a KNN (K-Nearest Neighbors) machine learning model using Streamlit. The deployed app will allow users to interact with the model by inputting values related to temperature, humidity, and other parameters to receive predictions on which type of seed they should grow.

Files:

This project consists of the following files:

crop_recommendation.csv: the seed dataset used to train the KNN machine learning model

saved_model.sav: the trained KNN machine learning model saved as a pickle file

main.py: the Streamlit app that deploys the KNN machine learning model


Steps

1.The trained KNN machine learning model is saved as a pickle file using the joblib library from Scikit-Learn. This allows the model to be loaded and used later for making predictions.

2. Create a Streamlit app and load the trained model
The Streamlit app is created in main.py using the Streamlit library. The trained machine learning model is loaded from the saved pickle file using joblib.

3. Create a user interface and input form for users to input values
The user interface of the app is created using Streamlit. An input form is created for the user to input values related to temperature, humidity, and other parameters.

4. Use the trained model to make predictions based on user input
When the user inputs values, the trained KNN machine learning model is used to make predictions based on the input values.

5. Display the predicted seed type to the user
The predicted seed type is displayed to the user using Streamlit.

Conclusion

In conclusion, this project demonstrates how to deploy a KNN machine learning model using Streamlit. The deployed app allows users to interact with the model by inputting values related to temperature, humidity, and other parameters to receive predictions on which type of seed they should grow. This type of deployment can be useful for providing easy access to machine learning models to users who may not have experience with coding or machine learning.

