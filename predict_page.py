import numpy as np
import pickle
import streamlit as st
import pandas as pd
import joblib
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
scaler=load('std_scaler.bin')
df = pd.read_csv('Crop_recommendation.csv')
names = df['label'].unique()
encoded_names=encoder.fit_transform(df['label'].unique())
def load_model():
    filename = 'saved_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model
model=load_model()
def show_predict_page():
    st.title("Agriculture Prediction :seedling: ")
    st.write("""### Based on the info you'll provide next , we'll give you the best possible seed to grow """)
    N=st.slider("Nitrogen",int(df['N'].min()),int(df['N'].max()),1)
    P=st.slider("Phosphorus",int(df['P'].min()),int(df['P'].max()),1)
    K=st.slider("Potassium",int(df['K'].min()),int(df['K'].max()),1)
    Temperature=st.slider("Temperature in Fahrenheit",int(df['temperature'].min()),int(df['temperature'].max()),1)
    Humidity=st.slider("Humidity",int(df['humidity'].min()),int(df['humidity'].max()),1)
    pH=st.slider("Ph",int(df['ph'].min()),int(df['ph'].max()),1)
    Rainfall=st.slider("Rainfall",int(df['rainfall'].min()),int(df['rainfall'].max()),1)
    # initialize list of lists
    data = [[N,P,K,Temperature,Humidity,pH,Rainfall]]
    # Create the pandas DataFrame
    df_example = pd.DataFrame(data, columns=['Nitrogen','Phosphorus','Potassium','Temperature','Humidity','pH','Rainfall'])
    df_example=scaler.transform(df_example)
    df_example = pd.DataFrame(df_example, columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH','Rainfall'])
    ok = st.button("Make Prediction")
    if ok:
        result=model.predict(df_example)
        index=np.where(encoded_names == result)
        st.subheader(f"In These Conditions You Can Seed {names[index][0]}")