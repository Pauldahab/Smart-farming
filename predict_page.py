import numpy as np
import pickle
import streamlit as st
import pandas as pd
import joblib
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
encoder=LabelEncoder()
scaler=load('std_scaler.bin')
df = pd.read_csv('Crop_recommendation.csv')
columns_df=['Nitrogen','Phosphorus','Potassium','Temperature','Humidity','pH','Rainfall','Label']
df.columns = columns_df
names = df['Label'].unique()
encoded_names=encoder.fit_transform(df['Label'].unique())
options=['Nitrogen','Phosphorus','Potassium','Temperature','Humidity','pH','Rainfall']
def load_model():
    filename = 'saved_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model
model=load_model()
def show_predict_page():
    st.title("Agriculture Prediction :seedling: ")
    st.write("""### See how each seed is affected by different conditions  """)
    x_axis_val=st.selectbox("Select X-Axis Value",options=options)
    button=st.button("Show Stats")
    if button:
        sns.displot(x=df[f'{x_axis_val}'], y=df['Label'], bins=20, edgecolor="black", color='black', facecolor='#ffb03b')
        plt.title(f"{x_axis_val}", size=20)
        figure=plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(figure)

    st.write("""### Based on the info you'll provide next , we'll give you the best possible seed to grow """)
    N=st.slider("Nitrogen",int(df['Nitrogen'].min()),int(df['Nitrogen'].max()),1)
    P=st.slider("Phosphorus",int(df['Phosphorus'].min()),int(df['Phosphorus'].max()),1)
    K=st.slider("Potassium",int(df['Potassium'].min()),int(df['Potassium'].max()),1)
    Temperature=st.slider("Temperature in Fahrenheit",int(df['Temperature'].min()),int(df['Temperature'].max()),1)
    Humidity=st.slider("Humidity",int(df['Humidity'].min()),int(df['Humidity'].max()),1)
    pH=st.slider("Ph",int(df['pH'].min()),int(df['pH'].max()),1)
    Rainfall=st.slider("Rainfall",int(df['Rainfall'].min()),int(df['Rainfall'].max()),1)
    # initialize list of lists
    data = [[N,P,K,Temperature,Humidity,pH,Rainfall]]
    # Create the pandas DataFrame
    df_example = pd.DataFrame(data, columns=options)
    df_example=scaler.transform(df_example)
    df_example = pd.DataFrame(df_example, columns=options)
    ok = st.button("Make Prediction")
    if ok:
        result=model.predict(df_example)
        index=np.where(encoded_names == result)
        st.subheader(f"In These Conditions You Can Seed {names[index][0]}")