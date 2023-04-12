import streamlit as st
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle

st.title('Medical Diagnostic Web App 👨🏽‍⚕')

# Step 1: Load the model
model = open("rfc_tuned.pickle", 'rb')
clf = pickle.load(model)
model.close()

# Step 2: Get the front end user input

pregs = st.number_input('Pregnancies', 1, 20, step = 1)
glucose = st.slider('Glucose', 40.0, 200.0, 40.0)
bp = st.slider('BloodPressure', 24, 122, 24)
skin = st.slider('SkinThickness', 7, 99, 7)
ins = st.slider('Insulin', 14.0, 846.0, 14.0) 
bmi = st.slider('BMI', 18.2, 67.1, 18.2) 
diab = st.slider('DiabetesPedigreeFunction', 0.078, 2.420, 0.078)
age = st.slider('Age', 21, 81, 21)

# Step 3: Converting user input to model input

data = {'Pregnancies' : pregs,
        'Glucose': glucose, 
        'BloodPressure' : bp,
        'SkinThickness' : skin,
        'Insulin' : ins,
       'BMI' : bmi, 
        'DiabetesPedigreeFunction' : diab,
        'Age' : age}

input_data = pd.DataFrame([data])

# Step 4 : Get the predictions

preds = clf.predict(input_data)[0]
if st.button('Predict'):
    if preds == 1:
        st.error('The person has diabetes')
    if preds == 0:
        st.success('The person is Diabetes Free')
