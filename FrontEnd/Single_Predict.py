import streamlit as st
import pandas as pd
import requests
import time

st.markdown("## Single Customer Prediction")
st.sidebar.markdown("## Single Predict")


Gender = st.selectbox('Gender', ['Male', 'Female'])        
Ever_Married = st.selectbox('Ever Married?', ['No', 'Yes'])     
Age = st.text_input('Age')     
Graduated = st.selectbox('Graduated?', ['No', 'Yes']) 
Profession = st.selectbox('Profession', ['Healthcare', 'Engineer', 'Lawyer', 'Entertainment',
                                            'Artist', 'Executive', 'Doctor', 'Homemaker', 'Marketing'])
Work_Experience = st.text_input('Work Experience (Years)')
Spending_Score = st.selectbox('Spending Score', ['Low', 'Average', 'High'])
Family_Size = st.text_input('Family Size')
Var_1 = st.selectbox('Variable', ['Cat_1', 'Cat_2', 'Cat_3','Cat_4', 'Cat_5', 'Cat_6', 'Cat_7'])

if st.button('Predict'):
    deployment_data = [{'Gender':Gender,
                        'Ever_Married':Ever_Married,
                        'Age':Age,
                        'Graduated':Graduated,
                        'Profession':Profession,
                        'Work_Experience':Work_Experience,
                        'Spending_Score':Spending_Score,
                        'Family_Size':Family_Size,
                        'Var_1':Var_1}]
    response = requests.post('http://127.0.0.1:8000/single_predict', json=deployment_data)
    st.success(f"Customer Segment Prediction:  {response.text}")
