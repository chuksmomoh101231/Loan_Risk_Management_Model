#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML


# In[5]:


h2o.init()


# In[6]:


automl = h2o.upload_mojo('credit_risk_check_model_MOJO')


def predict(input_df):
    predictions_df = automl.predict(input_df)
    predictions = predictions_df['Label'][0]
    return predictions



def run():

    from PIL import Image
    #image = Image.open('logo_pycaret.png')
    #image_hospital = Image.open('hospital.jpg')

    #st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict loan defaulters')
    #st.sidebar.success('https://www.h2o')
    
    #st.sidebar.image(image_hospital)

    st.title("Credit Risk Classification App")

    if add_selectbox == 'Online':
        person_age = st.number_input('Age', min_value=0, max_value=100, value=0)
        person_income = st.number_input('Income', min_value=0, max_value=7000000, value=0)
        person_home_ownership = st.selectbox('House Ownership', ['RENT', 'MORTGAGE','OWN','OTHER'])
        person_emp_length  = st.number_input('Length of Employment', min_value=0, max_value=1000, value=0)
        loan_intent = st.selectbox('Loan Intention', ['EDUCATION', 'MEDICAL','VENTURE','PERSONAL','DEBTCONSOLIDATION','HOMEIMPROVEMENT'])
        loan_grade = st.selectbox('Loan Grade', ['A','B','C','D','E','F','G'])
        loan_amnt = st.number_input('Loan Amount', min_value=0, max_value=7000000, value=0)
        loan_int_rate = st.number_input('Interest Rate', min_value=0.0, max_value=1000.0, value=0.0)
        loan_percent_income = st.number_input('Loan Income Percentage', min_value=0.0, max_value=1000.0, value=0.0)
        cb_person_default_on_file = st.selectbox('Default Previously?', ['Y','N'])
        cb_person_cred_hist_length = st.number_input('Length of Credit History', min_value=0, max_value=1000, value=0)
        
        
        output=""
    
        input_dict = {'person_age' : person_age, 'person_income' : person_income, 'person_home_ownership' : person_home_ownership, 'person_emp_length' : person_emp_length, 
                'loan_intent' : loan_intent, 'loan_grade' : loan_grade,'loan_amnt':loan_amnt,'loan_int_rate':loan_int_rate,
                'loan_percent_income':loan_percent_income,'cb_person_default_on_file':cb_person_default_on_file,
                'cb_person_cred_hist_length':cb_person_cred_hist_length}
    
        input_df = pd.DataFrame([input_dict])
    
        input_df = h2o.H2OFrame(input_df)
        
        if st.button("Predict"):
            output = automl.predict(input_df).as_data_frame()
            st.write(output)
            
            
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            data = h2o.H2OFrame(data)
            predictions = automl.predict(data).as_data_frame()
            predictions = predictions.join(data.as_data_frame())
            st.write(predictions)
            
            @st.cache
            
            def convert_df(df):
                return df.to_csv(index = False, header=True).encode('utf-8')
            csv = convert_df(predictions)
            st.download_button(label="Download data as CSV",data=csv,
                file_name='credit_risk_prediction.csv',mime='text/csv')
            
            
    
if __name__ == '__main__':
    run()
    
    

