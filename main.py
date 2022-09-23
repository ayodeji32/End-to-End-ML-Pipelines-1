# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 08:55:15 2022

@author: aaboa
"""

# Import Needed Libraries
import joblib
import uvicorn
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import List
import json

# FastAPI libray
from fastapi import FastAPI, File, UploadFile

#%%
# Initialize model artifacte files. This will be loaded at the start of FastAPI model server.
# We will only need the list of features for the deployment tasks. Pipeline Preprocessing 
# Steps are all saved in the trained_pipeline.pkl file
trained_pipeline_path = r'.\mlruns\1\fb4c96c610f345329fc341b7186e31c2\artifacts\trained_pipeline\Trained_Pipeline.pkl'
loaded_pipeline = joblib.load(trained_pipeline_path)
#loaded_pipeline.get_params()['scaler'].min_  # Check one of the steps e.g. scaler
# Get list of Model Features
FEATURES = loaded_pipeline.feature_names_in_

#%%

# Initiate app instance
app = FastAPI(title='Customer Segmentation', version='1.0',
              description='OneVsRest Model is Used for Predictions')

#%%
# Classes for Data Type validation to ensure that the data passed to the model by the user is valid

class DictValidator(BaseModel):
    '''
    This helper Class is used to validate the data type of values
    in a dictionary (a row) passed to it. It inherits the BAseModel from Pydantic
    which uses the Python Type declaration to validate data.
    '''
    Gender : str
    Ever_Married: str
    Age: int
    Graduated: str
    Profession : str
    Work_Experience : int
    Spending_Score : str
    Family_Size : int
    Var_1 : str


class DataFrameValidator(BaseModel):
    '''
    This helper class is used to validate the data types in the 
    Test dataframe. It inherits the BAseModel from Pydantic and uses a similar 
    helper class to validate the dataframe row by row by passing 
    the dataframe as a list of dictionaries. Each row is a dictionary.
    '''
    df_dict: List[DictValidator]

#%%
# Creating the End Points

# API for the Home Endpoint
@app.get('/home')
def read_home():
    """
     Home endpoint which can be used to test the availability of the application.
     """
    return {'message': 'System is healthy'}


# API for the "Upload CSV" endpoint
@app.post("/uploadcsv/")
def upload_csv(csv_file: UploadFile = File(...)):
    '''
    Parameters
    ----------
    csv_file : UploadFile, optional
        DESCRIPTION. The default is File(...).

    Returns
    -------
    parsed : json object
        DESCRIPTION.
        
    This end point uploads the Test data in csv provided by user
    and stores the file locally to be used by the batch_predict endpoint.
    The purpose of this upload csv endpoint is to display to the user
    the uploaded data before prediction is made.
    '''
    test_data_df = pd.read_csv(csv_file.file)
    # save imported data to csv to be used by the predict endpoint
    test_data_df.to_csv('deployment_data.csv')
    # convert dataframe to json to return to client
    res = test_data_df.to_json(orient="records")
    parsed = json.loads(res)
    # Return response back to client
    return parsed


# API endpoint for making batch prediction against the request received from client.
@app.post("/batch_predict")
def batch_predict():
    '''
    Parameters
    -----------
    No parameter is required.

    Returns
    -------
    parsed_prediction : TYPE
        DESCRIPTION.
    The predict endpoint reads the csv file stored by the uploadcsv endpoint,
    and makes predictions on the data. It returns a json object to the client.
    
    '''
    deployment_data = pd.read_csv(r'Data/uploadedfile.csv',index_col='ID') # Keeping track of the Customer ID's

    # Validate data types in dataframe
    val = deployment_data.to_dict(orient="records")
    DataFrameValidator(df_dict = val)

    #deployment_data = pd.DataFrame.from_dict(deployment_data,orient='index')
    deployment_data = deployment_data[FEATURES]
    # Deal with missing values in test data
    deployment_data.dropna(inplace=True)
    
    # Create prediction
    predicted_segmentations = loaded_pipeline.predict(deployment_data)
    deployment_data['Segmentation Prediction'] = predicted_segmentations
    deployment_data.reset_index(drop=False,inplace=True)
    deployment_data.to_csv(r'Data/predictions.csv')

    res = deployment_data.to_json(orient="records")
    parsed_prediction = json.loads(res)
    # Return response back to client
    return parsed_prediction


# API endpoint for making batch prediction against the request received from client.
@app.post("/single_predict")
def single_predict(data:DictValidator):
    #data_dict = data.dict()
    data_df = pd.DataFrame.from_dict(data,orient='index').T
    data_df= data_df[FEATURES]
    predicted_segmentation = loaded_pipeline.predict(data_df)
    return {'Customer Segment' : predicted_segmentation}



if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) 