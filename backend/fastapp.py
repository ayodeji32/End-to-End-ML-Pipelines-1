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
from typing import List, Optional
import json
import os.path
import mlflow
#from subprocess import call

# FastAPI libray
from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from pydantic import parse_obj_as

#from model_config.training_pipeline import TrainingPipeline

#%%
# Initialize model artifacte files. This will be loaded at the start of FastAPI model server.
# We will only need the list of features for the deployment tasks. Pipeline Preprocessing 
# Steps are all saved in the trained_pipeline.pkl file
current_file_path = os.path.normpath(__file__).rsplit(os.sep, maxsplit=1)[0]
trained_pipeline_path = f"{current_file_path}/model_config/model/Trained_Pipeline.pkl"
loaded_pipeline = joblib.load(trained_pipeline_path)

# trained_pipeline_path = f'{current_file_path}/model_config/model/model'
# loaded_pipeline = mlflow.pyfunc.load_model(trained_pipeline_path)
#loaded_pipeline.get_params()['scaler'].min_  # Check one of the steps e.g. scaler
# Get list of Model Features
FEATURES = loaded_pipeline.feature_names_in_
custumer_id_col = 'ID'

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
    ID : Optional[int]=None
    Gender : str
    Ever_Married: str
    Age: int
    Graduated: str
    Profession : str
    Work_Experience : int
    Spending_Score : str
    Family_Size : int
    Var_1 : str



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
def batch_predict(deployment_data: List[DictValidator]):
    '''
    Parameters
    -----------
    No parameter is required.

    Returns
    -------
    parsed_prediction : TYPE
        DESCRIPTION.
    The predict endpoint received the uploaded data as a json
    object, makes predictions and returns a json object to the client.
    
    '''
    
    # Convert the pydantic model to python dictionary
    deployment_data = pd.DataFrame.from_records(deployment_data)
    cols = {}
    for col in deployment_data:
        col_name = deployment_data.loc[0,col][0]
        cols[col_name] = [i[1] for i in deployment_data.loc[:,col].values]

    deployment_data = pd.DataFrame(cols)

    print(deployment_data)
    customer_ids = deployment_data[custumer_id_col]
    deployment_data = deployment_data[FEATURES]

    # Create prediction
    predicted_segmentations = loaded_pipeline.predict(deployment_data)
    deployment_data['Segmentation Prediction'] = predicted_segmentations
    deployment_data[custumer_id_col] = customer_ids
    output_cols_ordered = [custumer_id_col]
    output_cols_ordered.extend(deployment_data.columns[:-1])
    deployment_data = deployment_data[output_cols_ordered]
    deployment_data.reset_index(drop=True,inplace=True)
    #deployment_data.to_csv(r'Data/predictions_1.csv')

    res = deployment_data.to_json(orient="columns")
    parsed_prediction = json.loads(res)
    # Return response back to client
    return parsed_prediction



# API endpoint for making batch prediction against the request received from client.
@app.post("/single_predict")
def single_predict(data:DictValidator):
    #data_dict = data.dict()
    data = jsonable_encoder(data)
    data_df = pd.DataFrame.from_dict(data,orient='index').T
    data_df= data_df[FEATURES]
    predicted_segmentation = loaded_pipeline.predict(data_df)
    return {predicted_segmentation[0]}



if __name__ == '__main__':
    uvicorn.run("fastapp:app", host="0.0.0.0", port=8000, reload=True) 
