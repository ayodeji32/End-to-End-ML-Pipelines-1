# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:06:13 2022

@author: aaboa


This file shows steps to deploy the trained pipeline on the local machine,
without using any model API end points.

"""

# Import Needed Libraries
import joblib
import pandas as pd


#%%
# Load the Test Dataset
test_data = pd.read_csv(r'Test.csv')

# We will only need the list of features for the deployment task. Pipeline Preprocessing 
# Steps are all saved in the trained_pipeline.pkl file
trained_pipeline_path = r'.\mlruns\1\fb4c96c610f345329fc341b7186e31c2\artifacts\trained_pipeline\Trained_Pipeline.pkl'
loaded_pipeline = joblib.load(trained_pipeline_path)
#loaded_pipeline.get_params()['scaler'].min_  # Check one of the preprocessing steps e.g. scaler

# Get list of Model Features
FEATURES = loaded_pipeline.feature_names_in_

#%%
# Get the required features from the Test Data file
deployment_data = test_data[FEATURES]
# Deal with missing values in test data
deployment_data.dropna(inplace=True)

#%%
# Perform Inference
predicted_segmentations = loaded_pipeline.predict(deployment_data)