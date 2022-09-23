# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:47:53 2022

@author: aaboa
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# from feature-engine for feature engineering
from feature_engine.imputation import (
    MeanMedianImputer,
    CategoricalImputer,
)

from feature_engine.encoding import (
    CountFrequencyEncoder,
    OneHotEncoder)

    
#from feature_engine.transformation import LogTransformer

# importing a custom mapper for ordinal encoded variable
import custom_transformers as transformer


#%%


def pipe_configuration():

    # categorical variables with NA in train set
    # variables to impute with the most frequent category
    CAT_NA_VARS = ['Ever_Married',
                    'Graduated',
                    'Profession',
                    'Var_1',
                  ]
    
    # numerical variables with NA in train set
    NUM_NA_VARS = ['Work_Experience',
                    'Family_Size',
                  ]
    
    # categorical variables for ordinal encoding
    ORDINAL_ENCOD_CAT_VARS = ['Spending_Score']
    ORDINAL_CAT_VARS_MAPPING = {'Low': 1, 'Average': 2, 'High': 3}
    
    # variables for binary encoding
    BINARY_ENCOD_CAT_VARS = ['Gender','Ever_Married', 'Graduated']
    
    # variables for count frequency encoding
    FREQUENCY_ENCOD_CAT_VARS = ['Profession','Var_1']
    
    FEATURES = ['Gender', 
                'Ever_Married', 
                'Age', 
                'Graduated', 
                'Profession',
                'Work_Experience', 
                'Spending_Score', 
                'Family_Size', 
                'Var_1']
    
    
    # set up the pipeline
    PIPESTEPS = [
                # ===== IMPUTATION =====
                ('mean_imputation', MeanMedianImputer(
                    imputation_method='mean', variables=NUM_NA_VARS)),
                ('frequent_imputation', CategoricalImputer(
                    imputation_method='frequent', variables=CAT_NA_VARS)),
                # ====== ENCODING ======
                ('binary_encoding', OneHotEncoder(
                    drop_last_binary=True, variables=BINARY_ENCOD_CAT_VARS)),
                ('countfreq', CountFrequencyEncoder(
                    encoding_method='frequency', variables=FREQUENCY_ENCOD_CAT_VARS)),
                # === mappers ===
                ('mapper_spending_score', transformer.Mapper(
                    variables=ORDINAL_ENCOD_CAT_VARS, mappings=ORDINAL_CAT_VARS_MAPPING)),
                ('scaler', MinMaxScaler()),
                ]
    
    return FEATURES, PIPESTEPS

