# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 18:57:42 2022

@author: aaboa
"""

import pandas as pd
import numpy as np


from sklearn.pipeline import Pipeline  # TrainingPipeline inherits from Sklearn Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import mlflow

# import signature to validate data input and output for the model
from mlflow.models.signature import infer_signature


import time
import os


#mlflow.get_tracking_uri()
#mlflow.set_tracking_uri("File:///./Final/mlruns")

class TrainingPipeline(Pipeline):

    ''' 
    TrainingPipeline extends from Scikit-Learn Pipeline class.
    Additional functionality to track model metrics and log model artifacts with mlflow
    
    '''
    
    def __init__(self, steps):
        super().__init__(steps)
        
    def fit(self, X_train, y_train):
        self.__pipeline = super().fit(X_train, y_train)
        return self.__pipeline
    
    
    def create_confusion_matrix_plot(self, model, y_true, y_pred, segmentation_labels):
        
        # Plot Confusion Matrix
        confusion_matrix_ = confusion_matrix(y_true, y_pred, labels=segmentation_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_,
                                       display_labels=segmentation_labels)
        #fig, ax = plt.subplots()
        disp.plot()
        confusion_matrix_path = r'./confusion_matric.png'
        plt.savefig(confusion_matrix_path)
        
        return confusion_matrix_path
    

    
    def get_metrics(self, y_true, y_pred, y_pred_prob, segmentation_labels):
        
        acc = accuracy_score(y_true, y_pred)
   
        class_report = classification_report(y_true, y_pred, digits=3)
        with open(r'./classification_report.txt', "w") as f:
            f.write(class_report)
        
        print('Model Accuracy: {}'.format(acc))
        print('=============== Classification Report ==================================')
        print(class_report)
        
        return {'accuracy': round(acc, 2), 'classification_report':class_report}

    def make_model_name(self, experiment_name, run_name):
        clock_time = time.ctime().replace(' ', '-')
        return experiment_name + '_' + run_name + '_' + clock_time
    
    def log_model(self, model_key, X_test, y_test, trained_pipeline_save_path, segmentation_labels, experiment_name, run_name, run_params=None):
        
        model = self.__pipeline.get_params()[model_key]
        
        y_pred = self.__pipeline.predict(X_test)
        
        y_pred_prob = self.__pipeline.predict_proba(X_test)
        
        # y_pred = self.onehot_to_multilabel(y_pred, segmentation_labels)
        # y_test = self.onehot_to_multilabel(y_test, segmentation_labels)
        
   
        run_metrics = self.get_metrics(y_test, y_pred, y_pred_prob, segmentation_labels)
        confusion_matrix_disp = self.create_confusion_matrix_plot(model, y_test, y_pred, segmentation_labels)
        
        
        client = mlflow.tracking.MlflowClient()
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            experiment = client.get_experiment_by_name(experiment_name)
        except:
            experiment = client.get_experiment_by_name(experiment_name)
        
        # Using sqlite:///mlruns.db as the local store for tracking and registry
        #mlflow.set_tracking_uri("sqlite:///mlflow.db") # 
        #mlflow.set_tracking_uri(".\Final\mlruns")
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=run_name):
            
            # Log Parameters
            if not run_params == None:
                for name in run_params:
                    mlflow.log_param(name, run_params[name])
            
            # Log Metrics
            for name in run_metrics:
                if isinstance(run_metrics[name], (int, float)):
                    mlflow.log_metric(name, run_metrics[name])
            
            # Log Artifacts (confusion matrix plot & ROC AUC curve)
            if confusion_matrix_disp:
                mlflow.log_artifact('classification_report.txt', 'classification_report')
                mlflow.log_artifact(confusion_matrix_disp, 'confusion_matrix')
                # removing copies of the artifacts once successfully logged into mlflow
                os.remove('classification_report.txt')
                os.remove(confusion_matrix_disp)
            
            mlflow.log_artifact(trained_pipeline_save_path, 'trained_pipeline')
            # if not roc_auc_plot_path == None:
            #     mlflow.log_artifact(roc_auc_plot_path, "roc_auc_plot")
            
            # Log Model
            model_name = self.make_model_name(experiment_name, run_name)   
            signature = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(sk_model=self.__pipeline, artifact_path='model',signature=signature)
                
        print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))
        

