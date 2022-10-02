# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 00:47:08 2022

@author: aaboa
"""

# from Keras to build the model
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



class RandomForestClassifierModel():
    
    def __init__(self, segmentation_labels,FEATURES,random_state):
        self.segmentation_labels = segmentation_labels
        self.FEATURES = FEATURES
        self.random_state = random_state
    
    # define baseline model
    def keras_model(self):
        # create model
        model = Sequential()
        model.add(Dense(hidden_neurons, input_dim=len(self.FEATURES), activation=hidden_activation))
        model.add(Dense(output_neurons, activation=output_activation))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
                      metrics=['accuracy'])
        return model
    
    #if hyper_parameter_tuning:
    def hyper_parameter_tuning(self):
        # Parameters to search
        hidden_neurons = 500
        epochs = [10, 25]
        batch_size = [500, 1000]
        
        # Parameters to leave constant
        hidden_activation = 'relu'
        output_activation = 'softmax'
        #optimizer = 'adam'
        output_neurons = len(self.segmentation_labels)
        
        param_grid = {
                      'epochs': epochs,
                      'batch_size': batch_size} #'units':hidden_neurons,
        
        classifier = KerasClassifier(build_fn = self.keras_model(), 
                                     epochs=epochs, 
                                     batch_size=batch_size, 
                                     verbose=1)
        
        self.estimator_model = RandomizedSearchCV(estimator = classifier, 
                                             param_distributions = param_grid,
                                               cv = 10, verbose=2, 
                                             random_state=self.random_state, 
                                             n_jobs = -1)
        
        # Make a list of Parameters to log in Mlflow
        self.run_parameters = {}
        
        return self.estimator_model, self.run_parameters
        
    
    
    def build_base_model(self):
        
        self.hidden_neurons = 1000
        hidden_activation = 'relu'
        output_neurons = len(self.segmentation_labels)
        output_activation = 'softmax'
        epochs = 10
        batch_size = 8000
        learning_rate = 0.0001
        
        
        def keras_model():
            # create model
            model = Sequential()
            model.add(Dense(self.hidden_neurons, input_dim=len(self.FEATURES), activation=hidden_activation))
            model.add(Dense(output_neurons, activation=output_activation))
            # Compile model
            model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
                          metrics=['accuracy'])
            return model

        estimator_model = KerasClassifier(build_fn= keras_model(), 
                                          epochs=epochs, 
                                          batch_size=batch_size, 
                                          verbose=1)
        
        # Make a list of Parameters to log in Mlflow
        run_parameters = {'hidden_neurons': self.hidden_neurons,
                          'hidden_activation': hidden_activation,
                          'output_neurons': output_neurons,
                          'output_activation': output_activation,
                          'epochs': epochs,
                          'batch_size': batch_size,
                         'learning_rate': learning_rate}
        
        
        return estimator_model, run_parameters
    
    