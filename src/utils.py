import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_object:
            dill.dump(obj, file_object)


    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates multiple models using GridSearchCV for hyperparameter tuning
    and returns a dictionary of model names and their test scores.

    Args:
        X_train (array-like): Training feature data.
        y_train (array-like): Training target data.
        X_test (array-like): Test feature data.
        y_test (array-like): Test target data.
        models (dict): Dictionary of model names and their instances.
        param (dict): Dictionary of model names and their hyperparameter grids.

    Returns:
        dict: Model names and their corresponding test accuracy scores.
    """
    try:
        report = {}

        for model_name, model in models.items():
            try:
                logging.info(f"Starting GridSearchCV for {model_name}")
                
                # Fetch the hyperparameter grid for the current model
                param_grid = param.get(model_name, {})

                # Perform GridSearchCV
                gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
                gs.fit(X_train, y_train)

                # Get the best parameters and train the model
                best_params = gs.best_params_
                logging.info(f"Best parameters for {model_name}: {best_params}")
                model.set_params(**best_params)
                model.fit(X_train, y_train)

                # Make predictions for train and test sets
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Evaluate the model
                train_model_score = accuracy_score(y_train, y_train_pred)
                test_model_score = accuracy_score(y_test, y_test_pred)

                # Log results and store the test score
                logging.info(f"{model_name} - Train Accuracy: {train_model_score:.4f}, Test Accuracy: {test_model_score:.4f}")
                report[model_name] = test_model_score

                # Optional: Log classification report for better insights
                logging.info(f"Classification Report for {model_name}:\n{classification_report(y_test, y_test_pred)}")

            except Exception as model_error:
                report[model_name] = None  # Assign None if the model fails

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_object:
            return dill.load(file_object)
        
    except Exception as e:
        raise CustomException(e,sys)