import os 
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Spliting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                        "Decision Tree": DecisionTreeClassifier(),
                        "Random Forest": RandomForestClassifier(),
                    }
            params = {
                        "Decision Tree": {
                            'criterion': ['gini', 'entropy'],
                            'max_depth': [5, 10, 15],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 5],
                        },
                        "Random Forest": {
                            'n_estimators': [50, 100],
                            'max_depth': [5, 10],
                            'min_samples_split': [2, 5],
                            'min_samples_leaf': [1, 2],
                            'max_features': ['sqrt', 'log2'],
                        },
                    }

            
            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param = params)

            #getting best model score from dict
            best_model_score = max(model_report.values())

            #getting best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("Model accuracy is less than 60%")
            logging.info("Best found model is: %s",best_model_name)

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)

            return accuracy
        except Exception as e:
            raise CustomException(e,sys)