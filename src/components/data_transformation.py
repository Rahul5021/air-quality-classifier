import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    label_encoder_file_path = os.path.join('artifacts', 'label_encoder.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            num_features = [
                "Temperature",
                "Humidity",
                "PM2.5",
                "PM10",
                "NO2",
                "SO2",
                "CO",
            ]
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")),
                    ('scaler', StandardScaler())
                ]
            )
            logging.info("Numerical columns scaling completed.")
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("read train and test data completed")
            # Replace inf/-inf with NaN
            train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            logging.info("obtaining preprocessing object")

            preprocessor = self.get_data_transformer()

            target_column_name = "Air Quality"

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Encodning target varaible")
            le = LabelEncoder()
            target_feature_train_df = le.fit_transform(target_feature_train_df)
            target_feature_test_df = le.transform(target_feature_test_df)

            logging.info("Applying preprocessing object on train and test dataaframe")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Concatenate features and target to create the final train/test arrays
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saving Preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor
            )
            # Save the LabelEncoder object
            save_object(
                file_path=self.data_transformation_config.label_encoder_file_path,
                obj=le
            )
            return(
                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.label_encoder_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)