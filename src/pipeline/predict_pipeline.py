import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            label_encoder_path = 'artifacts/label_encoder.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            label_encoder = load_object(file_path=label_encoder_path)

            data_scaled = preprocessor.transform(features)
            y_pred = model.predict(data_scaled)

            # Ensure predictions are in integer format
            y_pred = y_pred.astype(int)

            decoded_labels = label_encoder.inverse_transform(y_pred)
            return decoded_labels
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:

    def __init__(
            self,
            temperature: float,
            humidity: float,
            pm25: float,
            pm10: float,
            no2: float,
            so2: float,
            co: float,  
        ):
        self.temperature = temperature
        self.humidity = humidity
        self.pm25 = pm25
        self.pm10 = pm10
        self.no2 = no2
        self.so2 = so2
        self.co = co

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Temperature": [self.temperature],
                "Humidity": [self.humidity],
                "PM2.5": [self.pm25],
                "PM10": [self.pm10],
                "NO2": [self.no2],
                "SO2": [self.so2],
                "CO": [self.co],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
            