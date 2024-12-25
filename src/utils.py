import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_object:
            dill.dump(obj, file_object)


    except Exception as e:
        raise CustomException(e,sys)