from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## route for homepage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            temperature=request.form.get('temperature'),
            humidity=request.form.get('humidity'),
            pm25=request.form.get('pm25'),
            pm10=request.form.get('pm10'),
            no2=request.form.get('no2'),
            so2=request.form.get('so2'),
            co=request.form.get('co'),
            population_density=request.form.get('population_density')
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)
        return render_template('home.html',results=prediction[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)