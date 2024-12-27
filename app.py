from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__,static_folder='static')
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Parse JSON data from request
        data = request.get_json()
        temperature = data.get('temperature')
        humidity = data.get('humidity')
        pm25 = data.get('pm25')
        pm10 = data.get('pm10')
        no2 = data.get('no2')
        so2 = data.get('so2')
        co = data.get('co')

        # Create dataframe for prediction
        data = CustomData(
            temperature=temperature,
            humidity=humidity,
            pm25=pm25,
            pm10=pm10,
            no2=no2,
            so2=so2,
            co=co,
        )
        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)
        print(prediction)

        # Return prediction result
        return jsonify({'result': prediction[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
