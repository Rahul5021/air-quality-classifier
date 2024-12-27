# Air Quality Classification Web App

This web app is a demonstration project that classifies the air quality of a given location based on various air quality metrics like PM2.5, PM10, NO2, SO2, CO, and others. The app is built using Flask and aims to showcase an end-to-end machine learning pipeline, including data ingestion, transformation, model training, and prediction.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Setup and Instalaltion](#setup-and-installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [License](#license)

## Project Overview

This web application allows users to input air quality data (such as temperature, humidity, PM2.5, PM10, NO2, SO2, and CO) and receive a classification of the air quality. The project involves the following key steps:
1. **Data Ingestion**: Raw air quality data is ingested, cleaned, and split into training and test datasets.
2. **Data Transformation**: The data is preprocessed, including missing value imputation and scaling.
3. **Model Training**: Various machine learning models are trained to predict air quality based on the features.
4. **Prediction**: The trained model predicts air quality based on the user-provided input.

## Technologies Used
- **Flask**: A lightweight Python web framework for building web applications.
- **Scikit-learn**: For machine learning tasks, including data preprocessing, model training, and evaluation.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib and Seaborn**: For data visualization.

## Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/rahul5021/air-quality-classification.git
cd air-quality-classification
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Application

### 1. Run the Flask application
```bash
python app.py
```
The app will be available at `http://127.0.0.1:5000/`.

## Usage

- Open the app in your browser.
- Select the location.
- Click on the predict button to get the air quality classification.

## File Descriptions
- `app.py`: The main application file that runs the Flask server.
- `data_ingestion.py`: Responsible for loading and splitting the dataset into training and test sets.
- `data_transfomation.py`: Performs data preprocessing, including scaling and encoding.
- `model_training.py`: Contains the logic for training different machine learning models and selecting the best one based on accuracy.
- `requirements.txt`: Contains all the Python libraries required to run the project.
- `setup.py`: Script to install the project as a package.
- `logger.py`: Custom logger for logging events and errors.
- `utils.py`: Utility functions for data manipulation and evaluating model.
- `exception.py`: Custom exceptions for handling errors.

## Model Training and Evaluation
The project uses the following models for classification:
- **Random Forest Classifier**
- **Decision Tree**

The models are evaluated based on accuracy, and the best-performing model is selected for deployment. The performance of the model is tested using a holdout test set, and accuracy metrics are logged.

## License
This project is licensed under the MIT License - see the [LICENSE](#license) file for details.


