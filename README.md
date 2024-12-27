# Air Quality Classification Web App

This web app is a demonstration project that classifies the air quality of a given location based on various air quality metrics like PM2.5, PM10, NO2, SO2, CO, and others. The app is built using Flask and aims to showcase an end-to-end machine learning pipeline, including data ingestion, transformation, model training, and prediction.

## Table of Contents
- Project Overview(#Project Overview)
- Technologies Used
- Project Structure
- Setup and Instalaltion
- Usage
- File Descriptions
- Model Training and Evaluation
- Liecense

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


