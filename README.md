# MLOps Zoomcamp 2025 - Module 1 Assignment

This repository contains the solutions for Module 1 of the MLOps Zoomcamp (https://github.com/DataTalksClub/mlops-zoomcamp/tree/main) by DataTalksClub.  
The goal of this module is to build a baseline machine learning pipeline for predicting New York Yellow Taxi trip durations using linear regression (Regression task).


## Dataset

We used the **NYC Yellow Taxi Trip Records** for:

- **January 2023** (training)
- **February 2023** (validation)

The dataset includes fields like pickup/dropoff timestamps, locations, passenger count, trip distance, etc.  
You can find the raw data [on the NYC Taxi & Limousine Commission website](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).


## Objective

Train a regression model that predicts the duration (in minutes) of a taxi trip using only:

- `PULocationID` (pickup location)
- `DOLocationID` (dropoff location)


## Steps Performed

1. **Data Preprocessing**
   - Compute trip duration in minutes
   - Filter out extreme durations (<1 min or >60 mins)
2. **Feature Engineering**
   - Cast location IDs to strings
   - Use `DictVectorizer` for one-hot encoding
3. **Model Training**
   - Trained `LinearRegression` on the January data
4. **Model Evaluation**
   - Computed RMSE on training and validation data


## Results

| Dataset    | RMSE  |
|------------|-------|
| Train (Jan)| 7.65  |
| Val (Feb)  | 7.81  |


### Requirements

Install dependencies (e.g., using pip):

pip install -r requirements.txt