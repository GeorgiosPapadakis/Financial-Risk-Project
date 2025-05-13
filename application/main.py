from fastapi import FastAPI, Query
import joblib
from pydantic import BaseModel

import pandas as pd

import mlflow
import mlflow.sklearn


app = FastAPI()

# Define the schema of the input data when we call the api for predictions
class PredictionInput(BaseModel):
    LIMIT_BAL: float
    SEX        :  int  
    EDUCATION  :  int  
    MARRIAGE   :  int  
    AGE        :  int  
    PAY_0      :  int  
    PAY_2      :  int  
    PAY_3      :  int  
    PAY_4      :  int  
    PAY_5      :  int  
    PAY_6      :  int  
    BILL_AMT1  :  float
    BILL_AMT2  :  float
    BILL_AMT3  :  float
    BILL_AMT4  :  float
    BILL_AMT5  :  float
    BILL_AMT6  :  float
    PAY_AMT1   :  float
    PAY_AMT2   :  float
    PAY_AMT3   :  float
    PAY_AMT4   :  float
    PAY_AMT5   :  float
    PAY_AMT6   :  float

# Function to load the model either from MLflow or joblib
def load_model_from_source(load_mlflow: bool = False):
    if load_mlflow:
        # We load the model from MLflow that is registered in the model registry
        model_uri = "models:/random_forest_model/latest"
        model = mlflow.sklearn.load_model(model_uri)
    else:
        # We load the model from joblib
        model = joblib.load("models/random_forest_model.joblib")
    return model

@app.post("/predict/")
async def predict(data: PredictionInput,
                load_mlflow: bool = Query(default = False, description="Load the latest model from MLflow if True, else load from the local files the preselected model.")):
    
    # Load the model from either MLflow or joblib based on the query parameter
    rf_clf = load_model_from_source(load_mlflow=load_mlflow)

    # Extract the features from the incoming request
    # Take the input data as a dictionary
    input_data = data.dict()
    # Convert the input data to a pandas DataFrame
    df = pd.DataFrame([input_data])

    
    # Make a prediction using the Random Forest model
    prediction = rf_clf.predict(df)
    
    # Return the prediction
    return {"prediction": int(prediction[0])}

'''
When we run the code, at the link of the FastAPI server, we must add /docs to see the documentation of the API.
'''
