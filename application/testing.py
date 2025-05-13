from fastapi import FastAPI, Query
from fastapi.responses import Response
import joblib
from pydantic import BaseModel, Field 

import pandas as pd

import mlflow
import mlflow.sklearn
import os

# Import the FastAPI library
app = FastAPI()

# Add a health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}


# Use a function to show information about the variables in the dataset
@app.get("/info", tags=["Information"])
def get_info():
    markdown_content = """
# Credit Risk Prediction API - Input Guide

This API predicts whether a user is likely to default on a credit card payment next month.

## Input Variables

- **LIMIT_BAL**: Amount of given credit in NT dollars (includes individual and family/supplementary credit).
- **SEX**: Gender (1 = male, 2 = female).
- **EDUCATION**: Education level (1 = graduate school, 2 = university, 3 = high school, 4 = others).
- **MARRIAGE**: Marital status (1 = married, 2 = single, 3 = others).
- **AGE**: Age in years.

## Payment History (Repayment Status)

- **PAY_0** to **PAY_6**: Repayment status in months September to April (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)').

## Bill & Payment Amounts

- **BILL_AMT1** to **BILL_AMT6**: Bill statement amounts from September to April.
- **PAY_AMT1** to **PAY_AMT6**: Amounts of previous payments made in the same months.

---

Visit the [Swagger UI](/docs) and try out the `/predict` endpoint.
"""
    return Response(content=markdown_content, media_type="text/markdown")

# Define the schema of the input data when we call the api for predictions
class PredictionInput(BaseModel):
    LIMIT_BAL: float  = Field(..., description="Amount of given credit in NT dollars (includes individual and family/supplementary credit")
    SEX        :  int = Field(..., description = 'Gender (1 = male, 2 = female)')
    EDUCATION  :  int  = Field(..., description = '(1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)')
    MARRIAGE   :  int  = Field(..., description = 'Marital status (1=married, 2=single, 3=others)')
    AGE        :  int  = Field(..., description = 'Age in years')
    PAY_0      :  int  = Field(..., description = 'Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)')
    PAY_2      :  int  = Field(..., description = 'Repayment status in August, 2005')
    PAY_3      :  int  = Field(..., description = 'Repayment status in July, 2005')
    PAY_4      :  int  = Field(..., description = 'Repayment status in June, 2005')
    PAY_5      :  int  = Field(..., description = 'Repayment status in May, 2005 ')
    PAY_6      :  int  = Field(..., description = 'Repayment status in April, 2005')
    BILL_AMT1  :  float = Field(..., description = 'Amount of bill statement in September, 2005 (NT dollar)')
    BILL_AMT2  :  float = Field(..., description = 'Amount of bill statement in August, 2005 (NT dollar)')
    BILL_AMT3  :  float = Field(..., description = 'Amount of bill statement in July, 2005 (NT dollar)')
    BILL_AMT4  :  float = Field(..., description = 'Amount of bill statement in June, 2005 (NT dollar)')
    BILL_AMT5  :  float = Field(..., description = 'Amount of bill statement in May, 2005 (NT dollar)')
    BILL_AMT6  :  float = Field(..., description = 'Amount of bill statement in April, 2005 (NT dollar)')
    PAY_AMT1   :  float = Field(..., description = 'Amount of previous payment in September, 2005 (NT dollar)')
    PAY_AMT2   :  float = Field(..., description = 'Amount of previous payment in August, 2005 (NT dollar)')
    PAY_AMT3   :  float = Field(..., description = 'Amount of previous payment in July, 2005 (NT dollar)')
    PAY_AMT4   :  float = Field(..., description = 'Amount of previous payment in June, 2005 (NT dollar)')
    PAY_AMT5   :  float = Field(..., description = 'Amount of previous payment in May, 2005 (NT dollar)')
    PAY_AMT6   :  float = Field(..., description = 'Amount of previous payment in April, 2005 (NT dollar)')

# Function to load the model either from MLflow or joblib
def load_model_from_source(load_mlflow: bool = False):
    if load_mlflow or os.getenv("LOAD_MLFLOW", "false").lower() == "true":
        # We load the model from MLflow that is registered in the model registry with name "random_forest_model"
        
        model_uri = f"mlruns/0/{os.getenv('MLFLOW_RUN_ID')}/artifacts/random_forest_model"
        model = mlflow.sklearn.load_model(model_uri)
    else:
        # We load the model from joblib
        model = joblib.load("models/random_forest_model.joblib")
    return model

@app.post("/predict/", tags=["Prediction"])
async def predict(data: PredictionInput,
                load_mlflow: bool = Query(default = False, description="Load the latest model from MLflow if True, else load from the local files the preselected model. \n\n For a guide on input variables, see the [/info](/info) endpoint.")):
    
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

    explanation = {
        0: " The user is **likely to pay their credit** next month.",
        1: " The user is **likely to default** on their payment next month."
    }
    return {"prediction": int(prediction[0]),
            'explanation': explanation[int(prediction[0])],}

# Run the FastAPI server with the command:
# uvicorn testing:app --reload
'''
When we run the code, at the link of the FastAPI server, we must add /docs to see the documentation of the API.
'''
