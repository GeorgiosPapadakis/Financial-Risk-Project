import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature 

import pandas as pd
import joblib
import json

# Load the model from joblib
rf_clf = joblib.load("models/random_forest_model.joblib")

# Load the input data for inference
with open('validation_input.txt','r') as f:
    input_data = json.load(f)

# Convert the input data to a pandas DataFrame
df = pd.DataFrame([input_data])

# Infer the signature of the model
signature = infer_signature(df, rf_clf.predict(df))

# Run an experiment with MLflow to track the model
with mlflow.start_run(run_name="RandomForest_Run"):
    # Log model parameters if available (example)
    mlflow.log_param("model_type", "RandomForestClassifier")

    # Log the model
    mlflow.sklearn.log_model(
        rf_clf,
        artifact_path="random_forest_model",
        input_example=df,
        signature=signature)

    # Log a tag that I am the developer of the model
    mlflow.set_tag("developer", "Giorgos_Papadakis")

# Run the code and the run at the terminal "mlflow ui" to see the experiment

# Register the model manually in the MLflow model registry
# or try to run the code below to register the model automatically
#
# model_uri = f'runs:/{mlflow.active_run().info.run_id}/random_forest_model'
# mlflow.register_model(model_uri, "RandomForestModel")