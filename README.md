# Financial Risk Project 🚀

A FastAPI-based service for financial risk classification using machine learning models.

##  Features

-  FastAPI REST endpoints for model inference
-  Option to load model from local file or MLflow
-  Trained with Random Forest ( and XGBoost, in process)
-  Containerized with Docker for deployment
-  Render deployed at https://financial-risk-project.onrender.com/docs

## 🧠 Model Loading Logic

You can choose between:
- **Local model** (default)
- **MLflow-logged model**

This is controlled by environment variables:

```bash
LOAD_MLFLOW=true
MLFLOW_RUN_ID=<your_run_id>
