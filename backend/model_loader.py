import os
import mlflow
import mlflow.pyfunc

MODEL_NAME = os.getenv("MODEL_NAME", "awesome-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")

def load_model():
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    token = os.environ["MLFLOW_TRACKING_TOKEN"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    # Alias-based URI (replacement for stages)
    uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    return mlflow.pyfunc.load_model(uri)
