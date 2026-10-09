
import os

import mlflow
from dotenv import load_dotenv

# Load credentials from the local .env file
load_dotenv()

dagshub_username = os.getenv("DAGSHUB_USERNAME")
dagshub_token = os.getenv("DAGSHUB_TOKEN")

if not dagshub_username or not dagshub_token:
    raise ValueError(
        "DAGSHUB_USERNAME or DAGSHUB_TOKEN is missing from .env"
    )

# Configure MLflow authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Configure remote DagsHub tracking
tracking_uri = (
    "https://dagshub.com/"
    "jjsinghrandhawa/bank-customer-churn.mlflow"
)
tracking_uri="https://dagshub.com/jaskaranjit007/bank-customer-churn.mlflow"

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("bank-customer-churn")

print("Tracking URI:", mlflow.get_tracking_uri())

# Log one lightweight test run
with mlflow.start_run(run_name="dagshub-connection-test"):
    mlflow.log_param("connection_test", True)
    mlflow.log_param("project", "bank-customer-churn")
    mlflow.log_metric("test_metric", 1.0)

print("DagsHub MLflow connection test completed.")
