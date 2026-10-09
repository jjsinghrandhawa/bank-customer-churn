import sys
import pandas as pd
import os
from src.logger import logger
from src.exception import BankChurnException
import warnings
warnings.filterwarnings("ignore")

def load_local_data(file_path: str) -> pd.DataFrame:
    

    try:
        logger.info(f"Starting local data ingestion from: {file_path}")

        df = pd.read_csv(file_path)

        logger.info(
            f"Local data ingestion completed successfully. "
            f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
        )
        raw_data_directory = os.path.join("data","raw")

        os.makedirs(raw_data_directory,exist_ok=True)

        # Destination for the raw dataset
        raw_data_path = os.path.join(
            raw_data_directory,
            "Customer-Churn-Records.csv"
        )

        # Save a copy inside the project
        df.to_csv(
            raw_data_path,
            index=False
        )

        logger.info(
            f"Raw dataset saved successfully to: {raw_data_path}"
        )


        return df

    except Exception as error:
        logger.error(
            f"Failed to load local dataset from: {file_path}"
        )

        raise BankChurnException(error, sys)