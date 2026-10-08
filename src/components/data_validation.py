import sys
import pandas as pd

from src.logger import logger
from src.exception import BankChurnException


EXPECTED_COLUMNS = [
    "RowNumber",
    "CustomerId",
    "Surname",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
    "Complain",
    "Satisfaction Score",
    "Card Type",
    "Point Earned",
]


def validate_columns(df: pd.DataFrame) -> bool:
    """
    Validate that the dataset contains the expected columns.
    """

    try:
        logger.info("Starting column validation.")

        actual_columns = df.columns.tolist()

        missing_columns = [
            column
            for column in EXPECTED_COLUMNS
            if column not in actual_columns
        ]

        unexpected_columns = [
            column
            for column in actual_columns
            if column not in EXPECTED_COLUMNS
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        if unexpected_columns:
            logger.warning(
                f"Unexpected columns found: {unexpected_columns}"
            )

        logger.info("Column validation completed successfully.")

        return True

    except Exception as error:
        logger.error("Column validation failed.")

        raise BankChurnException(error, sys)


def validate_dataset_shape(
    df: pd.DataFrame,
    minimum_rows: int = 100
) -> bool:
    """
    Validate that the dataset contains a reasonable
    number of observations.
    """

    try:
        logger.info("Starting dataset shape validation.")

        rows, columns = df.shape

        if rows < minimum_rows:
            raise ValueError(
                f"Dataset contains only {rows} rows. "
                f"Expected at least {minimum_rows} rows."
            )

        if columns != len(EXPECTED_COLUMNS):
            raise ValueError(
                f"Expected {len(EXPECTED_COLUMNS)} columns, "
                f"but found {columns}."
            )

        logger.info(
            f"Dataset shape validation successful: "
            f"{rows} rows, {columns} columns."
        )

        return True

    except Exception as error:
        logger.error("Dataset shape validation failed.")

        raise BankChurnException(error, sys)


def validate_target_column(df: pd.DataFrame) -> bool:
    """
    Validate the target variable Exited.
    """

    try:
        logger.info("Starting target column validation.")

        if "Exited" not in df.columns:
            raise ValueError(
                "Target column 'Exited' is missing."
            )

        target_values = set(df["Exited"].dropna().unique())

        if not target_values.issubset({0, 1}):
            raise ValueError(
                f"Invalid values found in Exited: {target_values}"
            )

        if df["Exited"].isna().any():
            raise ValueError(
                "Target column 'Exited' contains missing values."
            )

        logger.info(
            "Target column validation completed successfully."
        )

        return True

    except Exception as error:
        logger.error("Target column validation failed.")

        raise BankChurnException(error, sys)