import os
import sys
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE

from src.logger import logger
from src.exception import BankChurnException


TARGET_COLUMN = "Exited"

DROP_COLUMNS = [
    "RowNumber",
    "CustomerId",
    "Surname",
    "Complain"
]


def create_preprocessor(
    numerical_features: list,
    categorical_features: list
) -> ColumnTransformer:

    try:

        logger.info(
            "Creating preprocessing pipeline."
        )

        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numerical",
                    numerical_pipeline,
                    numerical_features
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features
                )
            ],
            remainder="drop"
        )

        logger.info(
            "Preprocessing pipeline created successfully."
        )

        return preprocessor

    except Exception as error:

        logger.error(
            "Failed to create preprocessing pipeline."
        )

        raise BankChurnException(
            error,
            sys
        )


def transform_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
):

    try:

        logger.info(
            "Starting data transformation."
        )

        df = df.copy()

        logger.info(
            f"Original dataset shape: {df.shape}"
        )

        # ----------------------------------------------------
        # Drop identifiers and leakage
        # ----------------------------------------------------

        columns_to_drop = [
            column
            for column in DROP_COLUMNS
            if column in df.columns
        ]

        df = df.drop(
            columns=columns_to_drop
        )

        logger.info(
            f"Dropped columns: {columns_to_drop}"
        )

        # ----------------------------------------------------
        # Validate target
        # ----------------------------------------------------

        if TARGET_COLUMN not in df.columns:

            raise ValueError(
                f"Target column '{TARGET_COLUMN}' "
                "not found in dataset."
            )

        # ----------------------------------------------------
        # Features and target
        # ----------------------------------------------------

        X = df.drop(
            columns=[TARGET_COLUMN]
        )

        y = df[TARGET_COLUMN]

        # ----------------------------------------------------
        # Train-test split
        # ----------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        logger.info(
            f"X_train shape: {X_train.shape}"
        )

        logger.info(
            f"X_test shape: {X_test.shape}"
        )

        # ----------------------------------------------------
        # Identify feature types
        # ----------------------------------------------------

        numerical_features = X_train.select_dtypes(include=["number"]).columns.tolist()

        categorical_features = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        logger.info(
            f"Numerical features: "
            f"{numerical_features}"
        )

        logger.info(
            f"Categorical features: "
            f"{categorical_features}"
        )

        # ----------------------------------------------------
        # Create preprocessor
        # ----------------------------------------------------

        preprocessor = create_preprocessor(
            numerical_features,
            categorical_features
        )

        # ----------------------------------------------------
        # Fit only on training data
        # ----------------------------------------------------

        X_train_transformed = (
            preprocessor.fit_transform(X_train)
        )

        X_test_transformed = (
            preprocessor.transform(X_test)
        )

        logger.info(
            f"X_train transformed shape: "
            f"{X_train_transformed.shape}"
        )

        logger.info(
            f"X_test transformed shape: "
            f"{X_test_transformed.shape}"
        )

        # ----------------------------------------------------
        # SMOTE for standalone transformed data
        #
        # This is NOT used for CV model tuning.
        # Model tuning will apply SMOTE inside each CV fold.
        # ----------------------------------------------------

        smote = SMOTE(
            random_state=random_state
        )

        X_train_resampled, y_train_resampled = (
            smote.fit_resample(
                X_train_transformed,
                y_train
            )
        )

        logger.info(
            f"X_train after SMOTE: "
            f"{X_train_resampled.shape}"
        )

        logger.info(
            f"y_train after SMOTE: "
            f"{y_train_resampled.shape}"
        )

        # ----------------------------------------------------
        # Save preprocessor
        # ----------------------------------------------------

        preprocessing_directory = os.path.join(
            "artifacts",
            "preprocessing"
        )

        os.makedirs(
            preprocessing_directory,
            exist_ok=True
        )

        preprocessor_path = os.path.join(
            preprocessing_directory,
            "preprocessor.pkl"
        )

        joblib.dump(
            preprocessor,
            preprocessor_path
        )

        logger.info(
            f"Preprocessor saved to: "
            f"{preprocessor_path}"
        )

        # ----------------------------------------------------
        # Return everything required by future stages
        # ----------------------------------------------------

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            X_train_transformed,
            X_test_transformed,
            X_train_resampled,
            y_train_resampled,
            preprocessor
        )

    except Exception as error:

        logger.error(
            "Data transformation failed."
        )

        raise BankChurnException(
            error,
            sys
        )