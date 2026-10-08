import os
import sys
import json
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier

from src.logger import logger
from src.exception import BankChurnException


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

MODEL_DIRECTORY = os.path.join(
    "artifacts",
    "models"
)

os.makedirs(
    MODEL_DIRECTORY,
    exist_ok=True
)

MLFLOW_EXPERIMENT_NAME = (
    "bank-customer-churn"
)


# ============================================================
# MLflow Configuration
# ============================================================

def setup_mlflow():

    try:

        mlflow.set_experiment(
            MLFLOW_EXPERIMENT_NAME
        )

        logger.info(
            f"MLflow experiment configured: "
            f"{MLFLOW_EXPERIMENT_NAME}"
        )

    except Exception as error:

        logger.error(
            "Failed to configure MLflow."
        )

        raise BankChurnException(
            error,
            sys
        )


# ============================================================
# Create Model Pipelines
# ============================================================

def create_model_pipelines(
    preprocessor
):

    try:

        logger.info(
            "Creating model pipelines."
        )

        models = {}

        # ====================================================
        # Logistic Regression
        # ====================================================

        models[
            "Logistic Regression"
        ] = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "smote",
                    SMOTE(
                        random_state=RANDOM_STATE
                    )
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE
                    )
                )
            ]
        )

        # ====================================================
        # Random Forest
        # ====================================================

        models[
            "Random Forest"
        ] = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "smote",
                    SMOTE(
                        random_state=RANDOM_STATE
                    )
                ),
                (
                    "model",
                    RandomForestClassifier(
                        random_state=RANDOM_STATE,
                        n_jobs=-1
                    )
                )
            ]
        )

        # ====================================================
        # XGBoost
        # ====================================================

        models[
            "XGBoost"
        ] = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "smote",
                    SMOTE(
                        random_state=RANDOM_STATE
                    )
                ),
                (
                    "model",
                    XGBClassifier(
                        random_state=RANDOM_STATE,
                        eval_metric="logloss",
                        n_jobs=-1
                    )
                )
            ]
        )

        # ====================================================
        # HistGradientBoosting
        # ====================================================

        models[
            "HistGradientBoosting"
        ] = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "smote",
                    SMOTE(
                        random_state=RANDOM_STATE
                    )
                ),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        random_state=RANDOM_STATE
                    )
                )
            ]
        )

        logger.info(
            "All model pipelines created successfully."
        )

        return models

    except Exception as error:

        logger.error(
            "Failed to create model pipelines."
        )

        raise BankChurnException(
            error,
            sys
        )


# ============================================================
# Hyperparameter Grids
# ============================================================

def create_parameter_grids():

    try:

        logger.info(
            "Creating hyperparameter grids."
        )

        parameter_grids = {

            "Logistic Regression": {

                "model__C": [
                    0.01,
                    0.1,
                    1.0,
                    10.0
                ],

                "model__solver": [
                    "liblinear",
                    "lbfgs"
                ]
            },

            "Random Forest": {

                "model__n_estimators": [
                    100,
                    200
                ],

                "model__max_depth": [
                    None,
                    10,
                    20
                ],

                "model__min_samples_split": [
                    2,
                    5
                ],

                "model__min_samples_leaf": [
                    1,
                    2
                ],

                "model__max_features": [
                    "sqrt",
                    "log2"
                ]
            },

            "XGBoost": {

                "model__n_estimators": [
                    100,
                    200
                ],

                "model__max_depth": [
                    3,
                    5
                ],

                "model__learning_rate": [
                    0.05,
                    0.1
                ],

                "model__subsample": [
                    0.8,
                    1.0
                ],

                "model__colsample_bytree": [
                    0.8,
                    1.0
                ]
            },

            "HistGradientBoosting": {

                "model__max_iter": [
                    100,
                    200
                ],

                "model__learning_rate": [
                    0.05,
                    0.1
                ],

                "model__max_leaf_nodes": [
                    15,
                    31
                ],

                "model__l2_regularization": [
                    0.0,
                    1.0
                ]
            }
        }

        logger.info(
            "Hyperparameter grids created successfully."
        )

        return parameter_grids

    except Exception as error:

        logger.error(
            "Failed to create hyperparameter grids."
        )

        raise BankChurnException(
            error,
            sys
        )


# ============================================================
# Evaluate Model
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    try:

        y_pred = model.predict(
            X_test
        )

        y_probability = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        metrics = {

            "accuracy": accuracy_score(
                y_test,
                y_pred
            ),

            "precision": precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "recall": recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "f1": f1_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "roc_auc": roc_auc_score(
                y_test,
                y_probability
            ),

            "pr_auc": average_precision_score(
                y_test,
                y_probability
            )
        }

        return metrics

    except Exception as error:

        logger.error(
            "Model evaluation failed."
        )

        raise BankChurnException(
            error,
            sys
        )


# ============================================================
# Train and Tune Models
# ============================================================

def train_models(
    models,
    parameter_grids,
    X_train,
    y_train,
    X_test,
    y_test
):

    try:

        logger.info(
            "Starting model training and "
            "hyperparameter tuning."
        )

        setup_mlflow()

        results = {}

        best_models = {}

        # ====================================================
        # Train Each Model
        # ====================================================

        for model_name, pipeline in models.items():

            print(
                "\n" + "=" * 70
            )

            print(
                f"TRAINING MODEL: {model_name}"
            )

            print(
                "=" * 70
            )

            logger.info(
                f"Starting training: {model_name}"
            )

            # ------------------------------------------------
            # Start MLflow run
            # ------------------------------------------------

            with mlflow.start_run(
                run_name=model_name
            ):

                # ------------------------------------------------
                # Log model name
                # ------------------------------------------------

                mlflow.log_param(
                    "model_name",
                    model_name
                )

                mlflow.log_param(
                    "random_state",
                    RANDOM_STATE
                )

                mlflow.log_param(
                    "cv_folds",
                    5
                )

                mlflow.log_param(
                    "scoring",
                    "roc_auc"
                )

                mlflow.log_param(
                    "smote",
                    True
                )

                # ------------------------------------------------
                # Grid Search
                # ------------------------------------------------

                grid_search = GridSearchCV(

                    estimator=pipeline,

                    param_grid=parameter_grids[
                        model_name
                    ],

                    scoring="roc_auc",

                    cv=5,

                    n_jobs=-1,

                    verbose=1,

                    refit=True
                )

                # ------------------------------------------------
                # Train
                # ------------------------------------------------

                grid_search.fit(
                    X_train,
                    y_train
                )

                # ------------------------------------------------
                # Best model
                # ------------------------------------------------

                best_model = (
                    grid_search.best_estimator_
                )

                # ------------------------------------------------
                # Best parameters
                # ------------------------------------------------

                best_params = (
                    grid_search.best_params_
                )

                # ------------------------------------------------
                # CV score
                # ------------------------------------------------

                cv_roc_auc = (
                    grid_search.best_score_
                )

                # ------------------------------------------------
                # Evaluate on test set
                # ------------------------------------------------

                metrics = evaluate_model(
                    best_model,
                    X_test,
                    y_test
                )

                # ------------------------------------------------
                # Log best parameters
                # ------------------------------------------------

                for parameter, value in (
                    best_params.items()
                ):

                    mlflow.log_param(
                        parameter,
                        value
                    )

                # ------------------------------------------------
                # Log CV metric
                # ------------------------------------------------

                mlflow.log_metric(
                    "cv_roc_auc",
                    cv_roc_auc
                )

                # ------------------------------------------------
                # Log test metrics
                # ------------------------------------------------

                for metric_name, value in (
                    metrics.items()
                ):

                    mlflow.log_metric(
                        f"test_{metric_name}",
                        value
                    )

                # ------------------------------------------------
                # Log model
                # ------------------------------------------------

                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    name="model",
                    serialization_format="cloudpickle")

                # ------------------------------------------------
                # Store results
                # ------------------------------------------------

                results[
                    model_name
                ] = {

                    "best_params":
                        best_params,

                    "cv_roc_auc":
                        cv_roc_auc,

                    "test_metrics":
                        metrics
                }

                best_models[
                    model_name
                ] = best_model

                # ------------------------------------------------
                # Print results
                # ------------------------------------------------

                print(
                    "\nBest Parameters:"
                )

                print(
                    best_params
                )

                print(
                    "\nCross-Validation ROC-AUC:"
                )

                print(
                    f"{cv_roc_auc:.4f}"
                )

                print(
                    "\nTest Metrics:"
                )

                for metric_name, value in (
                    metrics.items()
                ):

                    print(
                        f"{metric_name}: "
                        f"{value:.4f}"
                    )

                logger.info(
                    f"Completed training: "
                    f"{model_name}"
                )

        return (
            results,
            best_models
        )

    except Exception as error:

        logger.error(
            "Model training failed."
        )

        raise BankChurnException(
            error,
            sys
        )


# ============================================================
# Select Best Model
# ============================================================

def select_best_model(
    results,
    best_models
):

    try:

        logger.info(
            "Selecting best model using "
            "CV ROC-AUC."
        )

        best_model_name = max(
            results,

            key=lambda model_name:
            results[
                model_name
            ][
                "cv_roc_auc"
            ]
        )

        best_model = (
            best_models[
                best_model_name
            ]
        )

        logger.info(
            f"Best model selected: "
            f"{best_model_name}"
        )

        return (
            best_model_name,
            best_model
        )

    except Exception as error:

        logger.error(
            "Best model selection failed."
        )

        raise BankChurnException(
            error,
            sys
        )


# ============================================================
# Save Best Model
# ============================================================

def save_best_model(
    model,
    model_name
):

    try:

        logger.info(
            "Saving best model."
        )

        model_path = os.path.join(
            MODEL_DIRECTORY,
            "best_model.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        metadata_path = os.path.join(
            MODEL_DIRECTORY,
            "best_model_metadata.json"
        )

        metadata = {

            "model_name":
                model_name,

            "model_path":
                model_path
        }

        with open(
            metadata_path,
            "w"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4
            )

        logger.info(
            f"Best model saved: "
            f"{model_path}"
        )

        return model_path

    except Exception as error:

        logger.error(
            "Failed to save best model."
        )

        raise BankChurnException(
            error,
            sys
        )