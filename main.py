import sys
import warnings
warnings.filterwarnings("ignore")

from src.logger import logger
from src.exception import BankChurnException

from src.components.data_ingestion import (
    load_local_data
)

from src.components.data_validation import (
    validate_columns,
    validate_dataset_shape,
    validate_target_column
)

from src.components.data_transformation import (
    transform_data
)

from src.components.model_training import (
    create_model_pipelines,
    create_parameter_grids,
    train_models,
    select_best_model,
    save_best_model
)

from src.components.model_training import register_best_model
from src.components.model_evaluation import generate_evaluation_report

# ============================================================
# Configuration
# ============================================================

FILE_PATH = r"F:\CV\archive\Customer-Churn-Records.csv"


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":

    try:

        logger.info(
            "Bank Customer Churn pipeline started."
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "BANK CUSTOMER CHURN ML PIPELINE"
        )

        print(
            "=" * 70
        )

        # ====================================================
        # STEP 1: DATA INGESTION
        # ====================================================

        print(
            "\n[STEP 1] Data Ingestion"
        )

        df = load_local_data(
            FILE_PATH
        )

        print(
            "Dataset loaded successfully."
        )

        print(
            f"Dataset shape: {df.shape}"
        )

        # ====================================================
        # STEP 2: DATA VALIDATION
        # ====================================================

        print(
            "\n[STEP 2] Data Validation"
        )

        validate_columns(
            df
        )

        validate_dataset_shape(
            df
        )

        validate_target_column(
            df
        )

        print(
            "All data validation checks passed."
        )

        logger.info(
            "All data validation checks passed."
        )

        # ====================================================
        # STEP 3: DATA TRANSFORMATION
        # ====================================================

        print(
            "\n[STEP 3] Data Transformation"
        )

        (
            X_train,
            X_test,
            y_train,
            y_test,
            X_train_transformed,
            X_test_transformed,
            X_train_resampled,
            y_train_resampled,
            preprocessor
        ) = transform_data(
            df
        )

        # ====================================================
        # TRANSFORMATION RESULTS
        # ====================================================

        print(
            "\n" + "-" * 70
        )

        print(
            "TRANSFORMATION RESULTS"
        )

        print(
            "-" * 70
        )

        print(
            f"\nOriginal X_train shape: "
            f"{X_train.shape}"
        )

        print(
            f"Original X_test shape: "
            f"{X_test.shape}"
        )

        print(
            f"\nTransformed X_train shape: "
            f"{X_train_transformed.shape}"
        )

        print(
            f"Transformed X_test shape: "
            f"{X_test_transformed.shape}"
        )

        print(
            f"\nX_train shape after SMOTE: "
            f"{X_train_resampled.shape}"
        )

        print(
            f"y_train shape after SMOTE: "
            f"{y_train_resampled.shape}"
        )

        print(
            f"\ny_test shape: "
            f"{y_test.shape}"
        )

        # ====================================================
        # TRAINING DISTRIBUTION BEFORE SMOTE
        # ====================================================

        print(
            "\nTraining target distribution "
            "BEFORE SMOTE:"
        )

        print(
            y_train.value_counts()
        )

        print(
            "\nTraining target percentage "
            "BEFORE SMOTE:"
        )

        print(
            y_train
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

        # ====================================================
        # TRAINING DISTRIBUTION AFTER SMOTE
        # ====================================================

        print(
            "\nTraining target distribution "
            "AFTER SMOTE:"
        )

        print(
            y_train_resampled.value_counts()
        )

        print(
            "\nTraining target percentage "
            "AFTER SMOTE:"
        )

        print(
            y_train_resampled
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

        # ====================================================
        # TESTING DISTRIBUTION
        # ====================================================

        print(
            "\nTesting target distribution:"
        )

        print(
            y_test.value_counts()
        )

        print(
            "\nTesting target percentage:"
        )

        print(
            y_test
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

        # ====================================================
        # PREPROCESSOR VERIFICATION
        # ====================================================

        print(
            "\nPreprocessor type:"
        )

        print(
            type(preprocessor)
        )

        # ====================================================
        # STEP 4: MODEL TRAINING
        # ====================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "[STEP 4] MODEL TRAINING "
            "AND HYPERPARAMETER TUNING"
        )

        print(
            "=" * 70
        )

        logger.info(
            "Starting model training and "
            "hyperparameter tuning."
        )

        # ----------------------------------------------------
        # Create model pipelines
        # ----------------------------------------------------

        models = create_model_pipelines(
            preprocessor
        )

        print(
            "\nModels selected for training:"
        )

        for model_name in models:

            print(
                f"  - {model_name}"
            )

        # ----------------------------------------------------
        # Create hyperparameter grids
        # ----------------------------------------------------

        parameter_grids = (
            create_parameter_grids()
        )

        # ----------------------------------------------------
        # Train and tune models
        # ----------------------------------------------------

        (
            results,
            best_models
        ) = train_models(

            models=models,

            parameter_grids=parameter_grids,

            # IMPORTANT:
            # Use ORIGINAL training data here.
            # SMOTE is handled inside the
            # cross-validation pipeline.
            X_train=X_train,

            y_train=y_train,

            X_test=X_test,

            y_test=y_test
        )
        generate_evaluation_report(
            results=results,
            best_models=best_models,
            X_test=X_test,
            y_test=y_test,
)

        # ====================================================
        # STEP 5: MODEL COMPARISON
        # ====================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "[STEP 5] MODEL COMPARISON"
        )

        print(
            "=" * 70
        )

        for model_name, result in results.items():

            metrics = result[
                "test_metrics"
            ]

            print(
                f"\n{model_name}"
            )

            print(
                "-" * 40
            )

            print(
                f"CV ROC-AUC : "
                f"{result['cv_roc_auc']:.4f}"
            )

            print(
                f"Test ROC-AUC : "
                f"{metrics['roc_auc']:.4f}"
            )

            print(
                f"Test PR-AUC : "
                f"{metrics['pr_auc']:.4f}"
            )

            print(
                f"Test F1 : "
                f"{metrics['f1']:.4f}"
            )

            print(
                f"Test Precision : "
                f"{metrics['precision']:.4f}"
            )

            print(
                f"Test Recall : "
                f"{metrics['recall']:.4f}"
            )

            print(
                f"Test Accuracy : "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                "\nBest Parameters:"
            )

            print(
                result["best_params"]
            )

        # ====================================================
        # STEP 6: SELECT BEST MODEL
        # ====================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "[STEP 6] BEST MODEL SELECTION"
        )

        print(
            "=" * 70
        )

        (
            best_model_name,
            best_model
        ) = select_best_model(
            results,
            best_models
        )
        best_model_uri = results[best_model_name]["model_uri"]

        registered_model = register_best_model(
             model_uri=best_model_uri,
            registered_model_name="BankCustomerChurn",
        )

        print(
            f"Registered model: BankCustomerChurn "
            f"(version {registered_model.version})"
        )

        print(
            f"\nBest Model: "
            f"{best_model_name}"
        )

        best_metrics = results[
            best_model_name
        ][
            "test_metrics"
        ]

        print(
            "\nBest Model Test Performance:"
        )

        print(
            f"ROC-AUC   : "
            f"{best_metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC    : "
            f"{best_metrics['pr_auc']:.4f}"
        )

        print(
            f"F1        : "
            f"{best_metrics['f1']:.4f}"
        )

        print(
            f"Precision  : "
            f"{best_metrics['precision']:.4f}"
        )

        print(
            f"Recall     : "
            f"{best_metrics['recall']:.4f}"
        )

        print(
            f"Accuracy   : "
            f"{best_metrics['accuracy']:.4f}"
        )

        # ====================================================
        # STEP 7: SAVE BEST MODEL
        # ====================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "[STEP 7] SAVING BEST MODEL"
        )

        print(
            "=" * 70
        )

        model_path = save_best_model(
            best_model,
            best_model_name
        )

        print(
            f"\nBest model saved successfully:"
        )

        print(
            model_path
        )

        logger.info(
            f"Best model saved: {model_path}"
        )

        # ====================================================
        # PIPELINE COMPLETED
        # ====================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "BANK CUSTOMER CHURN ML PIPELINE "
            "COMPLETED SUCCESSFULLY"
        )

        print(
            "=" * 70
        )

        logger.info(
            "Bank Customer Churn pipeline "
            "completed successfully."
        )

    except Exception as error:

        logger.error(
            "Bank Customer Churn pipeline failed."
        )

        raise BankChurnException(
            error,
            sys
        )