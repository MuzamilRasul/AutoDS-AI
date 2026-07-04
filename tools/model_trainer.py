from typing import Tuple
import os
import joblib

import pandas as pd

from sklearn.model_selection import train_test_split

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.base import BaseEstimator

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class ModelTrainer:
    """
    AutoDS AI Model Trainer

    Responsibilities
    ----------------
    - Prepare features and target
    - Split dataset
    - Train multiple ML models
    - Evaluate every model
    - Compare model performance
    - Select the best model
    - Save trained models
    - Load trained models
    """

    def __init__(self) -> None:
        """
        Initialize all machine learning models.
        """

        self.models = {
            "Logistic Regression": LogisticRegression(
                random_state=42,
                max_iter=1000
            ),

            "Decision Tree": DecisionTreeClassifier(
                random_state=42
            ),

            "Random Forest": RandomForestClassifier(
                random_state=42,
                n_estimators=100
            ),

            "KNN": KNeighborsClassifier(
                n_neighbors=5
            ),

            "SVM": SVC(
                probability=True,
                random_state=42
            ),

            "Naive Bayes": GaussianNB()
        }

    def _validate_dataframe(
        self,
        df: pd.DataFrame
    ) -> None:
        """
        Validate that the input is a pandas DataFrame.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

    def prepare_features(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix (X) and target vector (y).
        """

        self._validate_dataframe(df)

        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found."
            )

        X = df.drop(columns=[target_column])
        y = df[target_column]

        return X, y

    def split_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split dataset into training and testing sets.
        """

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

        return X_train, X_test, y_train, y_test

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> BaseEstimator:
        """
        Train a Logistic Regression model.

        NOTE: This still trains a single hardcoded model and does not
        yet use self.models. Update this once you have instructions
        for training/comparing all models.
        """

        model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )

        model.fit(X_train, y_train)

        return model

    def train_all_models(
        self,
        X_train,
        y_train
    ):
        """
        Train all available machine learning models.

        Returns
        -------
        dict
            Dictionary containing all trained models.
        """

        trained_models = {}

        print("\n🚀 Training Multiple Machine Learning Models...\n")

        for model_name, model in self.models.items():

            print(f"Training {model_name}...")

            model.fit(
                X_train,
                y_train
            )

            trained_models[model_name] = model

            print(f"✅ {model_name} completed.")

        print("\n🎉 All models trained successfully!\n")

        return trained_models

    def evaluate_all_models(
        self,
        trained_models,
        X_test,
        y_test
    ):
        """
        Evaluate all trained models.

        Returns
        -------
        dict
            Dictionary containing evaluation results
            for every model.
        """

        results = {}

        print("\n📊 Evaluating Models...\n")

        for model_name, model in trained_models.items():

            predictions = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            results[model_name] = {

                "model": model,

                "accuracy": accuracy,

                "precision": precision,

                "recall": recall,

                "f1_score": f1,

                "confusion_matrix": confusion_matrix(
                    y_test,
                    predictions
                ),

                "classification_report": classification_report(
                    y_test,
                    predictions
                )

            }

            print(
                f"✅ {model_name} | "
                f"Accuracy: {accuracy:.4f}"
            )

        return results

    def get_best_model(self, evaluation_results):
        """
        Select the best performing model based on accuracy.

        Parameters
        ----------
        evaluation_results : dict
            Results returned by evaluate_all_models()

        Returns
        -------
        tuple
            (best_model_name, best_model, best_metrics)
        """

        best_model_name = None
        best_model = None
        best_metrics = None
        best_accuracy = 0

        for model_name, metrics in evaluation_results.items():

            if metrics["accuracy"] > best_accuracy:

                best_accuracy = metrics["accuracy"]
                best_model_name = model_name
                best_model = metrics["model"]
                best_metrics = metrics

        print("\n==============================")
        print("🏆 BEST MODEL SELECTED")
        print("==============================")

        print(f"Model    : {best_model_name}")
        print(f"Accuracy : {best_accuracy:.4f}")

        return (
            best_model_name,
            best_model,
            best_metrics
        )

    def evaluate_model(
        self,
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> dict:
        """
        Evaluate the trained machine learning model.
        """

        predictions = model.predict(X_test)

        results = {
            "accuracy": accuracy_score(
                y_test,
                predictions
            ),
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions
            )
        }

        return results

    def save_model(
        self,
        model: BaseEstimator,
        file_path: str
    ) -> None:
        """
        Save trained model to disk.
        """

        joblib.dump(model, file_path)

    def load_model(
        self,
        file_path: str
    ) -> BaseEstimator:
        """
        Load a trained model from disk.
        """

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Model file not found: {file_path}"
            )

        model = joblib.load(file_path)

        return model

    def predict(
        self,
        model: BaseEstimator,
        X: pd.DataFrame
    ) -> list:
        """
        Make predictions using a trained model.

        Parameters
        ----------
        model : BaseEstimator
            Trained machine learning model.

        X : pd.DataFrame
            Input features.

        Returns
        -------
        list
            Predicted values.
        """

        predictions = model.predict(X)

        return predictions.tolist()