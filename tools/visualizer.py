import os

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


class Visualizer:

    def __init__(self):

        self.output_folder = "outputs/plots"

        os.makedirs(
            self.output_folder,
            exist_ok=True
        )

    def correlation_heatmap(self, df):

        correlation = df.corr(numeric_only=True)

        plt.figure(figsize=(10, 8))

        plt.imshow(
            correlation,
            cmap="coolwarm",
            interpolation="nearest"
        )

        plt.colorbar()

        plt.xticks(
            range(len(correlation.columns)),
            correlation.columns,
            rotation=90
        )

        plt.yticks(
            range(len(correlation.columns)),
            correlation.columns
        )

        plt.title("Correlation Heatmap")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_folder,
                "correlation_heatmap.png"
            )
        )

        plt.close()

        print("✅ Correlation Heatmap Saved")

    def target_distribution(self, df, target_column):

        plt.figure(figsize=(7,5))

        df[target_column].value_counts().plot(
            kind="bar"
        )

        plt.title("Target Distribution")
        plt.xlabel(target_column)
        plt.ylabel("Count")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_folder,
                "target_distribution.png"
            )
        )

        plt.close()

        print("✅ Target Distribution Saved")

    def histograms(self, df):

        df.hist(
            figsize=(14,10),
            bins=20
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_folder,
                "histograms.png"
            )
        )

        plt.close()

        print("✅ Histograms Saved")

    def boxplots(self, df):

        numeric_df = df.select_dtypes(include="number")

        plt.figure(figsize=(15,7))

        numeric_df.boxplot(rot=90)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_folder,
                "boxplots.png"
            )
        )

        plt.close()

        print("✅ Boxplots Saved")

    def missing_values(self, df):

        missing = df.isnull().sum()

        plt.figure(figsize=(10,5))

        missing.plot(kind="bar")

        plt.title("Missing Values")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_folder,
                "missing_values.png"
            )
        )

        plt.close()

        print("✅ Missing Values Plot Saved")

    def confusion_matrix_plot(self, model, X_test, y_test):

        plt.figure(figsize=(6,6))

        ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test
        )

        plt.title("Confusion Matrix")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_folder,
                "confusion_matrix.png"
            )
        )

        plt.close()

        print("✅ Confusion Matrix Saved")

    def roc_curve(self, model, X_test, y_test):

        plt.figure(figsize=(6,6))

        RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test
        )

        plt.title("ROC Curve")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_folder,
                "roc_curve.png"
            )
        )

        plt.close()

        print("✅ ROC Curve Saved")

    def precision_recall_curve(self, model, X_test, y_test):

        plt.figure(figsize=(6,6))

        PrecisionRecallDisplay.from_estimator(
            model,
            X_test,
            y_test
        )

        plt.title("Precision Recall Curve")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_folder,
                "precision_recall_curve.png"
            )
        )

        plt.close()

        print("✅ Precision Recall Curve Saved")

    def feature_importance(self, model, X):

        if not hasattr(model, "feature_importances_"):

            print("⚠️ Feature importance not available for this model.")

            return

        importance = model.feature_importances_

        plt.figure(figsize=(10,6))

        plt.barh(
            X.columns,
            importance
        )

        plt.title("Feature Importance")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_folder,
                "feature_importance.png"
            )
        )

        plt.close()

        print("✅ Feature Importance Saved")