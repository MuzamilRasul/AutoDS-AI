import os
import json

from tools.data_inspector import DataInspector
from agents.schema_agent import SchemaAgent
from tools.cleaner import Cleaner
from tools.model_trainer import ModelTrainer
from tools.visualizer import Visualizer


def run_pipeline(file_path: str):
    """
    Full AutoDS AI Pipeline

    Steps
    -----
    1. Inspect dataset
    2. Generate schema using Gemini
    3. Clean dataset automatically
    4. Train Machine Learning model
    5. Save trained model
    6. Load model and make predictions
    7. Save evaluation report
    8. Save predictions & project artifacts
    """

    print("\n==============================")
    print("🚀 STARTING AUTO DATA PIPELINE")
    print("==============================\n")

    # -----------------------------
    # STEP 1: DATA INSPECTOR
    # -----------------------------
    inspector = DataInspector()
    df = inspector.load_dataset(file_path)

    if df is None:
        print("❌ Failed to load dataset")
        return

    dataset_info = inspector.inspect_dataset(df)

    print("\n📊 Dataset Info Generated")

    # -----------------------------
    # STEP 2: SCHEMA AGENT (AI)
    # -----------------------------
    print("\n🧠 Generating Schema using Gemini...\n")

    schema_agent = SchemaAgent()
    schema = schema_agent.analyze_schema(dataset_info)

    print("\n✅ Schema Generated:")
    print(schema)

    # -----------------------------
    # STEP 3: CLEANER ENGINE
    # -----------------------------
    print("\n🧹 Starting Data Cleaning...\n")

    cleaner = Cleaner()

    df = cleaner.drop_columns(df, schema)
    df = cleaner.remove_duplicates(df)
    df = cleaner.fill_missing_values(df, schema)
    df = cleaner.encode_categorical_columns(df, schema)

    try:
        df = cleaner.scale_numeric_columns(df, schema)
    except Exception as e:
        print(f"⚠️ Scaling skipped: {e}")

    print("\n✅ CLEANING COMPLETED!")

    # -----------------------------
    # FINAL CLEANED DATASET
    # -----------------------------
    print("\n==============================")
    print("📦 FINAL CLEANED DATASET")
    print("==============================")

    print(df.head())

    print("\n📜 CLEANING LOG:")
    for log in cleaner.get_cleaning_log():
        print("•", log)

    # -----------------------------
    # SAVE CLEANED DATASET
    # -----------------------------
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "cleaned_dataset.csv"
    )

    df.to_csv(output_path, index=False)

    print("\n💾 CLEANED DATASET SAVED SUCCESSFULLY!")
    print(f"📁 Location: {output_path}")

    # ============================================
    # DATA VISUALIZATIONS
    # ============================================

    visualizer = Visualizer()

    print("\n==============================")
    print("📊 GENERATING DATA VISUALIZATIONS")
    print("==============================")

    visualizer.correlation_heatmap(df)

    visualizer.missing_values(df)

    visualizer.target_distribution(
        df,
        target_column=schema["target_column"]
    )

    visualizer.histograms(df)

    visualizer.boxplots(df)

    # ==================================================
    # STEP 4 : MACHINE LEARNING MODEL TRAINING
    # ==================================================

    print("\n==============================")
    print("🤖 MACHINE LEARNING")
    print("==============================")

    trainer = ModelTrainer()

    target_column = schema["target_column"]

    # -----------------------------------------
    # Prepare Features
    # -----------------------------------------

    X, y = trainer.prepare_features(
        df,
        target_column
    )

    # -----------------------------------------
    # Train / Test Split
    # -----------------------------------------

    X_train, X_test, y_train, y_test = trainer.split_dataset(
        X,
        y
    )

    # -----------------------------------------
    # Train All Models
    # -----------------------------------------

    trained_models = trainer.train_all_models(
        X_train,
        y_train
    )

    # -----------------------------------------
    # Evaluate All Models
    # -----------------------------------------

    evaluation_results = trainer.evaluate_all_models(
        trained_models,
        X_test,
        y_test
    )

    # -----------------------------------------
    # Select Best Model
    # -----------------------------------------

    best_model_name, best_model, best_metrics = trainer.get_best_model(
        evaluation_results
    )

    print("\n==============================")
    print("📊 MODEL LEADERBOARD")
    print("==============================")

    for model_name, metrics in evaluation_results.items():

        print(
            f"{model_name:<25}"
            f"{metrics['accuracy']:.4f}"
        )

    print("\n🏆 Best Model")
    print(f"Model    : {best_model_name}")
    print(f"Accuracy : {best_metrics['accuracy']:.4f}")

    # ==================================================
    # STEP 5 : SAVE TRAINED MODEL
    # ==================================================

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(
        model_dir,
        "best_model.pkl"
    )

    trainer.save_model(
        best_model,
        model_path
    )

    print("\n💾 MODEL SAVED SUCCESSFULLY!")
    print(f"📁 Location: {model_path}")

    # ==================================================
    # STEP 6 : LOAD MODEL & MAKE PREDICTIONS
    # ==================================================

    print("\n==============================")
    print("🔮 MODEL PREDICTIONS")
    print("==============================")

    loaded_model = trainer.load_model(
        model_path
    )

    print("✅ MODEL LOADED SUCCESSFULLY!")

    predictions = trainer.predict(
        loaded_model,
        X_test
    )

    print("\n📌 First 10 Predictions:")
    print(predictions[:10])

    print("\n🎉 Prediction completed successfully!")

    # ============================================
    # MACHINE LEARNING VISUALIZATIONS
    # ============================================

    print("\n==============================")
    print("📈 GENERATING ML VISUALIZATIONS")
    print("==============================")

    visualizer.confusion_matrix_plot(
        loaded_model,
        X_test,
        y_test
    )

    try:

        visualizer.roc_curve(
            loaded_model,
            X_test,
            y_test
        )

    except Exception:

        print("⚠️ ROC Curve skipped.")

    try:

        visualizer.precision_recall_curve(
            loaded_model,
            X_test,
            y_test
        )

    except Exception:

        print("⚠️ Precision-Recall Curve skipped.")

    visualizer.feature_importance(
        loaded_model,
        X
    )

    # ==================================================
    # STEP 7 : SAVE EVALUATION REPORT
    # ==================================================

    report_path = os.path.join(
        output_dir,
        "evaluation_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as report:

        report.write("=====================================\n")
        report.write("      AutoDS AI Evaluation Report\n")
        report.write("=====================================\n\n")

        report.write(f"Problem Type : {schema['problem_type']}\n")
        report.write(f"Target Column : {schema['target_column']}\n\n")

        report.write(
            f"Best Model : {best_model_name}\n"
        )

        report.write(
            f"Accuracy : {best_metrics['accuracy']:.4f}\n\n"
        )

        report.write("Confusion Matrix\n")
        report.write("----------------\n")
        report.write(
            str(best_metrics["confusion_matrix"])
        )
        report.write("\n\n")

        report.write("Classification Report\n")
        report.write("---------------------\n")
        report.write(
            best_metrics["classification_report"]
        )
        report.write("\n\n")

        report.write("Cleaning Log\n")
        report.write("------------\n")

        for log in cleaner.get_cleaning_log():
            report.write(f"- {log}\n")

    print("\n📝 EVALUATION REPORT SAVED!")
    print(f"📁 Location: {report_path}")

    # ==================================================
    # SAVE MODEL COMPARISON
    # ==================================================

    import pandas as pd

    comparison_data = []

    for model_name, metrics in evaluation_results.items():

        comparison_data.append({

            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1_score"]

        })

    comparison_df = pd.DataFrame(comparison_data)

    comparison_df = comparison_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    comparison_path = os.path.join(
        output_dir,
        "model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    print("\n📊 Model comparison saved!")
    print(f"📁 Location: {comparison_path}")

    # ==================================================
    # STEP 8 : SAVE PREDICTIONS & PROJECT ARTIFACTS
    # ==================================================

    print("\n==============================")
    print("📦 SAVING PROJECT ARTIFACTS")
    print("==============================")

    # ---------------------------------------
    # Save predictions
    # ---------------------------------------

    predictions_df = X_test.copy()
    predictions_df["Actual"] = y_test.values
    predictions_df["Prediction"] = predictions

    predictions_path = os.path.join(
        output_dir,
        "predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    print("✅ Predictions saved!")
    print(f"📁 Location: {predictions_path}")

    # ---------------------------------------
    # Save AI schema
    # ---------------------------------------

    schema_path = os.path.join(
        output_dir,
        "dataset_schema.json"
    )

    with open(schema_path, "w", encoding="utf-8") as file:
        json.dump(
            schema,
            file,
            indent=4
        )

    print("✅ Dataset schema saved!")
    print(f"📁 Location: {schema_path}")

    # ---------------------------------------
    # Save project summary
    # ---------------------------------------

    summary_path = os.path.join(
        output_dir,
        "run_summary.txt"
    )

    with open(summary_path, "w", encoding="utf-8") as file:

        file.write("=====================================\n")
        file.write("        AutoDS AI Run Summary\n")
        file.write("=====================================\n\n")

        file.write(f"Dataset : {file_path}\n")
        file.write(f"Problem Type : {schema['problem_type']}\n")
        file.write(f"Target Column : {schema['target_column']}\n\n")

        file.write(f"Rows : {len(df)}\n")
        file.write(f"Columns : {len(df.columns)}\n\n")

        file.write(f"Best Model : {best_model_name}\n")
        file.write(f"Accuracy : {best_metrics['accuracy']:.4f}\n\n")

        file.write("Generated Files\n")
        file.write("---------------------\n")
        file.write("- cleaned_dataset.csv\n")
        file.write("- evaluation_report.txt\n")
        file.write("- model_comparison.csv\n")
        file.write("- predictions.csv\n")
        file.write("- dataset_schema.json\n")
        file.write("- best_model.pkl\n")

    print("✅ Run summary saved!")
    print(f"📁 Location: {summary_path}")

    print("\n🎉 ALL PROJECT FILES GENERATED SUCCESSFULLY!")

    return df


# -----------------------------
# RUN PIPELINE
# -----------------------------
if __name__ == "__main__":

    file_path = "data/raw/train.csv"

    run_pipeline(file_path)