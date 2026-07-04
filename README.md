# 🤖 AutoDS AI

### AI-Powered End-to-End Automated Machine Learning Platform

> Analyze • Clean • Train • Compare • Visualize • Report

AutoDS AI is an intelligent end-to-end machine learning platform that automates the complete data science workflow. Simply upload a CSV dataset, and the system automatically analyzes the data, cleans it, trains multiple machine learning models, selects the best-performing model, generates visualizations, and produces downloadable reports.

This project demonstrates the integration of AI-assisted data understanding with automated machine learning in a professional workflow.

---

# 🚀 Features

## 📊 Intelligent Dataset Inspection

* Automatically loads CSV datasets
* Detects numerical and categorical columns
* Identifies missing values
* Detects duplicate records
* Generates dataset statistics

---

## 🧠 AI Dataset Understanding (Gemini)

Uses Google Gemini AI to automatically determine:

* Problem Type
* Target Column
* Columns to Drop
* Missing Value Strategy
* Encoding Strategy
* Feature Scaling Strategy
* AI Reasoning

---

## 🧹 Automatic Data Cleaning

Performs:

* Remove unnecessary columns
* Handle missing values
* Remove duplicate records
* Label Encoding
* One-Hot Encoding
* Feature Scaling

---

## 🤖 Automated Machine Learning

Automatically trains multiple machine learning models:

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Gaussian Naive Bayes

The platform automatically:

* Trains all models
* Evaluates performance
* Compares accuracy
* Selects the best model
* Saves the best model

---

## 📈 Visualization Engine

Automatically generates:

* Correlation Heatmap
* Missing Values Plot
* Target Distribution
* Histograms
* Boxplots
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve
* Feature Importance (when supported)

---

## 📄 Automatic Report Generation

AutoDS AI generates:

* Evaluation Report
* Model Comparison Report
* Predictions CSV
* Dataset Schema JSON
* Run Summary
* Cleaned Dataset

---

## 🌐 Professional Streamlit Dashboard

Interactive web application with:

* CSV Upload
* Dataset Preview
* AI Dataset Schema
* Model Leaderboard
* Best Model Information
* Accuracy Metrics
* Interactive Visualizations
* Downloadable Outputs

---

# 📂 Project Structure

```text
AutoDS_AI/
│
├── agents/
│   └── schema_agent.py
│
├── tools/
│   ├── cleaner.py
│   ├── data_inspector.py
│   ├── model_trainer.py
│   └── visualizer.py
│
├── models/
│
├── outputs/
│   ├── plots/
│   ├── predictions.csv
│   ├── cleaned_dataset.csv
│   ├── evaluation_report.txt
│   ├── model_comparison.csv
│   ├── dataset_schema.json
│   └── run_summary.txt
│
├── data/
│
├── app.py
├── main_pipeline.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* Google Gemini AI
* Joblib

---

# ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/MuzamilRasul/AutoDS_AI.git
```

Move into the project directory:

```bash
cd AutoDS_AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

# 📊 Sample Workflow

```text
Upload CSV
      │
      ▼
Dataset Inspection
      │
      ▼
Gemini AI Analysis
      │
      ▼
Automatic Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Train Multiple ML Models
      │
      ▼
Compare Performance
      │
      ▼
Select Best Model
      │
      ▼
Generate Visualizations
      │
      ▼
Generate Reports
      │
      ▼
Download Results
```

---



* Dashboard Home
* Dataset Upload
* AI Dataset Schema
* Model Leaderboard
* Visualizations
* Reports

---

# 📁 Generated Outputs

```
models/
    best_model.pkl

outputs/
    cleaned_dataset.csv
    predictions.csv
    evaluation_report.txt
    model_comparison.csv
    dataset_schema.json
    run_summary.txt

outputs/plots/
    correlation_heatmap.png
    missing_values.png
    target_distribution.png
    histograms.png
    boxplots.png
    confusion_matrix.png
    roc_curve.png
    precision_recall_curve.png
```

---

# 🎯 Future Improvements

* SHAP Explainability
* Business Insights using Gemini
* PDF Report Export
* FastAPI Integration
* Docker Support
* Cloud Deployment
* Authentication System

---

# 👨‍💻 Author

**Muzamil Rasul**

Data Scientist | Machine Learning Engineer | Python Developer

GitHub: https://github.com/MuzamilRasul

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
