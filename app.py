import streamlit as st
import pandas as pd
import os
import json
from main_pipeline import run_pipeline

# --------------------------------------------------
# LOAD EXTERNAL CSS
# --------------------------------------------------

def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AutoDS AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# PROFESSIONAL SIDEBAR
# ==========================================

with st.sidebar:

    st.image(
        "https://img.icons8.com/fluency/96/artificial-intelligence.png",
        width=90
    )

    st.title("🤖 AutoDS AI")

    st.markdown("---")

    st.subheader("🚀 Platform Features")

    st.success("AI Dataset Inspection")
    st.success("Gemini Schema Detection")
    st.success("Automatic Data Cleaning")
    st.success("Full Automated Machine Learning")
    st.success("Best Model Selection")
    st.success("Data Visualization")
    st.success("AI Reports")
    st.success("Download Center")

    st.markdown("---")

    st.subheader("🧠 Machine Learning Models")

    st.write("✅ Logistic Regression")
    st.write("✅ Decision Tree")
    st.write("✅ Random Forest")
    st.write("✅ KNN")
    st.write("✅ SVM")
    st.write("✅ Naive Bayes")

    st.markdown("---")

    st.subheader("📈 Project Status")

    st.progress(100)

    st.success("Project Completion: 100%")

    st.markdown("---")

    st.info(
        """
### 🛠 Built With

- Gemini AI
- Python
- Scikit-Learn
- Pandas
- Streamlit
- Matplotlib
"""
    )

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "Not Uploaded"

if "best_model" not in st.session_state:
    st.session_state.best_model = "--"

if "best_accuracy" not in st.session_state:
    st.session_state.best_accuracy = "--"

if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False

# =====================================================
# HERO SECTION
# =====================================================

st.markdown(
    """
    <style>

    .main-title{
        font-size:55px;
        font-weight:800;
        color:#2E86DE;
        text-align:center;
        margin-bottom:0px;
    }

    .sub-title{
        font-size:26px;
        color:#555555;
        text-align:center;
        margin-top:-15px;
        margin-bottom:20px;
    }

    .description{
        font-size:20px;
        text-align:center;
        color:#666666;
        margin-bottom:30px;
    }

    .hero-box{
        background:linear-gradient(90deg,#E3F2FD,#FFFFFF);
        padding:30px;
        border-radius:18px;
        box-shadow:0px 4px 15px rgba(0,0,0,0.10);
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-box">

    <div class="main-title">
        🤖 AutoDS AI
    </div>

    <div class="sub-title">
        AI-Powered End-to-End Automated Machine Learning Platform
    </div>

    <div class="description">
        Analyze • Clean • Train • Compare • Visualize • Report
        <br><br>
        Upload any CSV dataset and let AI automatically inspect,
        clean, train multiple machine learning models, compare
        performance, generate visualizations, and create reports
        in one click.
    </div>

    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =====================================================
# DASHBOARD OVERVIEW
# =====================================================

st.markdown("## 📊 Dashboard Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📂 Dataset",
        st.session_state.dataset_name,
        "Uploaded" if st.session_state.dataset_name != "Not Uploaded" else "Waiting"
    )

with col2:
    st.metric(
        "🤖 Models",
        "6",
        "Available"
    )

with col3:
    st.metric(
        "🏆 Best Model",
        st.session_state.best_model,
        "Done" if st.session_state.pipeline_done else "Pending"
    )

with col4:
    st.metric(
        "🎯 Accuracy",
        st.session_state.best_accuracy,
        "Done" if st.session_state.pipeline_done else "Pending"
    )

st.markdown("---")

# ==========================================
# DATASET UPLOAD SECTION
# ==========================================

st.markdown(
    '<div class="section-title">📂 Upload Your Dataset</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    st.success("✅ Dataset uploaded successfully!")

    st.session_state.dataset_name = uploaded_file.name

    df = pd.read_csv(uploaded_file)

    st.markdown("### 👀 Dataset Preview")

    st.dataframe(df.head(), use_container_width=True)

    rows, cols = df.shape

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", rows)

    with col2:
        st.metric("Columns", cols)

st.markdown("---")

# ==========================================
# RUN PIPELINE BUTTON
# ==========================================

run_analysis = st.button(
    "🚀 Run Full Auto Machine Learning",
    use_container_width=True
)

if uploaded_file is not None and run_analysis:

    with open("data/raw/uploaded_dataset.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ Dataset uploaded successfully!")

    with st.spinner("🤖 Running Full Auto Machine Learning Pipeline..."):

        run_pipeline("data/raw/uploaded_dataset.csv")

    st.session_state.pipeline_done = True

    comparison_path = "outputs/model_comparison.csv"

    if os.path.exists(comparison_path):
        comparison_df = pd.read_csv(comparison_path)
        best_model = comparison_df.iloc[0]

        st.session_state.best_model = best_model["Model"]
        st.session_state.best_accuracy = f"{best_model['Accuracy']:.4f}"

    st.rerun()

st.markdown("---")

# =====================================================
# RESULTS TABS (persist after rerun)
# =====================================================

if st.session_state.pipeline_done:

    st.success("🎉 Analysis Completed Successfully!")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Results",
        "🧠 AI Schema",
        "📊 Visualizations",
        "📥 Downloads"
    ])

    # ---------------------------------------------------
    # TAB 1 — RESULTS
    # ---------------------------------------------------
    with tab1:

        comparison_path = "outputs/model_comparison.csv"

        if os.path.exists(comparison_path):
            comparison_df = pd.read_csv(comparison_path)

            best_model = comparison_df.iloc[0]

            st.markdown(
                "<h2 style='text-align:center;'>🏆 Machine Learning Results</h2>",
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric("🏆 Best Model", best_model["Model"])

            with col2:
                st.metric("🎯 Accuracy", f"{best_model['Accuracy']:.4f}")

            st.markdown("---")
            st.subheader("📊 Model Leaderboard")

            comparison_df.index = comparison_df.index + 1

            st.dataframe(
                comparison_df,
                use_container_width=True
            )
        else:
            st.warning("Model comparison file not found.")

    # ---------------------------------------------------
    # TAB 2 — AI SCHEMA
    # ---------------------------------------------------
    with tab2:

        schema_path = "outputs/dataset_schema.json"

        if os.path.exists(schema_path):

            with open(schema_path) as f:
                schema = json.load(f)

            st.markdown(
                "<h2 style='text-align:center;'>🧠 AI Dataset Schema</h2>",
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric("🧩 Problem Type", schema.get("problem_type", "--").title())

            with col2:
                st.metric("🎯 Target Column", schema.get("target_column", "--"))

            st.markdown("")

            def render_tags(title, items, color="#2E86DE"):
                if not items:
                    return
                tags_html = "".join(
                    f"<span style='background:{color}15;color:{color};"
                    f"padding:6px 14px;margin:4px;border-radius:20px;"
                    f"display:inline-block;font-size:14px;font-weight:600;'>{item}</span>"
                    for item in items
                )
                st.markdown(f"**{title}**", unsafe_allow_html=False)
                st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)
                st.markdown("")

            render_tags("🔢 Numeric Columns", schema.get("numeric_columns", []), "#2E86DE")
            render_tags("🔤 Categorical Columns", schema.get("categorical_columns", []), "#8E44AD")
            render_tags("🗑️ Dropped Columns", schema.get("drop_columns", []), "#E74C3C")
            render_tags("🔡 Encoding Required", schema.get("encoding_required", []), "#F39C12")
            render_tags("📏 Scaling Required", schema.get("scaling_required", []), "#16A085")

            missing_strategy = schema.get("missing_value_strategy", {})

            if missing_strategy:
                st.markdown("**🩹 Missing Value Strategy**")

                strategy_df = pd.DataFrame(
                    list(missing_strategy.items()),
                    columns=["Column", "Fill Strategy"]
                )
                strategy_df.index = strategy_df.index + 1

                st.dataframe(strategy_df, use_container_width=True)

            reasoning = schema.get("reasoning", "")

            if reasoning:
                st.markdown("**🧠 AI Reasoning**")
                st.markdown(
                    f"""
                    <div style="
                        background:#F8F9FA;
                        border-left:4px solid #2E86DE;
                        padding:16px 20px;
                        border-radius:8px;
                        font-size:15px;
                        line-height:1.6;
                        color:#333333;
                    ">
                    {reasoning}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("Dataset schema file not found.")

    # ---------------------------------------------------
    # TAB 3 — VISUALIZATIONS
    # ---------------------------------------------------
    with tab3:

        st.markdown(
            "<h2 style='text-align:center;'>📈 Generated Visualizations</h2>",
            unsafe_allow_html=True
        )

        plots = [
            ("Correlation Heatmap", "correlation_heatmap.png"),
            ("Missing Values", "missing_values.png"),
            ("Target Distribution", "target_distribution.png"),
            ("Histograms", "histograms.png"),
            ("Boxplots", "boxplots.png"),
            ("Confusion Matrix", "confusion_matrix.png"),
            ("ROC Curve", "roc_curve.png"),
            ("Precision Recall Curve", "precision_recall_curve.png"),
        ]

        any_plot_found = False

        for title, filename in plots:

            image_path = os.path.join("outputs", "plots", filename)

            if os.path.exists(image_path):
                any_plot_found = True
                st.markdown(f"### {title}")
                st.image(image_path, use_container_width=True)

        if not any_plot_found:
            st.warning("No visualizations found yet.")

    # ---------------------------------------------------
    # TAB 4 — DOWNLOADS
    # ---------------------------------------------------
    with tab4:

        st.markdown(
            "<h2 style='text-align:center;'>📥 Download Center</h2>",
            unsafe_allow_html=True
        )

        downloads = [
            ("🧹 Cleaned Dataset", "outputs/cleaned_dataset.csv", "text/csv"),
            ("🔮 Predictions", "outputs/predictions.csv", "text/csv"),
            ("📄 Report", "outputs/report.pdf", "application/pdf"),
            ("📊 Model Comparison", "outputs/model_comparison.csv", "text/csv"),
            ("🏆 Best Model", "outputs/best_model.pkl", "application/octet-stream"),
            ("🧠 Dataset Schema", "outputs/dataset_schema.json", "application/json"),
        ]

        any_file_found = False

        for label, path, mime in downloads:

            if os.path.exists(path):
                any_file_found = True

                with open(path, "rb") as f:
                    st.download_button(
                        label=f"Download {label}",
                        data=f,
                        file_name=os.path.basename(path),
                        mime=mime,
                        use_container_width=True
                    )

        if not any_file_found:
            st.warning("No downloadable files found yet.")

# =====================================================
# PROFESSIONAL FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
<div style="text-align:center">

### 🤖 AutoDS AI

AI-Powered End-to-End Automated Machine Learning Platform

Built with ❤️ using

Gemini AI • Streamlit • Scikit-Learn • Python

<p style="font-size:22px; font-weight:700; color:#2E86DE; margin-top:15px;">
Developed by Muzamil Rasul
</p>

</div>
""",
unsafe_allow_html=True
)