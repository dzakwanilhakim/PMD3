import streamlit as st
import pandas as pd
import joblib
from PIL import Image
from backend import Prediction

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="PMD3"
)

# --- Example Data ---
EXAMPLE_DATA = pd.DataFrame({
    "Sample_Name": ["Patient001", "Patient002", "Patient003", "Patient004", "Patient005"],
    "MYH11":  [16.425486, 12.04352229, 16.35163783, 12.10629686, 16.0078846],
    "PRMT1":  [10.39456214, 8.612377672, 11.55593554, 12.44179342, 11.10911678],
    "TENT4A": [8.870025006, 10.31208222, 9.015171788, 9.027366233, 8.955317584],
})

# --- Header and Logos ---
header_col1, header_col2 = st.columns([4, 1])

with header_col1:
    st.title("PMD3")
    st.subheader("Prostate Metastatic Detection based on 3 RNA Biomarker")
    st.write("https://github.com/dzakwanilhakim/PMD3")
    st.write("https://github.com/dzakwanilhakim/prostate-metastasis-ml-rna")

with header_col2:
    try:
        st.image(Image.open('assets/logo1.png'), width=120)
    except FileNotFoundError:
        st.warning("logo1.png not found.")

st.divider()

# --- Caching Function to Load Models ---
@st.cache_resource
def load_models_and_data():
    try:
        scaler = joblib.load('models/scaler_3.pkl')
        x_background = joblib.load('models/X_background.pkl')
        lr_model = joblib.load('models/best_lr.pkl')
        nn_model = None
        return scaler, x_background, lr_model, nn_model
    except FileNotFoundError as e:
        st.error(f"Model loading error: {e}. Please ensure all .pkl files are in the models/ directory.")
        return None, None, None, None

# --- Tabs ---
tab1, tab2 = st.tabs(["Prediction", "Documentation"])

# --- PREDICTION TAB ---
with tab1:
    scaler, x_background, lr_model, nn_model = load_models_and_data()

    # Initialize session state
    if 'prediction_instance' not in st.session_state:
        st.session_state.prediction_instance = None
    if 'selected_row_index' not in st.session_state:
        st.session_state.selected_row_index = None
    if 'input_df' not in st.session_state:
        st.session_state.input_df = None

    # --- Step 1: Input Data ---
    st.subheader("1. Provide Patient Data")

    input_col1, input_col2 = st.columns([2, 1])

    with input_col1:
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            help="CSV format: Sample_Name, MYH11, PRMT1, TENT4A. Values in log2 or rlog format. Multiple patients allowed."
        )
        if uploaded_file is not None:
            st.session_state.input_df = pd.read_csv(uploaded_file)

    with input_col2:
        st.markdown("**No data on hand?**")
        if st.button("Load example data", use_container_width=True):
            st.session_state.input_df = EXAMPLE_DATA.copy()
        st.download_button(
            label="Download example CSV",
            data=EXAMPLE_DATA.to_csv(index=False).encode("utf-8"),
            file_name="example_patients.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # --- Preview & Predict ---
    if st.session_state.input_df is not None:
        with st.expander("Preview input data", expanded=False):
            st.dataframe(st.session_state.input_df, hide_index=True, use_container_width=True)

        st.subheader("2. Run Prediction")
        predict_clicked = st.button(
            "Predict (Logistic Regression)",
            type="primary",
            use_container_width=False,
        )

        if predict_clicked and lr_model:
            try:
                with st.spinner('Analyzing data... This may take a moment.'):
                    pred_instance = Prediction(
                        input_df=st.session_state.input_df,
                        scaler=scaler,
                        x_background=x_background,
                        model=lr_model,
                    )
                    pred_instance.run_analysis()

                st.session_state.prediction_instance = pred_instance
                st.session_state.selected_row_index = 0
                st.success("Analysis complete!")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

    # --- Results ---
    if st.session_state.prediction_instance:
        instance = st.session_state.prediction_instance

        results_data = []
        for idx, pred_info in instance.predictions.items():
            results_data.append({
                "Sample Name": instance.rawdata.loc[idx, "Sample_Name"],
                "Metastatic Probability": pred_info['probability'],
                "Prediction": pred_info['label']
            })
        results_df = pd.DataFrame(results_data)

        st.divider()
        st.subheader("Prediction Results")

        col1, col2 = st.columns([1.3, 1.5])

        with col1:
            st.dataframe(
                results_df.style.format({"Metastatic Probability": "{:.2f}"}),
                hide_index=True,
                use_container_width=True
            )

            selected_sample = st.selectbox(
                "Select a sample to view plots:",
                options=results_df['Sample Name'],
                index=st.session_state.selected_row_index or 0,
                key="sample_selector"
            )

            if selected_sample:
                selected_idx = results_df[results_df['Sample Name'] == selected_sample].index[0]
                st.session_state.selected_row_index = selected_idx

        with col2:
            selected_idx = st.session_state.selected_row_index
            if selected_idx is not None:
                sample_name = instance.rawdata.loc[selected_idx, "Sample_Name"]
                st.markdown(f"**Analysis for: {sample_name}**")

                with st.spinner('Generating plots...'):
                    waterfall_fig = instance.waterfall_plot(selected_idx)
                    st.pyplot(waterfall_fig)

                    barplot_fig = instance.barplot(selected_idx)
                    st.pyplot(barplot_fig)
            else:
                st.info("Select a sample on the left to display its explanation plots.")
    elif st.session_state.input_df is None:
        st.info("Upload a CSV file or load the example data to begin.")

# --- DOCUMENTATION TAB ---
with tab2:
    st.header("Documentation")

    st.subheader("Model Performance Metrics")
    try:
        st.image(Image.open('assets/model_performance.png'), caption="Model performance comparison.")
    except FileNotFoundError:
        st.warning("model_performance.png not found.")

    st.divider()

    st.subheader("Feature Importance (SHAP Beeswarm Plot)")
    try:
        st.image(Image.open('assets/beeswarm_plot.png'), caption="SHAP summary plot showing feature importance and impact.")
    except FileNotFoundError:
        st.warning("beeswarm_plot.png not found.")

# --- Footer ---
st.divider()
st.caption("Artificial Neural Network and Lasso-Logistic Regression for Prostate Adenocarcinoma Metastatic Classification and Biomarker Discovery, 2025. Dzakwanil Hakim, Elizabeth Loho")
st.caption("Dzakwanil Hakim")
st.caption("dzakwanilhd@gmail.com")