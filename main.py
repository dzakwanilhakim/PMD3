import streamlit as st
import pandas as pd
import joblib
from PIL import Image
# import tensorflow as tf # Not needed for LR-only testing
from backend import Prediction # Import the backend class

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="PMD3"
)

# --- Header and Logos ---
# Create columns for the header section
header_col1, header_col2 = st.columns([4, 1])

with header_col1:
    st.title("PMD3")
    st.subheader("Prostate Metastatic Detection based on 3 RNA Biomarker")
    
with header_col2:
    try:
        st.image(Image.open('logo1.png'), width=120)
    except FileNotFoundError:
        st.warning("logo1.png not found.")

st.divider()

# --- Caching Function to Load Models ---
@st.cache_resource
def load_models_and_data():
    """
    Load all necessary model and data files.
    The st.cache_resource decorator ensures this function runs only once.
    """
    try:
        scaler = joblib.load('scaler_3.pkl')
        x_background = joblib.load('X_background.pkl')
        lr_model = joblib.load('best_lr.pkl')
        # Suppress loading the Neural Network model for now to bypass the unpickling error.
        nn_model = None # Set nn_model to None
        return scaler, x_background, lr_model, nn_model
    except FileNotFoundError as e:
        st.error(f"Model loading error: {e}. Please ensure all .pkl files are in the root directory.")
        return None, None, None, None

# --- Main App UI ---
# Create tabs for different sections of the app
tab1, tab2 = st.tabs(["Prediction", "Documentation"])

# --- PREDICTION TAB ---
with tab1:
    # Load data and models securely
    scaler, x_background, lr_model, nn_model = load_models_and_data()

    # Initialize session state to store results and user selections
    if 'prediction_instance' not in st.session_state:
        st.session_state.prediction_instance = None
    if 'selected_row_index' not in st.session_state:
        st.session_state.selected_row_index = None

    # --- Sidebar for Controls ---
    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("1. Upload Patient Data", type=["csv"])
        st.caption("Data is in log2 or rlog format.")
        st.caption("CSV format: Sample_Name, MYH11, PRMT1, TENT4A.")
        st.caption("The file can contain more than one patient.")
        
        # Modified to only require the logistic regression model for testing
        if uploaded_file and lr_model:
            st.header("2. Predict")
            
            if st.button("Predict (Logistic Regression)", type="primary", use_container_width=True):
                try:
                    input_df = pd.read_csv(uploaded_file)
                    # Hardcode to use the logistic regression model
                    chosen_model = lr_model

                    # Instantiate the backend class and run the analysis
                    with st.spinner('Analyzing data... This may take a moment.'):
                        pred_instance = Prediction(
                            input_df=input_df,
                            scaler=scaler,
                            x_background=x_background,
                            model=chosen_model
                        )
                        pred_instance.run_analysis()
                    
                    # Store the results in the session state for reuse
                    st.session_state.prediction_instance = pred_instance
                    st.session_state.selected_row_index = 0 # Default to selecting the first row
                    st.success("Analysis complete!")

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")

    # --- Main Panel for Displaying Results ---
    if st.session_state.prediction_instance:
        instance = st.session_state.prediction_instance
        
        # Prepare results table with the requested column names
        results_data = []
        for idx, pred_info in instance.predictions.items():
            results_data.append({
                "Sample Name": instance.rawdata.loc[idx, "Sample_Name"],
                "Metastatic Probability": pred_info['probability'],
                "Prediction": pred_info['label']
            })
        results_df = pd.DataFrame(results_data)

        st.header("Prediction Results")
        st.markdown("Select a sample from the dropdown menu to view detailed plots.")

        # Create two columns for the table and the plots, making the left one wider
        col1, col2 = st.columns([1.3, 1.5])

        with col1:
            # Display the results table header
            st.dataframe(
                results_df.style.format({"Metastatic Probability": "{:.2f}"}),
                hide_index=True,
                use_container_width=True
            )
            
            # Add a clickable button to select a row for plotting
            selected_sample = st.selectbox(
                "Select a sample to view plots:",
                options=results_df['Sample Name'],
                index=st.session_state.selected_row_index or 0, # Default to the last selected or first
                key="sample_selector"
            )
            
            # Find the index of the selected sample and update the session state
            if selected_sample:
                selected_idx = results_df[results_df['Sample Name'] == selected_sample].index[0]
                st.session_state.selected_row_index = selected_idx


        with col2:
            selected_idx = st.session_state.selected_row_index
            if selected_idx is not None:
                sample_name = instance.rawdata.loc[selected_idx, "Sample_Name"]
                st.subheader(f"Analysis for: {sample_name}")

                # Generate and display plots from the backend
                with st.spinner('Generating plots...'):
                    waterfall_fig = instance.waterfall_plot(selected_idx)
                    st.pyplot(waterfall_fig)

                    barplot_fig = instance.barplot(selected_idx)
                    st.pyplot(barplot_fig)
            else:
                st.info("Select a sample on the left to display its explanation plots.")
    else:
        # Initial message when the app starts
        st.info("Upload a CSV file and click 'Predict' to begin the analysis.")

# --- DOCUMENTATION TAB ---
with tab2:
    st.header("Documentation")
    
    st.subheader("Model Performance Metrics")
    try:
        st.image(Image.open('model_performance.png'), caption="Model performance comparison.")
    except FileNotFoundError:
        st.warning("model_performance.png not found.")
        
    st.divider()
    
    st.subheader("Feature Importance (SHAP Beeswarm Plot)")
    try:
        st.image(Image.open('beeswarm_plot.png'), caption="SHAP summary plot showing feature importance and impact.")
    except FileNotFoundError:
        st.warning("beeswarm_plot.png not found.")


# --- Footer ---
st.divider()
st.caption("Artificial Neural Network and Lasso-Logistic Regression for Prostate Adenocarcinoma Metastatic Classification and Biomaker Discovery, 2025. Dzakwanil Hakim, Elizabeth Loho")
st.caption("Dzakwanil Hakim (21124026)")
st.caption("dzakwanilhd@gmail.com")
