import pandas as pd
import numpy as np
import shap
#import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#import tensorflow as tf

class Prediction:
    """
    Handles the core logic for making predictions and generating SHAP explanations.
    This class is designed to be imported by a user interface like Streamlit.
    """
    # The order of features the model expects.
    features_order = ["MYH11", "PRMT1", "TENT4A"]

    def __init__(self, input_df, scaler, x_background, model):
        """
        Initializes the Prediction object.

        Args:
            input_df (pd.DataFrame): The dataframe containing the input data.
            scaler: A fitted scikit-learn scaler object.
            x_background (np.array): The background dataset for the SHAP explainer.
            model: The trained prediction model (scikit-learn or Keras).
        """
        self.scaler = scaler
        self.model = model
        self.X_background = x_background

        # Validate and use the uploaded dataframe
        self.rawdata = input_df
        missing_cols = [col for col in ['Sample_Name'] + self.features_order if col not in self.rawdata.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in input file: {missing_cols}")

        self.features_data = self.rawdata[self.features_order]
        
        # Dictionaries to store results for each sample.
        self.predictions = {}
        self.explanations = {}
        
        # Scale features upon initialization.
        self._feature_scaling()

    def _feature_scaling(self):
        """Private method to scale the feature data."""
        self.scaled_data = pd.DataFrame(self.scaler.transform(self.features_data), columns=self.features_order)

    @staticmethod
    def _make_predict_fn(model):
        """
        Creates a unified prediction function that returns the probability of the positive class,
        handling both Keras and scikit-learn models.
        """
        # Check if the model has the 'predict_proba' method (scikit-learn standard)
        if hasattr(model, 'predict_proba'):
            return lambda x: model.predict_proba(x)[:, 1]
        # Check if the model is a Keras model
        elif hasattr(model, 'predict'):
             return lambda x: model.predict(x, verbose=0).flatten()
        else:
            raise TypeError("Model type not supported. Must have a 'predict' or 'predict_proba' method.")


    def run_analysis(self):
        """
        Runs the full analysis pipeline: predictions and SHAP value calculations.
        """
        predict_fn = self._make_predict_fn(self.model)
        all_probas = predict_fn(self.scaled_data)

        for idx, proba in enumerate(all_probas):
            self.predictions[idx] = {
                "probability": proba,
                "label": "Metastatic" if proba > 0.5 else "Non-metastatic"
            }

        # Initialize and run the SHAP explainer
        explainer = shap.KernelExplainer(predict_fn, shap.kmeans(self.X_background, 10))
        shap_values = explainer.shap_values(self.scaled_data, nsamples=100)

        # Store a SHAP Explanation object for each sample for easy plotting
        for idx in self.scaled_data.index:
            self.explanations[idx] = shap.Explanation(
                values=shap_values[idx],
                base_values=explainer.expected_value,
                data=self.scaled_data.iloc[idx].values,
                feature_names=self.features_order
            )

    def waterfall_plot(self, idx):
        """
        Generates a SHAP waterfall plot for a given sample index.
        Returns a matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(self.explanations[idx], show=False)
        ax.set_title(f"SHAP Feature Contribution Plot")
        plt.tight_layout()
        return fig

    def barplot(self, idx):
        """
        Generates a bar plot comparing sample expression to group averages.
        Returns a matplotlib Figure object.
        """
        # Hardcoded distributions for demonstration. In a real app, this could be configurable.
        mt_dist = {"MYH11": (9.95, 1.09), "PRMT1": (9.36, 0.69), "TENT4A": (10.50, 0.47)}
        non_mt_dist = {"MYH11": (15.36, 1.51), "PRMT1": (11.25, 0.44), "TENT4A": (9.02, 0.33)}
        
        genes = self.features_order
        x = np.arange(len(genes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot group averages
        mt_means = [mt_dist[g][0] for g in genes]
        mt_stds = [mt_dist[g][1] for g in genes]
        ax.bar(x - width, mt_means, width, yerr=mt_stds, label="Metastatic Avg.", color='red', alpha=0.6, capsize=5)

        notmt_means = [non_mt_dist[g][0] for g in genes]
        notmt_stds = [non_mt_dist[g][1] for g in genes]
        ax.bar(x, notmt_means, width, yerr=notmt_stds, label="Non-metastatic Avg.", color='blue', alpha=0.6, capsize=5)

        # Plot the current sample's data
        sample_vals = self.rawdata.loc[idx, genes].values
        sample_name = self.rawdata.loc[idx, 'Sample_Name']
        ax.bar(x + width, sample_vals, width, label=f"Sample: {sample_name}", color='green')

        ax.set_xticks(x)
        ax.set_xticklabels(genes)
        ax.set_ylabel("Expression Level")
        ax.set_title(f"Biomarker Expression Comparison")
        ax.legend()
        plt.tight_layout()
        return fig
