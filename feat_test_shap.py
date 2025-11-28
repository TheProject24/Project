import shap
import matplotlib.pyplot as plt
import pandas as pd
from feat_test_modelling import train_and_evaluate_model

def interpret_model_with_shap():
    model, X_test, y_test = train_and_evaluate_model()

    print("\n--- Starting SHAP Interpretation ---")

    # Create a SHAP Explainer
    explainer = shap.LinearExplainer(model, X_test.to_numpy())

    # 3. Calculate SHAP values for the test data
    shap_values = explainer.shap_values(X_test.to_numpy())

    # 4. Visualize the Global Feature Importance (Summary Plot)
    print("Generating SHAP Summary Plot...")
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=X_test.columns,
        show=False  # Prevent immediate display to handle with plt.show()
    )
    plt.title("SHAP Summary Plot: Reduced Features Impact on Pass/Fail Prediction")
    plt.show()


if __name__ == '__main__':
    # Temporarily suppress Pandas setting warnings for clean output
    pd.options.mode.chained_assignment = None
    interpret_model_with_shap()