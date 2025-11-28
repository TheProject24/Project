# Import the function from your data_prep.py file
from data_prep import load_and_split_data
from sklearn.linear_model import LogisticRegression
import joblib



def train_and_evaluate_model():
    data_file = "student-mat.csv"

    # The top 10 features based on the SHAP plot)
    top_10_features = [
        'failures', 'Mjob_services', 'goout', 'sex_M', 'famsup_yes',
        'Walc', 'guardian_mother', 'nursery_yes', 'age', 'Mjob_health'
    ]

    # Load the data, passing the selected_features list
    X_train, X_test, y_train, y_test = load_and_split_data(data_file, selected_features=top_10_features)

    print("--- Model Training & Evaluation (Reduced Features) ---")
    print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features.")

    # Train the Model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # Model Evaluation
    accuracy = model.score(X_test, y_test)
    print(f"Logistic Regression Model Training Complete!")
    print(f"Model Accuracy on Test Set: {accuracy:.4f}")

    print("--- --- --- --- ---")
    print("--- --- --- --- ---")
    print("--- --- --- --- ---")
    print("--- --- --- --- ---")
    print("--- --- --- --- ---")

    print("--- Saving Model and Columns for  Streamlit hosting ---")
    joblib.dump(model, 'reduced_student_logistic_model.pkl')
    joblib.dump(X_train.columns, 'reduced_model_features.pkl')

    # Return the trained model and the test sets
    return model, X_test, y_test


if __name__ == '__main__':
    model, X_test, y_test = train_and_evaluate_model()