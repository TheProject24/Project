from data_prep import load_and_split_data
from sklearn.linear_model import LogisticRegression
import joblib

def train_and_evaluate_model():
    data_file = "student-mat.csv"

    # Load the data
    X_train, X_test, y_train, y_test = load_and_split_data(data_file)

    print("--- Model Training & Evaluation ---")
    print(f"Training on {X_train.shape[0]} samples.")

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
    joblib.dump(model, 'student_logistic_model.pkl')
    joblib.dump(X_train.columns, 'model_features.pkl')

    # Return the trained model and the test sets
    return model, X_test, y_test


if __name__ == '__main__':
    model, X_test, y_test = train_and_evaluate_model()
