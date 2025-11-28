import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split_data(file_path, selected_features=None):

    df = pd.read_csv(file_path, sep=";")

    df['pass_fail'] = df['G3'].apply(lambda x: 1 if x > 10 else 0)

    df_encoded = pd.get_dummies(df, drop_first=True, dtype=int)

    drop_cols = ['G1', 'G2', 'G3', 'pass_fail']

    x_full = df_encoded.drop(columns=drop_cols)
    y = df_encoded['pass_fail']

    # --- FEATURE SELECTION STEP ---
    if selected_features:
        x = x_full[selected_features]
        print(f"--- Training with Reduced Feature Set ({len(selected_features)} features) ---")
    else:
        x = x_full
        print(f"--- Training with Full Feature Set ({x.shape[1]} features) ---")
    # -----------------------------

    # Train/Test Split (80% Train, 20% Test)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        random_state=42
    )

    return x_train, x_test, y_train, y_test