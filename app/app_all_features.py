from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model  = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load TRAINING data to get median values (CRITICAL!)
X_train_ref = pd.read_csv("data/train_test/X_train.csv")
feature_names = X_train_ref.columns.tolist()

# BEFORE scaling - get raw training data to compute medians
X_train_raw = pd.read_csv("data/processed/step3_feature_engineering.csv")

print(f"[OK] Modele attend {len(feature_names)} features")
print(f"[OK] Using MEDIAN values from training data for missing features")

# Get median values for numeric features (unscaled)
feature_medians = {}
for col in feature_names:
    if col in X_train_raw.columns:
        feature_medians[col] = X_train_raw[col].median()
    else:
        # For one-hot encoded features, default is 0
        feature_medians[col] = 0.0

print(f"[OK] Computed {len(feature_medians)} feature medians")


def get_float(form, key, default=None):
    """Read form field, return default if empty/invalid."""
    val = form.get(key, '').strip()
    try:
        return float(val) if val != '' else default
    except ValueError:
        return default


@app.route('/')
def home():
    return render_template('index.html', prediction=None, proba=None, error=None)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # --- Read form inputs ---
        age            = get_float(request.form, 'age')
        frequency      = get_float(request.form, 'frequency')
        monetary_total = get_float(request.form, 'monetary_total')
        total_trans    = get_float(request.form, 'total_transactions')
        total_quantity = get_float(request.form, 'total_quantity')
        weekend_ratio  = get_float(request.form, 'weekend_ratio')

        # CRITICAL: Start with MEDIAN values, not zeros!
        input_df = pd.DataFrame([feature_medians])

        # Now override with user inputs (only if provided)
        user_inputs = {
            'age': age,
            'frequency': frequency,
            'monetarytotal': monetary_total,
            'totaltransactions': total_trans,
            # Note: 'totalquantity' doesn't exist in processed data - ignoring it
            'weekendpurchaseratio': weekend_ratio,
        }

        print(f"\n[DEBUG] User inputs: {user_inputs}")

        # Fill in user values and track changes
        changes_made = []
        for col, val in user_inputs.items():
            if val is not None:
                if col in input_df.columns:
                    input_df.at[0, col] = val  # Use .at for single value assignment
                    changes_made.append(f"{col}={val}")
                else:
                    print(f"[WARN] Column '{col}' not found in features")

        print(f"[DEBUG] Changes made: {changes_made}")

        # Compute derived features IF we have the base features
        if frequency is not None and monetary_total is not None and frequency > 0:
            if 'avgbasketvalue' in input_df.columns:
                new_val = monetary_total / max(frequency, 1)
                input_df.at[0, 'avgbasketvalue'] = new_val
                changes_made.append(f"avgbasketvalue={new_val:.2f}")

        print(f"[DEBUG] Total changes: {len(changes_made)}")
        print(f"[DEBUG] Sample feature values - freq:{input_df['frequency'].values[0]:.2f}, monetary:{input_df['monetarytotal'].values[0]:.2f}")

        # Ensure exact column order
        input_df = input_df[feature_names]

        # --- Scaling ---
        # The scaler expects UNSCALED numeric features
        # We need to inverse what was scaled during training
        scaler_features = scaler.feature_names_in_.tolist()

        # Scale only the numeric columns
        input_df[scaler_features] = scaler.transform(input_df[scaler_features])

        # --- Prediction ---
        prediction = int(model.predict(input_df)[0])
        proba      = round(float(model.predict_proba(input_df)[0][1]) * 100, 1)

        print(f"[RESULT] Prediction: {prediction}, Churn prob: {proba}%")

        return render_template('index.html',
                               prediction=prediction,
                               proba=proba,
                               error=None)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('index.html',
                               prediction=None,
                               proba=None,
                               error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
