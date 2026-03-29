from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load SIMPLE model (only 5 features!)
model  = joblib.load("models/simple_model.pkl")
scaler = joblib.load("models/simple_scaler.pkl")

# Features the simple model expects
FEATURE_NAMES = ['age', 'frequency', 'monetarytotal', 'totaltransactions', 'weekendpurchaseratio']

# Load training data to get median values for missing fields
X_train_raw = pd.read_csv("data/processed/step3_feature_engineering.csv")
feature_medians = {col: X_train_raw[col].median() for col in FEATURE_NAMES if col in X_train_raw.columns}

print(f"[OK] Simple model loaded ({len(FEATURE_NAMES)} features)")
print(f"[OK] Feature medians computed")


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
        # Read form inputs
        age            = get_float(request.form, 'age')
        frequency      = get_float(request.form, 'frequency')
        monetary_total = get_float(request.form, 'monetary_total')
        total_trans    = get_float(request.form, 'total_transactions')
        weekend_ratio  = get_float(request.form, 'weekend_ratio')

        # Build input dictionary
        user_inputs = {
            'age': age,
            'frequency': frequency,
            'monetarytotal': monetary_total,
            'totaltransactions': total_trans,
            'weekendpurchaseratio': weekend_ratio,
        }

        print(f"\n[DEBUG] User inputs: {user_inputs}")

        # Create dataframe with user inputs, filling missing with median
        input_data = {}
        for feat in FEATURE_NAMES:
            if user_inputs.get(feat) is not None:
                input_data[feat] = user_inputs[feat]
            else:
                input_data[feat] = feature_medians.get(feat, 0.0)
                print(f"[DEBUG] Using median for {feat}: {input_data[feat]:.2f}")

        input_df = pd.DataFrame([input_data])

        print(f"[DEBUG] Final feature values:")
        for feat, val in input_data.items():
            print(f"  {feat}: {val:.2f}")

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = int(model.predict(input_scaled)[0])
        proba      = round(float(model.predict_proba(input_scaled)[0][1]) * 100, 1)

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
