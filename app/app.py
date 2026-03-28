from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model  = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load feature names from the training data
X_train_ref   = pd.read_csv("data/train_test/X_train.csv")
feature_names = X_train_ref.columns.tolist()

print(f"[OK] Modele attend {len(feature_names)} features")

# Get scaler's expected features
if hasattr(scaler, 'feature_names_in_'):
    scaler_features = scaler.feature_names_in_.tolist()
    print(f"[OK] Scaler attend {len(scaler_features)} features numeriques")
else:
    scaler_features = None
    print("[WARN] Scaler n'a pas de feature_names_in_")


def get_float(form, key, default=0.0):
    """Lire un champ du formulaire — retourne default si vide ou invalide."""
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
        # --- Lire le formulaire ---
        age            = get_float(request.form, 'age',                35.0)
        recency        = get_float(request.form, 'recency',            30.0)
        frequency      = get_float(request.form, 'frequency',           5.0)
        monetary_total = get_float(request.form, 'monetary_total',    500.0)
        total_trans    = get_float(request.form, 'total_transactions', 20.0)
        total_quantity = get_float(request.form, 'total_quantity',    100.0)
        weekend_ratio  = get_float(request.form, 'weekend_ratio',       0.4)

        # --- Créer un DataFrame vide avec UNE ligne à 0 ---
        input_df = pd.DataFrame(0.0, index=[0], columns=feature_names)

        # --- Remplir les colonnes numériques connues ---
        mapping = {
            'Age':               age,
            'Recency':           recency,
            'Frequency':         frequency,
            'MonetaryTotal':     monetary_total,
            'TotalTrans':        total_trans,
            'TotalQuantity':     total_quantity,
            'WeekendRatio':      weekend_ratio,
            'MonetaryPerDay':    monetary_total / (recency + 1),
            'AvgBasketValue':    monetary_total / (frequency + 1),
            'PurchaseIntensity': frequency      / (recency + 1),
            'TenureRatio':       recency        / (recency + 1),
        }

        for col, val in mapping.items():
            if col in input_df.columns:
                input_df[col] = val

        # ✅ Forcer exactement les colonnes du modèle (comble les manquantes à 0)
        input_df = input_df.reindex(columns=feature_names, fill_value=0.0)

        # --- Scaling ---
        # Only scale the numeric columns that the scaler was trained on
        if scaler_features is not None:
            # Find which scaler features exist in our input
            available_scaler_cols = [c for c in scaler_features if c in input_df.columns]
            missing_scaler_cols = [c for c in scaler_features if c not in input_df.columns]

            if missing_scaler_cols:
                # Add missing columns with 0 (median/mean would be better but 0 is safe)
                for col in missing_scaler_cols:
                    input_df[col] = 0.0

            # Now scale all the features the scaler expects
            input_df[scaler_features] = scaler.transform(input_df[scaler_features])
        else:
            # Fallback: scale all numeric columns
            num_cols = input_df.select_dtypes(include=['float64', 'int64']).columns
            input_df[num_cols] = scaler.transform(input_df[num_cols])

        # --- Prédiction ---
        prediction = int(model.predict(input_df)[0])
        proba      = round(float(model.predict_proba(input_df)[0][1]) * 100, 1)

        return render_template('index.html',
                               prediction=prediction,
                               proba=proba,
                               error=None)

    except Exception as e:
        return render_template('index.html',
                               prediction=None,
                               proba=None,
                               error=str(e))


if __name__ == "__main__":
    app.run(debug=True)