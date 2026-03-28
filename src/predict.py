"""
Script de prédiction pour utiliser le modèle entraîné.
"""
import joblib
import pandas as pd
import numpy as np
from preprocessing import fix_outliers, impute_missing


def load_model_and_scaler(model_path='models/best_model.pkl', scaler_path='models/scaler.pkl'):
    """
    Charge le modèle et le scaler sauvegardés.

    Args:
        model_path: Chemin vers le modèle (.pkl)
        scaler_path: Chemin vers le scaler (.pkl)

    Returns:
        model, scaler
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"✅ Modèle chargé depuis {model_path}")
    print(f"✅ Scaler chargé depuis {scaler_path}")
    return model, scaler


def preprocess_new_data(df, scaler, feature_names):
    """
    Prétraite de nouvelles données pour la prédiction.

    Args:
        df: DataFrame avec les nouvelles données
        scaler: StandardScaler déjà fitté
        feature_names: Liste des noms de features attendues par le modèle

    Returns:
        DataFrame prêt pour la prédiction
    """
    # 1. Fix outliers
    df = fix_outliers(df)

    # 2. Imputation
    df = impute_missing(df)

    # 3. Supprimer la colonne target si elle existe
    if 'churn' in df.columns:
        df = df.drop(columns=['churn'])

    # 4. Encodage des catégorielles
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 5. Aligner avec les features du modèle
    df = df.reindex(columns=feature_names, fill_value=0)

    # 6. Scaling
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = scaler.transform(df[num_cols])

    return df


def predict_churn(model, scaler, data_path, feature_names):
    """
    Prédire le churn pour de nouvelles données.

    Args:
        model: Modèle entraîné
        scaler: StandardScaler fitté
        data_path: Chemin vers le fichier CSV de nouvelles données
        feature_names: Liste des features attendues

    Returns:
        DataFrame avec les prédictions
    """
    # Charger les données
    df = pd.read_csv(data_path)
    print(f"📥 Données chargées : {df.shape}")

    # Garder une copie des IDs si présent
    customer_ids = df['customerid'].values if 'customerid' in df.columns else None

    # Prétraiter
    X = preprocess_new_data(df, scaler, feature_names)

    # Prédiction
    predictions = model.predict(X)
    probas = model.predict_proba(X)[:, 1]  # Probabilité de churn

    # Créer le DataFrame de résultats
    results = pd.DataFrame({
        'CustomerID': customer_ids if customer_ids is not None else range(len(predictions)),
        'Churn_Prediction': predictions,
        'Churn_Probability': probas
    })

    return results


if __name__ == "__main__":
    # Exemple d'utilisation
    print("=" * 50)
    print("📊 Script de Prédiction - Churn Client")
    print("=" * 50)

    # Charger le modèle et scaler
    model, scaler = load_model_and_scaler()

    # Charger les feature names depuis X_train
    X_train_ref = pd.read_csv('data/train_test/X_train.csv')
    feature_names = X_train_ref.columns.tolist()
    print(f"✅ {len(feature_names)} features attendues par le modèle")

    # Prédire sur de nouvelles données
    # NOTE: Remplacer 'data/new_customers.csv' par votre fichier de données
    try:
        results = predict_churn(
            model=model,
            scaler=scaler,
            data_path='data/test_new_customers.csv',  # Ajuster ce chemin
            feature_names=feature_names
        )

        print("\n📈 Résultats des prédictions :")
        print(results.head(10))

        # Sauvegarder les résultats
        output_path = 'reports/predictions.csv'
        results.to_csv(output_path, index=False)
        print(f"\n✅ Prédictions sauvegardées dans {output_path}")

    except FileNotFoundError:
        print("\n⚠️ Fichier de test non trouvé.")
        print("Créez un fichier 'data/test_new_customers.csv' avec les mêmes colonnes que les données d'entraînement.")
