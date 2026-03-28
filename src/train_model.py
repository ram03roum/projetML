# import joblib # Pour sauvegarder le modèle
# import numpy as np
# import pandas as pd

# from sklearn.cluster import KMeans
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     r2_score,
#     mean_absolute_error,
#     mean_squared_error,
# )
# from sklearn.model_selection import train_test_split, cross_val_score
# from preprocessing import preprocess_pipeline # pour prétraiter les données


# def evaluate_regression(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     return {
#         "r2": r2_score(y_test, y_pred),
#         "mae": mean_absolute_error(y_test, y_pred),
#         "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
#     }


# def run_kmeans(X, n_clusters=4):
#     print(f"\n🔎 Lancement de KMeans avec {n_clusters} clusters...")
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#     kmeans.fit(X)
#     labels = kmeans.labels_
#     counts = np.bincount(labels)
#     print(f"  Inertie : {kmeans.inertia_:.2f}")
#     print(f"  Répartition des clusters : {dict(enumerate(counts))}")
#     return kmeans


# def drop_monetary_leaks(df, target_col='MonetaryTotal'):
#     derived = [col for col in df.columns if col != target_col and col.lower().startswith('monetary')]
#     if derived:
#         print(f"  Colonnes dérivées de {target_col} supprimées pour la régression : {derived}")
#         return df.drop(columns=derived)
#     return df


# def load_regression_data(path, target_col='MonetaryTotal'):
#     df = pd.read_csv(path)
#     print(f"  Régression : shape initiale {df.shape}")
#     df = drop_monetary_leaks(df, target_col=target_col)
#     return df


# X_train, X_test, y_train, y_test, X_train_smote, y_train_smote, scaler = preprocess_pipeline(
#     '../data/processed/step3_feature_engineering.csv'
# )

# # Vérifications pipeline
# print("Train shape:", X_train_smote.shape)
# print("Test shape:", X_test.shape)

# print("Classes distribution (train):", np.bincount(y_train_smote))
# print("Classes distribution (test):", np.bincount(y_test))

# print("Baseline accuracy (majority class):", max(y_test.value_counts(normalize=True)))

# print(X_train.columns)

# def evaluate_model(model, X_test, y_test):
#     """
#     Évalue un modèle avec plusieurs métriques

#     """
#     y_pred = model.predict(X_test) # prédit le churn pour ces clients
#     return {
#         "accuracy": accuracy_score(y_test, y_pred),# % de bonnes prédictions
#         "precision": precision_score(y_test, y_pred), # parmi ceux prédits “churn”, combien sont vrais ?
#         "recall": recall_score(y_test, y_pred), # parmi ceux qui sont vraiment “churn”, combien sont détectés ?
#         "f1_score": f1_score(y_test, y_pred) # equilibre entre precision et recall
#     }


# def build_regression_dataset(path, target_col='MonetaryTotal'):
#     df = load_regression_data(path, target_col)
#     X = df.drop(columns=[target_col])
#     y = df[target_col]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     X_train = pd.get_dummies(X_train, drop_first=True)
#     X_test = pd.get_dummies(X_test, drop_first=True)
#     X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

#     return X_train, X_test, y_train, y_test


# # ENTRAÎNEMENT DES MODÈLES

# # 🔹 Modèle 1 : Logistic Regression

# # creation d'un modèle de régression logistique avec 
# # un nombre d'itérations élevé pour assurer la convergence
# lr = LogisticRegression(max_iter=1000) 
# lr.fit(X_train_smote, y_train_smote) 

# print("Logistic Regression entraîné ✔")


# # 🔹 Modèle 2 : Random Forest
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train_smote, y_train_smote)

# print("Random Forest entraîné ✔")

# scores = cross_val_score(rf, X_train_smote, y_train_smote, cv=5, scoring='f1')
# print("Cross-validation F1 scores:", scores)
# print("Mean CV F1:", scores.mean()) 

# print("\n🔍 Test sans SMOTE (sur train original)")

# rf.fit(X_train, y_train)
# evaluate_model(rf, X_test, y_test)



# # ÉVALUATION DES MODÈLES
# print("Évaluation de la Logistic Regression :")
# lr_results = evaluate_model(lr, X_test, y_test)
# rf_results = evaluate_model(rf, X_test, y_test)

# print("\nRésultats Logistic Regression :", lr_results)
# print("Résultats Random Forest :", rf_results)

# # CHOIX DU MEILLEUR MODÈLE
# if rf_results['f1_score'] > lr_results['f1_score']:
#     best_model = rf
#     best_name = "Random Forest"
# else:
#     best_model = lr
#     best_name = "Logistic Regression"

# print(f"\nMeilleur modèle : {best_name} ✔")

# from sklearn.metrics import confusion_matrix

# y_pred = rf.predict(X_test)
# print(confusion_matrix(y_test, y_pred))


# # ─────────────────────────────────────────
# #  Régression MonetaryTotal
# # ─────────────────────────────────────────
# print("\n🔧 Lancement de la régression sur MonetaryTotal")
# X_reg_train, X_reg_test, y_reg_train, y_reg_test = build_regression_dataset(
#     '../data/processed/step3_feature_engineering.csv',
#     target_col='MonetaryTotal'
# )

# lr_reg = LinearRegression()
# lr_reg.fit(X_reg_train, y_reg_train)
# print("Linear Regression entraîné pour la régression ✔")

# rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_reg.fit(X_reg_train, y_reg_train)
# print("Random Forest Regressor entraîné ✔")

# lr_reg_results = evaluate_regression(lr_reg, X_reg_test, y_reg_test)
# rf_reg_results = evaluate_regression(rf_reg, X_reg_test, y_reg_test)

# print("\nRésultats Linear Regression :", lr_reg_results)
# print("Résultats Random Forest Regressor :", rf_reg_results)


# # ─────────────────────────────────────────
# #  Clustering KMeans
# # ─────────────────────────────────────────
# run_kmeans(X_train, n_clusters=4)


# # SAUVEGARDE DU MEILLEUR MODÈLE
# # Sauvegarder modèle
# joblib.dump(best_model, '../models/best_model.pkl')

# # Sauvegarder scaler
# joblib.dump(scaler, '../models/scaler.pkl')

# print("Modèle et scaler sauvegardés ✔")



import joblib # Pour sauvegarder le modèle
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split, cross_val_score
from preprocessing import preprocess_pipeline # pour prétraiter les données


def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }


def run_kmeans(X, n_clusters=4):
    print(f"\n🔎 Lancement de KMeans avec {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    print(f"  Inertie : {kmeans.inertia_:.2f}")
    print(f"  Répartition des clusters : {dict(enumerate(counts))}")
    return kmeans


def drop_monetary_leaks(df, target_col='MonetaryTotal'):
    derived = [col for col in df.columns if col != target_col and col.lower().startswith('monetary')]
    if derived:
        print(f"  Colonnes dérivées de {target_col} supprimées pour la régression : {derived}")
        return df.drop(columns=derived)
    return df


def load_regression_data(path, target_col='MonetaryTotal'):
    df = pd.read_csv(path)
    print(f"  Régression : shape initiale {df.shape}")
    df = drop_monetary_leaks(df, target_col=target_col)
    return df


X_train, X_test, y_train, y_test, X_train_smote, y_train_smote, scaler = preprocess_pipeline(
    '../data/processed/step3_feature_engineering.csv'
)

# Vérifications pipeline
print("Train shape:", X_train_smote.shape)
print("Test shape:", X_test.shape)

print("Classes distribution (train):", np.bincount(y_train_smote))
print("Classes distribution (test):", np.bincount(y_test))

print("Baseline accuracy (majority class):", max(y_test.value_counts(normalize=True)))

print(X_train.columns)

def evaluate_model(model, X_test, y_test):
    """
    Évalue un modèle avec plusieurs métriques

    """
    y_pred = model.predict(X_test) # prédit le churn pour ces clients
    return {
        "accuracy": accuracy_score(y_test, y_pred),# % de bonnes prédictions
        "precision": precision_score(y_test, y_pred), # parmi ceux prédits “churn”, combien sont vrais ?
        "recall": recall_score(y_test, y_pred), # parmi ceux qui sont vraiment “churn”, combien sont détectés ?
        "f1_score": f1_score(y_test, y_pred) # equilibre entre precision et recall
    }


def build_regression_dataset(path, target_col='MonetaryTotal'):
    df = load_regression_data(path, target_col)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    return X_train, X_test, y_train, y_test


# ENTRAÎNEMENT DES MODÈLES

# 🔹 Modèle 1 : Logistic Regression

# creation d'un modèle de régression logistique avec 
# un nombre d'itérations élevé pour assurer la convergence
lr = LogisticRegression(max_iter=1000) 
lr.fit(X_train_smote, y_train_smote) 

print("Logistic Regression entraîné ✔")


# 🔹 Modèle 2 : Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_smote, y_train_smote)

print("Random Forest entraîné ✔")

scores = cross_val_score(rf, X_train_smote, y_train_smote, cv=5, scoring='f1')
print("Cross-validation F1 scores:", scores)
print("Mean CV F1:", scores.mean()) 

print("\n🔍 Test sans SMOTE (sur train original)")

rf.fit(X_train, y_train)
evaluate_model(rf, X_test, y_test)



# ÉVALUATION DES MODÈLES
print("Évaluation de la Logistic Regression :")
lr_results = evaluate_model(lr, X_test, y_test)
rf_results = evaluate_model(rf, X_test, y_test)

print("\nRésultats Logistic Regression :", lr_results)
print("Résultats Random Forest :", rf_results)

# CHOIX DU MEILLEUR MODÈLE
if rf_results['f1_score'] > lr_results['f1_score']:
    best_model = rf
    best_name = "Random Forest"
else:
    best_model = lr
    best_name = "Logistic Regression"

print(f"\nMeilleur modèle : {best_name} ✔")

from sklearn.metrics import confusion_matrix

y_pred = rf.predict(X_test)
print(confusion_matrix(y_test, y_pred))


# ─────────────────────────────────────────
#  Régression MonetaryTotal
# ─────────────────────────────────────────
print("\n🔧 Lancement de la régression sur MonetaryTotal")
X_reg_train, X_reg_test, y_reg_train, y_reg_test = build_regression_dataset(
    '../data/processed/step3_feature_engineering.csv',
    target_col='MonetaryTotal'
)

lr_reg = LinearRegression()
lr_reg.fit(X_reg_train, y_reg_train)
print("Linear Regression entraîné pour la régression ✔")

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_reg_train, y_reg_train)
print("Random Forest Regressor entraîné ✔")

lr_reg_results = evaluate_regression(lr_reg, X_reg_test, y_reg_test)
rf_reg_results = evaluate_regression(rf_reg, X_reg_test, y_reg_test)

print("\nRésultats Linear Regression :", lr_reg_results)
print("Résultats Random Forest Regressor :", rf_reg_results)


# ─────────────────────────────────────────
#  Clustering KMeans
# ─────────────────────────────────────────
run_kmeans(X_train, n_clusters=4)


# SAUVEGARDE DU MEILLEUR MODÈLE
# Sauvegarder modèle
joblib.dump(best_model, '../models/best_model.pkl')

# Sauvegarder scaler
joblib.dump(scaler, '../models/scaler.pkl')

print("Modèle et scaler sauvegardés ✔")