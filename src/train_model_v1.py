# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.metrics import (accuracy_score, classification_report,
#                              confusion_matrix, roc_auc_score,
#                              r2_score, mean_absolute_error,
#                              mean_squared_error)
# from sklearn.model_selection import GridSearchCV
# import joblib
# import warnings
# warnings.filterwarnings('ignore')


# def train_classifier(Xx_train, yy_train, model_type='random_forest'):
#     """Entraîner un modèle de classification."""
#     if model_type == 'random_forest':
#         model = RandomForestClassifier(
#             class_weight='balanced',
#             n_estimators=100,
#             random_state=42
#         )
#     elif model_type == 'logistic_regression':
#         model = LogisticRegression(
#             class_weight='balanced',
#             max_iter=1000,
#             random_state=42
#         )

#     model.fit(Xx_train, yy_train)
#     print(f"✅ Modèle {model_type} entraîné !")
#     return model


# def optimize_classifier(Xx_train, yy_train):
#     """Optimiser Random Forest avec GridSearchCV."""
#     param_grid = {
#         'n_estimators': [100, 200],
#         'max_depth': [10, 20, None],
#         'min_samples_split': [2, 5]
#     }

#     print("⏳ GridSearchCV en cours (2-3 min)...")
#     grid_search = GridSearchCV(
#         RandomForestClassifier(
#             class_weight='balanced',
#             random_state=42
#         ),
#         param_grid,
#         cv=5,
#         scoring='roc_auc',
#         n_jobs=-1,
#         verbose=1
#     )
#     grid_search.fit(Xx_train, yy_train)

#     print(f"✅ Meilleurs paramètres : {grid_search.best_params_}")
#     print(f"   Meilleur AUC-ROC : {grid_search.best_score_*100:.1f}%")
#     return grid_search.best_estimator_


# def evaluate_classifier(model, Xx_test, yy_test):
#     """Évaluer un modèle de classification."""
#     y_pred = model.predict(Xx_test)

#     print("=== Évaluation Classification ===")
#     print(f"Accuracy : {accuracy_score(yy_test, y_pred)*100:.1f}%")
#     print(f"AUC-ROC  : {roc_auc_score(yy_test, y_pred)*100:.1f}%")
#     print(classification_report(yy_test, y_pred))

#     # Matrice de confusion
#     cm = confusion_matrix(yy_test, y_pred)
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Fidèle', 'Parti'],
#                 yticklabels=['Fidèle', 'Parti'])
#     plt.title('Matrice de Confusion')
#     plt.ylabel('Réel')
#     plt.xlabel('Prédit')
#     plt.tight_layout()
#     plt.show()
#     return y_pred


# def train_regressor(Xx_train, yy_train):
#     """Entraîner un modèle de régression."""
#     model = RandomForestRegressor(
#         n_estimators=100,
#         random_state=42,
#         n_jobs=-1
#     )
#     model.fit(Xx_train, yy_train)
#     print("✅ Modèle de régression entraîné !")
#     return model


# def evaluate_regressor(model, Xx_test, yy_test):
#     """Évaluer un modèle de régression."""
#     y_pred = model.predict(Xx_test)

#     print("=== Évaluation Régression ===")
#     print(f"R²   : {r2_score(yy_test, y_pred):.3f}")
#     print(f"MAE  : {mean_absolute_error(yy_test, y_pred):.1f}£")
#     print(f"RMSE : {np.sqrt(mean_squared_error(yy_test, y_pred)):.1f}£")
#     return y_pred


# def plot_feature_importance(model, feature_names, top_n=15):
#     """Visualiser les features les plus importantes."""
#     importances = pd.DataFrame({
#         'Feature': feature_names,
#         'Importance': model.feature_importances_
#     }).sort_values('Importance', ascending=False).head(top_n)

#     plt.figure(figsize=(10, 6))
#     sns.barplot(data=importances, x='Importance',
#                 y='Feature', palette='viridis')
#     plt.title(f'Top {top_n} Features les plus importantes')
#     plt.tight_layout()
#     plt.show()
#     return importances


# def save_model(model, filepath):
#     """Sauvegarder un modèle."""
#     joblib.dump(model, filepath)
#     print(f"✅ Modèle sauvegardé : {filepath}")


# def load_model(filepath):
#     """Charger un modèle sauvegardé."""
#     model = joblib.load(filepath)
#     print(f"✅ Modèle chargé : {filepath}")
#     return model


import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 📥 Charger dataset
df = pd.read_csv("data/processed/step3_feature_engineering.csv")
df.columns = df.columns.str.lower()
print(df.columns.tolist())
df = df.rename(columns={'totalquantity': 'avgquantitypertransaction'})
# 🎯 Features (les 7 de ton interface)
FEATURES = [
    'age',
    'frequency',
    'monetarytotal',
    'totaltransactions',
    'weekendpurchaseratio',
    'avgquantitypertransaction',
    'recency'
]

TARGET = 'churn'

# ✔ Garder uniquement les colonnes nécessaires
df = df[FEATURES + [TARGET]].dropna()

# ✂️ Split
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 📏 Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 🤖 Modèle
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# 📊 Évaluation
y_pred = model.predict(X_test_scaled)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 💾 Sauvegarde
joblib.dump(model, "models/simple_model_v2.pkl")
joblib.dump(scaler, "models/simple_scaler_v2.pkl")

print("\n✅ Nouveau modèle (7 features) sauvegardé !")
