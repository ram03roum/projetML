"""
Script d'entra nement complet - Version Corrig e
Bas  sur le rapport de r f rence du professeur
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE

from preprocessing import preprocess_pipeline


def save_confusion_matrix(y_true, y_pred, filename='reports/confusion_matrix.png'):
    """Sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn (0)', 'Churn (1)'],
                yticklabels=['No Churn (0)', 'Churn (1)'])
    plt.title('Matrice de Confusion - Random Forest')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Pr dite')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[OK] Matrice de confusion sauvegard e: {filename}")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=15, filename='reports/feature_importance.png'):
    """Visualise l'importance des features."""
    if not hasattr(model, 'feature_importances_'):
        print("[WARN] Ce mod le n'a pas d'attribut feature_importances_")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 8))
    plt.title(f'Top {top_n} Features les Plus Importantes')
    plt.barh(range(top_n), importances[indices], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[OK] Feature importance sauvegard e: {filename}")
    plt.close()


def plot_clusters_pca(X, labels, filename='reports/clusters_pca.png'):
    """Visualise les clusters apr s r duction PCA."""
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.title('Clusters K-Means apr s ACP')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[OK] Visualisation clusters sauvegard e: {filename}")
    plt.close()


def evaluate_model(model, X_test, y_test):
    """ value un mod le de classification."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0)
    }


def evaluate_regression(model, X_test, y_test):
    """ value un mod le de r gression."""
    y_pred = model.predict(X_test)
    return {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }


def run_clustering(X, n_clusters=4):
    """Ex cute K-Means clustering."""
    print(f"\n{'='*60}")
    print(f"[CLUSTERING] CLUSTERING - K-Means avec {n_clusters} clusters")
    print(f"{'='*60}")

    # PCA pour r duction de dimension
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"  Variance expliqu e (10 composantes): {pca.explained_variance_ratio_.sum()*100:.2f}%")

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    # Statistiques
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n  Inertie: {kmeans.inertia_:.2f}")
    print(f"  R partition des clusters:")
    for cluster_id, count in zip(unique, counts):
        print(f"    - Cluster {cluster_id}: {count} clients ({count/len(labels)*100:.1f}%)")

    # Visualisation
    plot_clusters_pca(X, labels)

    # Sauvegarder le mod le
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    joblib.dump(pca, 'models/pca_model.pkl')
    print(f"\n[OK] Mod les clustering sauvegard s")

    return kmeans, labels


def run_classification(X_train, X_test, y_train, y_test, X_train_smote, y_train_smote):
    """Ex cute la classification avec optimisation."""
    print(f"\n{'='*60}")
    print("[CLASSIFICATION] CLASSIFICATION - Pr diction du Churn")
    print(f"{'='*60}")

    # V rifier le d s quilibre
    print(f"\n  Distribution des classes (test):")
    unique, counts = np.unique(y_test, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"    - Classe {val}: {count} ({count/len(y_test)*100:.1f}%)")

    # Mod le 1: Logistic Regression
    print(f"\n  [RESULTS] Entra nement Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_smote, y_train_smote)
    lr_results = evaluate_model(lr, X_test, y_test)
    print(f"     Accuracy: {lr_results['accuracy']:.4f} | F1: {lr_results['f1_score']:.4f}")

    # Mod le 2: Random Forest (sans SMOTE d'abord)
    print(f"\n  [RF] Entra nement Random Forest (sans SMOTE)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    rf_results = evaluate_model(rf, X_test, y_test)
    print(f"     Accuracy: {rf_results['accuracy']:.4f} | F1: {rf_results['f1_score']:.4f}")

    # Mod le 3: Random Forest avec GridSearchCV
    print(f"\n  [GRIDSEARCH] Optimisation avec GridSearchCV...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    print(f"     Meilleurs param tres: {grid_search.best_params_}")
    print(f"     Meilleur score CV: {grid_search.best_score_:.4f}")

    best_rf_results = evaluate_model(best_rf, X_test, y_test)
    print(f"     Test Accuracy: {best_rf_results['accuracy']:.4f} | F1: {best_rf_results['f1_score']:.4f}")

    # Comparaison des r sultats
    print(f"\n  [RESULTS] COMPARAISON DES MOD LES:")
    print(f"  {'-'*58}")
    print(f"  {'Mod le':<30} {'Accuracy':>12} {'F1-Score':>12}")
    print(f"  {'-'*58}")
    print(f"  {'Logistic Regression':<30} {lr_results['accuracy']:>12.4f} {lr_results['f1_score']:>12.4f}")
    print(f"  {'Random Forest':<30} {rf_results['accuracy']:>12.4f} {rf_results['f1_score']:>12.4f}")
    print(f"  {'Random Forest (GridSearch)':<30} {best_rf_results['accuracy']:>12.4f} {best_rf_results['f1_score']:>12.4f}")
    print(f"  {'-'*58}")

    # Choisir le meilleur
    if best_rf_results['f1_score'] >= rf_results['f1_score']:
        best_model = best_rf
        best_name = "Random Forest (GridSearch)"
        best_results = best_rf_results
    else:
        best_model = rf
        best_name = "Random Forest"
        best_results = rf_results

    print(f"\n  [BEST] Meilleur mod le: {best_name}")
    print(f"     Accuracy: {best_results['accuracy']*100:.2f}%")

    # Matrice de confusion
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  [RESULTS] Matrice de Confusion:")
    print(f"     {cm}")

    # Sauvegarder visualisations
    save_confusion_matrix(y_test, y_pred)
    plot_feature_importance(best_model, X_train.columns)

    return best_model, best_name, best_results


def run_regression(data_path):
    """Ex cute la r gression pour pr dire MonetaryTotal."""
    print(f"\n{'='*60}")
    print("[REGRESSION] R GRESSION - Pr diction des Revenus (MonetaryTotal)")
    print(f"{'='*60}")

    # Charger et pr parer les donn es
    df = pd.read_csv(data_path)

    # Supprimer les colonnes d riv es de MonetaryTotal (Data Leakage!)
    leak_cols = ['monetaryavg', 'monetarystd', 'monetaryperday', 'monetaryperday_log']
    df = df.drop(columns=[c for c in leak_cols if c in df.columns], errors='ignore')

    # Filtrer les outliers extr mes
    df = df[(df['monetarytotal'] > 0) & (df['monetarytotal'] < 20000)]
    print(f"  Donn es apr s filtrage: {df.shape}")

    # S lectionner features
    features_reg = ['frequency', 'totalquantity', 'uniqueproducts', 'age',
                    'avgdaysbetweenpurchases', 'supportticketscount',
                    'satisfactionscore', 'avgquantitypertransaction']

    features_reg = [f for f in features_reg if f in df.columns]

    X = df[features_reg].fillna(0)
    y = df['monetarytotal']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"  Features utilis es: {len(features_reg)}")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Mod les
    lr_reg = LinearRegression()
    lr_reg.fit(X_train, y_train)
    lr_reg_results = evaluate_regression(lr_reg, X_test, y_test)

    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train, y_train)
    rf_reg_results = evaluate_regression(rf_reg, X_test, y_test)

    # R sultats
    print(f"\n  [RESULTS] R SULTATS:")
    print(f"  {'-'*58}")
    print(f"  {'Mod le':<30} {'R ':>10} {'MAE':>15}")
    print(f"  {'-'*58}")
    print(f"  {'Linear Regression':<30} {lr_reg_results['r2']:>10.4f} {lr_reg_results['mae']:>15.2f} ")
    print(f"  {'Random Forest':<30} {rf_reg_results['r2']:>10.4f} {rf_reg_results['mae']:>15.2f} ")
    print(f"  {'-'*58}")

    best_reg_model = rf_reg if rf_reg_results['r2'] > lr_reg_results['r2'] else lr_reg
    joblib.dump(best_reg_model, 'models/regression_model.pkl')
    print(f"\n[OK] Mod le de r gression sauvegard ")

    return best_reg_model, rf_reg_results


# ===========================================================
# MAIN PIPELINE
# ===========================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("[PIPELINE] PIPELINE MACHINE LEARNING COMPLET")
    print("="*60)

    # Cr er dossiers si n cessaire
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # -----------------------------------------
    # 1. PR TRAITEMENT
    # -----------------------------------------
    print("\n[ETAPE]  TAPE 1: PR TRAITEMENT DES DONN ES")
    print("-"*60)

    X_train, X_test, y_train, y_test, X_train_smote, y_train_smote, scaler = preprocess_pipeline(
        'data/processed/step3_feature_engineering.csv'
    )

    # Save the preprocessed data for Flask app
    Path("data/train_test").mkdir(exist_ok=True, parents=True)
    X_train.to_csv('data/train_test/X_train.csv', index=False)
    X_test.to_csv('data/train_test/X_test.csv', index=False)
    y_train.to_csv('data/train_test/y_train.csv', index=False)
    y_test.to_csv('data/train_test/y_test.csv', index=False)
    print(f"\n[OK] Donnees train/test sauvegardees ({X_train.shape[1]} features)")

    # -----------------------------------------
    # 2. CLUSTERING
    # -----------------------------------------
    print("\n[ETAPE]  TAPE 2: CLUSTERING")
    print("-"*60)

    kmeans, labels = run_clustering(X_train, n_clusters=4)

    # -----------------------------------------
    # 3. CLASSIFICATION
    # -----------------------------------------
    print("\n[ETAPE]  TAPE 3: CLASSIFICATION")
    print("-"*60)

    best_model, best_name, results = run_classification(
        X_train, X_test, y_train, y_test,
        X_train_smote, y_train_smote
    )

    # Sauvegarder le meilleur mod le
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print(f"\n[OK] Mod le final et scaler sauvegard s")

    # -----------------------------------------
    # 4. R GRESSION
    # -----------------------------------------
    print("\n[ETAPE]  TAPE 4: R GRESSION")
    print("-"*60)

    reg_model, reg_results = run_regression('data/processed/step3_feature_engineering.csv')

    # -----------------------------------------
    # 5. R SUM  FINAL
    # -----------------------------------------
    print("\n" + "="*60)
    print("[OK] PIPELINE TERMIN  - R SUM  DES R SULTATS")
    print("="*60)

    print(f"\n[RESULTS] CLASSIFICATION (Churn):")
    print(f"   Meilleur mod le: {best_name}")
    print(f"   Accuracy: {results['accuracy']*100:.2f}%")
    print(f"   F1-Score: {results['f1_score']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall: {results['recall']:.4f}")

    print(f"\n[REGRESSION] R GRESSION (MonetaryTotal):")
    print(f"   R : {reg_results['r2']:.4f} ({reg_results['r2']*100:.1f}% variance expliqu e)")
    print(f"   MAE: {reg_results['mae']:.2f} ")
    print(f"   RMSE: {reg_results['rmse']:.2f} ")

    print(f"\n[CLUSTERING] CLUSTERING:")
    print(f"   4 segments clients identifi s")

    print(f"\n  Fichiers g n r s:")
    print(f"   - models/best_model.pkl")
    print(f"   - models/scaler.pkl")
    print(f"   - models/regression_model.pkl")
    print(f"   - models/kmeans_model.pkl")
    print(f"   - reports/confusion_matrix.png")
    print(f"   - reports/feature_importance.png")
    print(f"   - reports/clusters_pca.png")

    print("\n" + "="*60)
    print("[SUCCESS] Projet termin  avec succ s!")
    print("="*60 + "\n")
