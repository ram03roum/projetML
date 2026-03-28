"""
Fonctions utilitaires pour le projet ML Retail.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)


def plot_confusion_matrix(y_true, y_pred, labels=['No Churn', 'Churn'], save_path=None):
    """
    Affiche la matrice de confusion.

    Args:
        y_true: Vraies valeurs
        y_pred: Prédictions
        labels: Labels des classes
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Matrice de confusion sauvegardée : {save_path}")

    plt.show()


def plot_roc_curve(y_true, y_proba, save_path=None):
    """
    Affiche la courbe ROC.

    Args:
        y_true: Vraies valeurs
        y_proba: Probabilités prédites pour la classe positive
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Courbe ROC sauvegardée : {save_path}")

    plt.show()


def print_classification_metrics(y_true, y_pred, y_proba=None):
    """
    Affiche un rapport complet des métriques de classification.

    Args:
        y_true: Vraies valeurs
        y_pred: Prédictions
        y_proba: Probabilités (optionnel, pour l'AUC)
    """
    print("\n" + "="*50)
    print("📊 RAPPORT DE CLASSIFICATION")
    print("="*50)

    print("\n" + classification_report(y_true, y_pred,
                                       target_names=['No Churn', 'Churn']))

    if y_proba is not None:
        auc_score = roc_auc_score(y_true, y_proba)
        print(f"\n🎯 ROC AUC Score: {auc_score:.4f}")

    print("="*50 + "\n")


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Affiche l'importance des features pour les modèles arborescents.

    Args:
        model: Modèle entraîné (RandomForest, XGBoost, etc.)
        feature_names: Liste des noms de features
        top_n: Nombre de features à afficher
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    if not hasattr(model, 'feature_importances_'):
        print("⚠️ Ce modèle n'a pas d'attribut feature_importances_")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 8))
    plt.title(f'Top {top_n} Features les Plus Importantes')
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Feature importance sauvegardée : {save_path}")

    plt.show()


def analyze_churn_distribution(y):
    """
    Analyse la distribution des classes de churn.

    Args:
        y: Series ou array des valeurs de churn
    """
    counts = pd.Series(y).value_counts()
    percentages = pd.Series(y).value_counts(normalize=True) * 100

    print("\n" + "="*50)
    print("📊 DISTRIBUTION DES CLASSES")
    print("="*50)
    print(f"No Churn (0): {counts.get(0, 0):,} ({percentages.get(0, 0):.2f}%)")
    print(f"Churn (1):    {counts.get(1, 0):,} ({percentages.get(1, 0):.2f}%)")
    print(f"Total:        {len(y):,}")

    # Calcul du déséquilibre
    if len(counts) == 2:
        imbalance_ratio = counts.max() / counts.min()
        print(f"\n⚖️ Ratio de déséquilibre: {imbalance_ratio:.2f}:1")

        if imbalance_ratio > 3:
            print("⚠️ Classes très déséquilibrées ! Considérez SMOTE ou class_weight.")

    print("="*50 + "\n")


def save_model_report(model_name, metrics, save_path):
    """
    Sauvegarde un rapport de performance du modèle.

    Args:
        model_name: Nom du modèle
        metrics: Dictionnaire des métriques
        save_path: Chemin pour sauvegarder le rapport
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"# Rapport de Performance - {model_name}\n\n")
        f.write("## Métriques\n\n")
        for metric, value in metrics.items():
            f.write(f"- **{metric}**: {value:.4f}\n")
        f.write(f"\n*Généré automatiquement*\n")

    print(f"✅ Rapport sauvegardé : {save_path}")


def detect_data_leakage(X_train, y_train, threshold=0.95):
    """
    Détecte les features potentiellement en fuite de données.

    Args:
        X_train: Features d'entraînement
        y_train: Target
        threshold: Seuil de corrélation pour détecter la fuite

    Returns:
        Liste des colonnes suspectes
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        print("⚠️ Aucune colonne numérique trouvée")
        return []

    df_temp = pd.concat([X_train[numeric_cols], y_train], axis=1)
    correlations = df_temp.corr()[y_train.name].abs().sort_values(ascending=False)

    # Exclure la target elle-même
    correlations = correlations[correlations.index != y_train.name]

    leaky_features = correlations[correlations > threshold].index.tolist()

    if leaky_features:
        print("\n⚠️ ATTENTION: Features suspectes de fuite de données détectées !")
        print("="*50)
        for feat in leaky_features:
            print(f"  - {feat}: corrélation = {correlations[feat]:.4f}")
        print("="*50 + "\n")
    else:
        print("✅ Aucune fuite de données détectée (threshold={threshold})")

    return leaky_features


def compare_models(results_dict):
    """
    Compare les performances de plusieurs modèles.

    Args:
        results_dict: Dictionnaire {model_name: metrics_dict}
    """
    df = pd.DataFrame(results_dict).T

    print("\n" + "="*60)
    print("📊 COMPARAISON DES MODÈLES")
    print("="*60)
    print(df.to_string())
    print("="*60 + "\n")

    # Trouver le meilleur modèle par F1-score
    if 'f1_score' in df.columns:
        best_model = df['f1_score'].idxmax()
        print(f"🏆 Meilleur modèle (F1-score): {best_model}")
        print(f"   F1-score: {df.loc[best_model, 'f1_score']:.4f}\n")

    return df
