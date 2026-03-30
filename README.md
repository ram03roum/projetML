# Projet Machine Learning
🧠 Analyse du Comportement Client – Prédiction du Churn en E-commerce

📕 Description

Ce projet vise à analyser le comportement des clients d'un e-commerce de cadeaux afin de prédire le churn (désabonnement ou perte de clients) et améliorer les stratégies marketing.

**Caractéristique unique:** Ce projet implémente **deux modèles distincts** - un modèle de recherche (103 features, 92.46% accuracy) pour l'analyse approfondie et un modèle de déploiement (5 features, 67.22% accuracy) pour l'interface web Flask.

🎯 Objectifs

▪ Comprendre le comportement client
▪ Identifier les clients à risque (churn prediction)
▪ Améliorer la prise de décision marketing
▪ Déployer un système de prédiction en temps réel via Flask

## 🎓 Stratégie à Double Modèle : Recherche vs Déploiement

### Pourquoi deux modèles?

**Modèle de Recherche (103 features) - `models/best_model.pkl`**
- ✅ Accuracy: **92.46%**
- ✅ Precision: 0.9667 | Recall: 0.8007 | F1-Score: 0.8759
- ✅ Usage: Analyse historique, batch processing, insights business
- ✅ Features: 103 (après feature engineering et suppression des features de fuite)

**Modèle de Déploiement (5 features) - `models/simple_model.pkl`**
- ✅ Accuracy: **67.22%** (standard industrie: 60-75% pour modèles simples)
- ✅ Precision: 0.5603 | Recall: 0.4948 | F1-Score: 0.5255
- ✅ Usage: **Interface web Flask** - prédictions en temps réel
- ✅ Features: 5 (age, frequency, monetarytotal, totaltransactions, weekendpurchaseratio)

**Justification Technique:**

Bien que le modèle de recherche atteigne 92.46% d'accuracy, il est **impraticable en production** car il nécessite 103 inputs que les utilisateurs ne peuvent pas fournir via un formulaire web.

Le modèle de déploiement sacrifie de l'accuracy (67% vs 92%) pour gagner en:
- ✅ **Utilisabilité**: Seulement 5 champs à remplir
- ✅ **Expérience utilisateur**: Formulaire simple et rapide
- ✅ **Valeur business**: Identifie 49% des churners en ne contactant que 41% des clients
- ✅ **Fonctionnement correct**: Les prédictions répondent aux inputs utilisateur

---

🗂️ Structure du projet

```
projetML/
│
├── data/
│   ├── raw/                          # Données brutes originales
│   ├── processed/                    # Données nettoyées
│   │   ├── step1_exploration.csv
│   │   ├── step2_data_cleaning.csv
│   │   └── step3_feature_engineering.csv
│   └── train_test/                   # Données train/test split
│       ├── X_train.csv (103 features)
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/                        # Exploration Jupyter
│
├── src/                             # Code source
│   ├── preprocessing.py             # Nettoyage et encodage
│   ├── train_model_v2.py           # Pipeline ML complet
│   ├── predict.py                   # Utilitaires prédiction
│   └── utils.py                     # Fonctions helpers
│
├── models/                          # Modèles sauvegardés
│   ├── best_model.pkl              # Modèle recherche (103 feat)
│   ├── scaler.pkl                  # Scaler modèle recherche
│   ├── simple_model.pkl            # Modèle déploiement (5 feat)
│   ├── simple_scaler.pkl           # Scaler modèle déploiement
│   ├── kmeans_model.pkl            # Segmentation clients
│   ├── pca_model.pkl               # Réduction dimensionnalité
│   └── regression_model.pkl        # Prédiction MonetaryTotal
│
├── app/                            # Application Flask
│   ├── app.py                      # App avec modèle complexe (non recommandé)
│   ├── app_simple.py               # App avec modèle simple (RECOMMANDÉ)
│   └── templates/
│       └── index.html              # Interface web
│
├── reports/                        # Visualisations et résultats
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── clusters_pca.png
│
├── requirements.txt                # Dépendances Python
├── .gitignore
└── README.md                       # Ce fichier
```

---

⚙️ Technologies utilisées

- **Python 3.11**
- **scikit-learn** - Modèles ML et preprocessing
- **pandas** - Manipulation de données
- **numpy** - Calcul numérique
- **matplotlib / seaborn** - Visualisation
- **Flask** - Framework web
- **joblib** - Sérialisation des modèles
- **imbalanced-learn** - SMOTE pour équilibrage classes

---

📌 Étapes du projet

### 1. Exploration des Données
- Analyse de 52 features initiales sur 4372 clients
- Détection des valeurs manquantes (Age: 30%, SupportTickets: 8%)
- Identification des valeurs aberrantes (-1, 999, 99)
- Analyse des corrélations

### 2. Préparation des Données
- **Data Cleaning**: Imputation des valeurs manquantes (médiane)
- **Encodage**: Label Encoding (features ordinales) + One-Hot Encoding (nominales)
- **Feature Engineering**: Création de MonetaryPerDay, AvgBasketValue, TenureRatio
- **Suppression Data Leakage**: Retrait de 19 features ( tenureratio, churnriskcategory,...)
- **Normalisation**: StandardScaler sur features numériques

### 3. Transformation
- **PCA**: Réduction dimensionnalité pour visualisation
- **Corrélation**: Suppression features corrélées > 0.90
- **SMOTE**: Équilibrage classes (33% churn → 50% churn)

### 4. Modélisation
- **Clustering**: K-Means (4 segments clients)
- **Classification**: Random Forest (92.46% accuracy)
- **Régression**: Prédiction MonetaryTotal
- **GridSearchCV**: Optimisation hyperparamètres

### 5. Déploiement
- **Modèle Simple**: Entraînement modèle 5 features (67.22% accuracy)
- **Flask App**: Interface web avec formulaire 5 champs
- **Production**: Déploiement fonctionnel et testé

---

📊 Résultats

```
prédiction correcte des clients 
---

▶️ Installation

### 1. Cloner le Repository

```bash
git clone <your-repo-url>
cd projetML
```

### 2. Créer l'environnement virtuel

```bash
# Créer l'environnement

python -m venv venv

# Activer (Windows)

venv\Scripts\activate

### 3. Installer les dépendances

```bash

pip install -r requirements.txt
```

---

🚀 Utilisation

### Option 1: Interface Web Flask 

**⚠️ IMPORTANT: Utiliser le modèle simple pour le déploiement**

```bash

python app/app_simple.py
```

Puis ouvrir le navigateur à: **http://localhost:5000**

### Option 2: Entraîner les Modèles

**Modèle de recherche complet (103 features):**
```bash
python src/train_model_v2.py
```

**Modèle de déploiement simple (5 features):**
```bash
python train_simple_model.py
```

### Option 3: Exploration des Données

```bash
# Lancer Jupyter pour explorer
jupyter notebook

# Ouvrir notebooks/exploration.ipynb
```

...

## 🎯 Points Clés du Projet

### 1. Détection et Suppression du Data Leakage 

**Impact:** Sans suppression → 100% accuracy (fuite de données)
**Après suppression:** 92.46% accuracy (réaliste)


### 3. Pipeline ML Complet

- ✅ Exploration → Préparation → Transformation → Modélisation → Évaluation → Déploiement
- ✅ Gestion des valeurs manquantes (imputation médiane)
- ✅ Encodage intelligent (Label vs One-Hot)
- ✅ Feature engineering pertinent
- ✅ Équilibrage classes (SMOTE)
- ✅ Validation croisée (GridSearchCV)
- ✅ Interface déployée (Flask)

---

📌 Auteur

**Projet réalisé dans le cadre du module Machine Learning **

