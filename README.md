# 🧠 Projet Machine Learning – Prédiction du Churn E-commerce

## 📕 Description
Ce projet vise à analyser le comportement des clients d’un e-commerce de cadeaux afin de :
- prédire le **churn** (perte de clients)
- améliorer les **stratégies marketing**
- proposer un **système de prédiction en temps réel via Flask**

---

## 🎯 Objectifs
- 🔍 Comprendre le comportement client  
- ⚠️ Identifier les clients à risque (churn prediction)  
- 📈 Améliorer la prise de décision marketing  
- 🌐 Déployer une application web interactive  
- 📊 Fournir des insights actionnables  

---

## 📊 Résultats du Modèle

### 🔹 Classification (Churn)
- **Modèle** : Random Forest (optimisé avec GridSearchCV)  
- **Accuracy** : 82.86%  
- **F1-score** : 0.744  
- **Precision** : 0.739  
- **Recall** : 0.749  

📌 *Interprétation* :  
Le modèle présente un bon équilibre entre précision et rappel, essentiel pour détecter efficacement les clients à risque.

---

### 🔹 Régression (Revenu)
- **Modèle** : Random Forest Regressor  
- **R²** : 0.901  
- **MAE** : £217.5  
- **RMSE** : £355.4  

📌 *Interprétation* :  
Le modèle explique environ 90% de la variance du revenu client → très bonne performance prédictive.

---

## 🚀 Entraînement du Modèle

### 📁 Script principal
```bash
python src/train_model.py
````
Features utilisées ( 8 au total):

CUSTOM_FEATURES = [
    'frequency',
    'uniqueproducts',
    'avgdaysbetweenpurchases',
    'uniqueinvoices',
    'monetarytotal',
    'satisfactionscore',
    'regyear',
    'regmonth'
]

### ⚙️ Pipeline d'entraînement
- Chargement des données
- Nettoyage et imputation des valeurs manquantes
- Feature engineering
- Normalisation (StandardScaler)
- Équilibrage des classes (SMOTE)
- Entraînement (Random Forest)
- Optimisation (GridSearchCV)
- valuation (Accuracy, F1-score, etc.)
- Sauvegarde du modèle


### Synchronisation avec Flask
Le script met automatiquement à jour l'application Flask :

Features alignées : Même ordre dans training et déploiement
Preprocessing identique : Même scaler utilisé
Tests intégrés : Validation avec cas d'usage réalistes
# Après entraînement, lancer l'app Flask
```
python app/app.py
# Ouvrir : http://localhost:5000
```
```
🗂️ Structure du projet

projetML/
│
├── data/
│   ├── raw/                          # Données brutes originales
│   ├── processed/                    # Données nettoyées
│   │   ├── step1_exploration.csv
│   │   ├── step2_data_cleaning.csv
│   │   └── step3_feature_engineering.csv
│   │   └── feature_engineering.csv
│   └── train_test/                   # Données train/test split
│       ├── X_train.csv 
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/                        # Exploration Jupyter
│
├── src/                             # Code source
│   ├── preprocessing.py             # Nettoyage et encodage
│   ├── train_model.py            # Pipeline ML complet (recherche)
│   ├── train_simple_model.py        # Entraînement modèle simple
│   ├── predict.py                   # Utilitaires prédiction
│   └── utils.py                     # Fonctions helpers
│
├── models/                          # Modèles sauvegardés
│
├── app/                            # Application Flask
│   ├── app_simple.py               # 🆕 App Flask optimisée (RECOMMANDÉ)
│   └── templates/
│       └── index.html              # 🆕 Interface web mise à jour
│
├── docs/                           # 🆕 Documentation
│
├── reports/                        # Visualisations et résultats
│
├── requirements.txt                # Dépendances Python
├── .gitignore
└── README.md                       # Ce fichier

```
### ⚙️ Technologies utilisées

- Python 3.11
- scikit-learn - Modèles ML et preprocessing
- pandas - Manipulation de données
- numpy - Calcul numérique
- matplotlib / seaborn - Visualisation
- Flask - Framework web
- joblib - Sérialisation des modèles
- imbalanced-learn - SMOTE pour équilibrage classes
  
### 🚀 Démarrage Rapide (Quick Start)

1. Installation
```
# Cloner le repository
git clone https://github.com/ram03roum/projetML.git
cd projetML
```
```

# Créer et activer l'environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
# ou: source venv/bin/activate  # Linux/Mac
```
```

# Installer les dépendances
pip install -r requirements.txt
```

2. Entraîner le Modèle  (8 features)
```

# Entraîner le modèle de déploiement
python src/train_model.py
```

3. Lancer l'Interface Web
```

# Démarrer Flask
python app/app.py
Puis ouvrir: http://localhost:5000
```

📌 Étapes du projet

1. Exploration des Données
- Analyse de 52 features initiales sur 4372 clients
- Détection des valeurs manquantes (Age: 30%, SupportTickets: 8%)
-Identification des valeurs aberrantes (-1, 999, 99)
- Analyse des corrélations
  
2. Préparation des Données
- Data Cleaning: Imputation des valeurs manquantes (médiane)
- Encodage: Label Encoding (features ordinales) + One-Hot Encoding (nominales)
- Feature Engineering: Création de MonetaryPerDay, AvgBasketValue, TenureRatio
- Suppression Data Leakage: Retrait des features ( tenureratio, churnriskcategory,...)
- Normalisation: StandardScaler sur features numériques
  
3. Transformation
- PCA: Réduction dimensionnalité pour visualisation
- Corrélation: Suppression features corrélées > 0.80
- SMOTE: Équilibrage classes (33% churn → 50% churn)

5. Modélisation
- Clustering: K-Means (4 segments clients)
- Classification: Random Forest (bonne accuracy)
- Régression: Prédiction MonetaryTotal
- GridSearchCV: Optimisation hyperparamètres

6. Déploiement
- Modèle : Entraînement modèle 8 features personnalisées
- Script d'entraînement: src/train_model.py
- Flask App: Interface web avec formulaire 8 champs
- Production: Déploiement fonctionnel et testé
- Documentation: Guides complets d'utilisation
  
7. 🆕 Entraînement Modèle Simple Personnalisé
- Features sélectionnées: 8 features optimisées pour l'interface web
- Pipeline automatisé: Data loading, preprocessing, training, validation
- Synchronisation Flask: Mise à jour automatique de l'interface web
- Tests intégrés: Validation avec cas d'usage réalistes
3. Pipeline ML Complet
✅ Exploration → Préparation → Transformation → Modélisation → Évaluation → Déploiement
✅ Gestion des valeurs manquantes (imputation médiane)
✅ Encodage intelligent (Label vs One-Hot)
✅ Feature engineering pertinent
✅ Équilibrage classes (SMOTE)
✅ Validation croisée (GridSearchCV)
✅ Interface déployée (Flask)
📌 Auteur

**Projet réalisé dans le cadre du module Machine Learning **
