# 🧠 Projet Machine Learning - Prédiction du Churn E-commerce

**Analyse du Comportement Client et Système de Prédiction en Temps Réel**

## 📕 Description

Ce projet vise à analyser le comportement des clients d'un e-commerce de cadeaux afin de prédire le churn (désabonnement ou perte de clients) et améliorer les stratégies marketing.

**Caractéristique unique:** Ce projet implémente **deux modèles distincts** - un modèle de recherche (103 features, 92.46% accuracy) pour l'analyse approfondie et un modèle de déploiement (7 features, ~70% accuracy) pour l'interface web Flask.

## 🎯 Objectifs

- 🔍 Comprendre le comportement client
- ⚠️ Identifier les clients à risque (churn prediction)
- 📈 Améliorer la prise de décision marketing  
- 🌐 Déployer un système de prédiction en temps réel via Flask
- 📊 Fournir des insights actionnables pour le business

## 🎓 Stratégie à Double Modèle : Recherche vs Déploiement

### Pourquoi deux modèles?

**Modèle de Recherche (103 features) - `models/best_model.pkl`**
- ✅ Accuracy: **92.46%**
- ✅ Precision: 0.9667 | Recall: 0.8007 | F1-Score: 0.8759
- ✅ Usage: Analyse historique, batch processing, insights business
- ✅ Features: 103 (après feature engineering et suppression des features de fuite)

**Modèle de Déploiement (7 features) - `models/simple_model.pkl`**
- ✅ Accuracy: **65-80%** (réaliste pour déploiement)
- ✅ Precision: 0.55-0.70 | Recall: 0.45-0.65 | F1-Score: 0.50-0.65
- ✅ Usage: **Interface web Flask** - prédictions en temps réel
- ✅ Features: 7 (age, frequency, monetarytotal, totaltransactions, weekendpurchaseratio, avgquantitypertransaction, recency)
- ✅ **Nouveau**: Pipeline d'entraînement personnalisé avec features sélectionnées

**Justification Technique:**

Bien que le modèle de recherche atteigne 92.46% d'accuracy, il est **impraticable en production** car il nécessite 103 inputs que les utilisateurs ne peuvent pas fournir via un formulaire web.

Le modèle de déploiement sacrifie de l'accuracy (70% vs 92%) pour gagner en:
- ✅ **Utilisabilité**: Seulement 7 champs à remplir
- ✅ **Expérience utilisateur**: Formulaire simple et rapide
- ✅ **Maintenance**: Pipeline d'entraînement simplifié
- ✅ **Robustesse**: Moins de risques de data leakage
- ✅ **Performance**: Prédictions rapides (<100ms)

---

## 🚀 Entraînement du Modèle Simple

### Script d'Entraînement Personnalisé

**Fichier:** `src/train_simple_model.py`

```bash
# Entraîner le modèle simple avec 7 features
python src/train_simple_model.py
```

**Features utilisées (7 au total):**
```python
CUSTOM_FEATURES = [
    'age',                         
    'frequency',                   
    'monetarytotal',               
    'totaltransactions',           
    'weekendpurchaseratio',        
    'avgquantitypertransaction',   
    'recency'                      
]
```

**Pipeline d'entraînement:**
1. **Chargement des données** depuis `data/processed/step3_feature_engineering.csv`
2. **Sélection des features** et gestion des valeurs manquantes
3. **Préprocessing** : StandardScaler + SMOTE pour équilibrage
4. **Entraînement** : Random Forest avec paramètres optimisés
5. **Évaluation** : Métriques de performance et tests
6. **Sauvegarde** : Modèle + scaler + documentation

**Fichiers générés:**
- `models/simple_model.pkl` - Modèle Random Forest entraîné
- `models/simple_scaler.pkl` - StandardScaler ajusté 
- `models/simple_model_features.txt` - Liste des features et métriques
- `reports/custom_simple_model_confusion_matrix.png` - Matrice de confusion

### Synchronisation avec Flask

Le script met automatiquement à jour l'application Flask :
- **Features alignées** : Même ordre dans training et déploiement
- **Preprocessing identique** : Même scaler utilisé dans les 2 phases
- **Tests intégrés** : Validation avec cas d'usage réalistes

```bash
# Après entraînement, lancer l'app Flask
python app/app_simple.py
# Ouvrir : http://localhost:5000
```

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
│   ├── train_model_v2.py            # Pipeline ML complet (recherche)
│   ├── train_simple_model.py        # 🆕 Entraînement modèle simple
│   ├── predict.py                   # Utilitaires prédiction
│   └── utils.py                     # Fonctions helpers
│
├── models/                          # Modèles sauvegardés
│   ├── best_model.pkl              # Modèle recherche (103 feat)
│   ├── scaler.pkl                  # Scaler modèle recherche
│   ├── simple_model.pkl            # 🆕 Modèle déploiement (7 feat)
│   ├── simple_scaler.pkl           # 🆕 Scaler modèle déploiement
│   ├── simple_model_features.txt   # 🆕 Liste des features et métriques
│   ├── kmeans_model.pkl            # Segmentation clients
│   ├── pca_model.pkl               # Réduction dimensionnalité
│   └── regression_model.pkl        # Prédiction MonetaryTotal
│
├── app/                            # Application Flask
│   ├── app_simple.py               # 🆕 App Flask optimisée (RECOMMANDÉ)
│   └── templates/
│       └── index.html              # 🆕 Interface web mise à jour
│
├── docs/                           # 🆕 Documentation
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

## 🚀 Démarrage Rapide (Quick Start)

### 1. Installation

```bash
# Cloner le repository
git clone https://github.com/votre-username/projetML.git
cd projetML

# Créer et activer l'environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
# ou: source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Entraîner le Modèle Simple (7 features)

```bash
# Entraîner le modèle de déploiement
python src/train_simple_model.py
```

**Output attendu:**
```
✅ Success! Model trained with 7 features
📊 Final accuracy: XX.X%
```

### 3. Lancer l'Interface Web

```bash
# Démarrer Flask (depuis la racine du projet)
python app/app.py
```

**Puis ouvrir:** [http://localhost:5000](http://localhost:5000)

### 4. Tester une Prédiction

**Exemple - Client à Risque:**
- Age: `25`
- Weekend Ratio: `0.1` 
- Total Transactions: `3`
- Avg Quantity per Transaction: `1.2`
- Recency: `250` (8+ mois d'inactivité)
- Frequency: `2`
- Monetary Total: `150.00`

**Résultat attendu:** ⚠️ Customer WILL Churn (~80%)

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
- **Modèle Simple**: Entraînement modèle 7 features personnalisées
- **Script d'entraînement**: `src/train_simple_model.py` 
- **Flask App**: Interface web avec formulaire 7 champs
- **Production**: Déploiement fonctionnel et testé
- **Documentation**: Guides complets d'utilisation

### 6. 🆕 Entraînement Modèle Simple Personnalisé
- **Features sélectionnées**: 7 features optimisées pour l'interface web
- **Pipeline automatisé**: Data loading, preprocessing, training, validation
- **Synchronisation Flask**: Mise à jour automatique de l'interface web
- **Tests intégrés**: Validation avec cas d'usage réalistes

---

📊 Résultats

### Performance du Modèle Simple
```
✅ Features utilisées: 7
📊 Accuracy: 65-80%
⚡ Temps de réponse: <100ms
🎯 Cas d'usage: Interface web temps réel
```

### Exemples de Prédictions

**Client Fidèle:**
- Age: 45, Frequency: 15, Monetary: 2500£, Recency: 30 jours
- **Résultat:** ✅ No Churn (95% confident)

**Client à Risque:**
- Age: 25, Frequency: 2, Monetary: 150£, Recency: 250 jours  
- **Résultat:** ⚠️ Churn (85% confident)

---

## 🎯 Points Clés du Projet

### 1. Détection et Suppression du Data Leakage 

**Impact:** Sans suppression → 100% accuracy (fuite de données)  
**Après suppression:** 92.46% accuracy (réaliste)

### 2. Stratégie Double Modèle

- **Recherche**: 103 features, 92.46% accuracy, insights business
- **Déploiement**: 7 features, 70% accuracy, interface utilisateur

### 3. Pipeline ML Complet

- ✅ Exploration → Préparation → Transformation → Modélisation → Évaluation → Déploiement
- ✅ Gestion des valeurs manquantes (imputation médiane)
- ✅ Encodage intelligent (Label vs One-Hot)
- ✅ Feature engineering pertinent
- ✅ Équilibrage classes (SMOTE)
- ✅ Validation croisée (GridSearchCV)
- ✅ Interface déployée (Flask)

### 4. 🆕 Nouvelles Fonctionnalités

- **Pipeline d'entraînement personnalisé** pour le modèle simple
- **Documentation complète** avec guides étape par étape
- **Interface Flask optimisée** avec 7 features bien définies
- **Validation robuste** contre le data leakage
- **Tests automatisés** avec cas d'usage réalistes

---

📌 Auteur

**Projet réalisé dans le cadre du module Machine Learning **

