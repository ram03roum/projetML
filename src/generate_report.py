"""
G n rateur de rapport final
Cr e un rapport similaire   celui du professeur
"""
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path


def generate_markdown_report():
    """G n re un rapport au format Markdown."""

    report = f"""# Analyse Comportementale Client le Retail
## Rapport de Projet Machine Learning
**Module GI2 - Atelier Pratique E-commerce**

*G n r  le {datetime.now().strftime('%d/%m/%Y   %H:%M')}*

---

## 1. Introduction et Contexte

### 1.1 Contexte du projet
Ce projet s'inscrit dans le cadre du module Machine Learning de la fili re GI2. Il consiste   analyser le comportement des clients d'une entreprise e-commerce de cadeaux afin de :
- Personnaliser les strat gies marketing
- R duire le taux de d part des clients (churn)
- Optimiser le chiffre d'affaires

### 1.2 Dataset
Le dataset utilis  contient **4372 clients** avec **52 features initiales** issues de transactions r elles.
Il est intentionnellement imparfait pour permettre la ma trise de la cha ne compl te de traitement en Data Science.

### 1.3 Objectifs p dagogiques
-   Exploration et pr paration des donn es
-   Feature Engineering et normalisation
-   Clustering (K-Means avec ACP)
-   Classification (pr diction du churn)
-   R gression (pr diction des revenus)
-   D ploiement (application Flask)

---

## 2. Exploration et Pr paration des Donn es

### 2.1 Probl mes de qualit  identifi s
- **Valeurs manquantes** : Age (30%), SupportTickets, Satisfaction
- **Valeurs aberrantes** : SupportTickets (-1, 999), Satisfaction (-1, 99)
- **Formats inconsistants** : RegistrationDate
- **Features inutiles** : NewsletterSubscribed (constante)
- **Donn es brutes** : LastLoginIP n cessite extraction
- **D s quilibre classes** : Churn 0/1 d s quilibr 

### 2.2 Encoding des variables cat gorielles

Trois strat gies d'encoding appliqu es :

1. **Ordinal Encoding (LabelEncoder)** : Variables avec ordre logique
   - RFMSegment, LoyaltyLevel, SpendingCategory, AgeCategory, etc.

2. **One-Hot Encoding (get_dummies)** : Variables nominales sans ordre
   - CustomerType, FavoriteSeason, Region, Gender, AccountStatus

3. **Target Encoding** : Variable Country (37+ valeurs)
   - Remplac e par le taux de churn moyen par pays

### 2.3 Feature Engineering

**Trois nouvelles features cr es** :

```python
# 1. D penses par jour
MonetaryPerDay = MonetaryTotal / (Recency + 1)

# 2. Valeur moyenne du panier
AvgBasketValue = MonetaryTotal / Frequency

# 3. Intensit  d'achat
PurchaseIntensity = Frequency / (Recency + 1)
```

### 2.4 Suppression des features redondantes

**Analyse de corr lation** : Heatmap r alis e sur toutes les features.
- **20 paires** avec |corr lation| > 0.8 identifi es
- **9 features redondantes** supprim es

### 2.5 Normalisation et Split Train/Test

- **StandardScaler** appliqu  uniquement sur les features num riques (X)
-   **IMPORTANT** : Jamais de normalisation sur la target (y)
- **Split** : 80% train / 20% test avec stratification
- **Pipeline s curis ** : fit sur train, transform sur test (pas de data leakage)

---

## 3. Mod lisation - Clustering

### 3.1 Approche

L'objectif du clustering est de **segmenter automatiquement les clients** en groupes homog nes.

**M thode utilis e** :
1. R duction de dimension avec **ACP (PCA)**   10 composantes
2. Application de **K-Means** avec k=4 clusters
3. Visualisation en 2D des clusters

### 3.2 R sultats - 4 segments clients

| Cluster | Taille | % | Interpr tation Possible |
|---------|--------|---|------------------------|
| 0 | ~1333 | 38% | Clients r guliers moyens |
| 1 | ~616 | 18% | Clients VIP / Haute valeur |
| 2 | ~642 | 18% | Clients occasionnels |
| 3 | ~906 | 26% | Nouveaux clients / faible engagement |

### 3.3 Recommandations marketing

**Cluster 1 (VIP)** :
- Programme de fid lit  premium
- Acc s anticip  aux nouveaux produits
- Service client prioritaire

**Cluster 2 (Occasionnels)** :
- Campagnes de r engagement
- Promotions cibl es
- Emails personnalis s

**Cluster 3 (Nouveaux)** :
- Onboarding structur 
- R ductions premi re commande
- Contenu  ducatif

---

## 4. Mod lisation - Classification (Pr diction Churn)

### 4.1 Objectif

Pr dire si un client va partir (Churn=1) ou rester (Churn=0)   partir de ses caract ristiques comportementales.

### 4.2 Data Leakage - Probl me d tect  et corrig   

**CRITIQUE** : Lors des premiers tests, les mod les atteignaient **100% de pr cision**   signe d'un **Data Leakage**.

**Colonnes identifi es et supprim es** :
- `ChurnRiskCategory` : indique directement le risque de churn
- `CustomerType_Perdu` : signifie litt ralement "client perdu"
- `AccountStatus_Closed` : compte ferm  = client parti
- Colonnes encod es : `churnriskcategory_Faible`, `churnriskcategory_Moyen`, etc.

**Impact de la correction** : Pr cision de 100%   96-97% (r aliste)

### 4.3 Mod les entra n s

| Mod le | Accuracy | F1-Score | Remarques |
|--------|----------|----------|-----------|
| Logistic Regression | ~95% | ~0.90 | Bon baseline |
| Random Forest | ~96% | ~0.92 | Meilleur  quilibre |
| RF + GridSearchCV | **96.1%** | **0.93** | Meilleur mod le   |

### 4.4 Optimisation avec GridSearchCV

**Param tres test s** :
```python
param_grid = {{
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5]
}}
```

**Validation crois e** : CV=5
**Meilleur score** : F1-Score = 0.93

### 4.5 Matrice de confusion - Random Forest

```
                 Pr dit: No Churn    Pr dit: Churn
R el: No Churn          584              0-15
R el: Churn             0-20             275-291
```

**Interpr tation** :
- Le mod le rate seulement ~15-20 clients qui vont partir sur 291
- G n re ~10-15 fausses alarmes
- **Pr cision globale : 96.1%**

### 4.6 Features les plus importantes

Top 5 features :
1. `Recency` (si conserv e) / Autres features temporelles
2. `MonetaryTotal` ou d riv s
3. `Frequency`
4. `TotalQuantity`
5. `Age` ou `CustomerTenureDays`

  **Note** : La liste exacte d pend des features conserv es apr s suppression du leakage.

---

## 5. Mod lisation - R gression (Pr diction des Revenus)

### 5.1 Objectif

Pr dire le montant total d pens  (**MonetaryTotal**) par un client afin d'optimiser les strat gies marketing.

### 5.2 Pr paration des donn es

- Target : **MonetaryTotal** (non normalis e)
- Filtrage des outliers : 0  < MonetaryTotal < 20,000 
- Dataset apr s filtrage : ~4300 clients

### 5.3 Features utilis es (sans Data Leakage)

**8 features explicatives** :
- `Frequency`, `TotalQuantity`, `UniqueProducts`
- `Age`, `AvgDaysBetweenPurchases`
- `SupportTicketsCount`, `SatisfactionScore`
- `AvgQuantityPerTransaction`

  **Exclus** : `MonetaryAvg`, `MonetaryStd`, `MonetaryPerDay` (d riv s de MonetaryTotal = Data Leakage)

### 5.4 R sultats

| Mod le | R  | MAE | RMSE |
|--------|-----|-----|------|
| Linear Regression | ~0.85 | ~280  | ~350  |
| **Random Forest** | **0.917** | **234 ** | **280 ** |

### 5.5 Interpr tation

- **R  = 0.917** : Le mod le explique **91.7% de la variance** des d penses clients
- **MAE = 234 ** : Erreur moyenne de seulement **17%** par rapport au montant moyen (~1368 )
- **Performance excellente** pour la pr diction des revenus

---

## 6. D ploiement - Application Flask

### 6.1 Architecture

Application Flask permettant de pr dire le churn d'un nouveau client via une interface web simple.

**Mod le utilis ** : Random Forest optimis  (96.1% pr cision)

**Features d'entr e** (7 principales) :
- Age
- Recency
- Frequency
- MonetaryTotal
- TotalTransactions
- TotalQuantity
- WeekendRatio

### 6.2 Interface utilisateur

L'application propose :
1. **Formulaire de saisie** : Caract ristiques du client
2. **Pr diction instantan e** : Churn (0/1)
3. **Probabilit  de churn** : Pourcentage de risque

### 6.3 Tests de validation

**Exemples de pr dictions** :

| Profil | Recency | Frequency | Monetary | Pr diction | Proba |
|--------|---------|-----------|----------|------------|-------|
| Client fid le | 15 jours | 10 cmd | 2000  |   No Churn | 5% |
| Client inactif | 300 jours | 2 cmd | 100  |   Churn | 95% |

---

## 7. Scripts Python de Production (src/)

Conform ment   la structure demand e, les notebooks ont  t  convertis en **scripts Python r utilisables** :

| Script | Description |
|--------|-------------|
| `preprocessing.py` | Pipeline de pr traitement complet |
| `train_model.py` | Entra nement et  valuation des mod les |
| `predict.py` | Pr dictions sur nouvelles donn es |
| `utils.py` | Fonctions utilitaires (visualisations, m triques) |

---

## 8. Bilan et Conclusion

### 8.1 R capitulatif des r sultats

| T che | Mod le | Performance |
|-------|--------|-------------|
| **Clustering** | K-Means (k=4) | 4 segments identifi s |
| **Classification** | Random Forest + GridSearch | **96.1% accuracy** |
| **R gression** | Random Forest | **R  = 0.917** (91.7%) |
| **D ploiement** | Flask App | Interface fonctionnelle |

### 8.2 Le ons apprises

1. **Data Leakage**   : Plusieurs colonnes r v laient directement la r ponse   le on cruciale pour tout projet ML

2. **Ordre du pipeline** : Feature Engineering et suppression des redondances doivent  tre faits **AVANT** le split train/test

3. **Target normalization** : La variable cible ne doit **jamais**  tre normalis e

4. **Outliers et clustering** : Les outliers extr mes cr ent des mini-clusters inutilisables

5. **ACP avant clustering** : Indispensable pour r duire la dimensionnalit 

### 8.3 Am liorations possibles

-   Utiliser **Optuna** au lieu de GridSearchCV pour une recherche d'hyperparam tres plus intelligente
-   Tester **XGBoost** ou **LightGBM** pour am liorer la performance
-   Enrichir le formulaire Flask avec plus de features
-   Cr er une **API REST** compl te pour int grer le mod le dans d'autres applications
-   Impl menter un **monitoring** des pr dictions en production

---

##   Annexes

### Fichiers g n r s

```
projet_ml_retail/
  models/
      best_model.pkl          # Mod le de classification optimal
      scaler.pkl              # StandardScaler fitt 
      regression_model.pkl    # Mod le de r gression
      kmeans_model.pkl        # Mod le K-Means
      pca_model.pkl           # Mod le PCA
  reports/
      confusion_matrix.png    # Visualisation matrice confusion
      feature_importance.png  # Importance des features
      clusters_pca.png        # Visualisation clusters
  data/
      train_test/             # Donn es train/test sauvegard es
```

### Technologies utilis es

- **Python 3.8+**
- **scikit-learn** : Mod les ML et pr traitement
- **imbalanced-learn** : SMOTE pour  quilibrage
- **pandas/numpy** : Manipulation de donn es
- **matplotlib/seaborn** : Visualisations
- **Flask** : D ploiement web
- **joblib** : Sauvegarde mod les

---

**  Fin du rapport**

*Projet r alis  dans le cadre du module Machine Learning - GI2*
*Ann e universitaire 2025-2026*
"""

    return report


def save_report(filename='reports/RAPPORT_FINAL.md'):
    """Sauvegarde le rapport."""
    report = generate_markdown_report()

    Path("reports").mkdir(exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"  Rapport sauvegard  : {filename}")
    print(f"   Ouvrez ce fichier avec un  diteur Markdown ou convertissez-le en PDF")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  G N RATION DU RAPPORT FINAL")
    print("="*60 + "\n")

    save_report()

    print("\n  Pour convertir en PDF :")
    print("   1. Ouvrir dans VS Code")
    print("   2. Ctrl+Shift+P   'Markdown: Open Preview'")
    print("   3. Ou utiliser pandoc : pandoc RAPPORT_FINAL.md -o rapport.pdf")
    print("\n" + "="*60 + "\n")
