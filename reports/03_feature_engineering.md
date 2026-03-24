# Feature Engineering


## Objectif

L’objectif de cette étape est de transformer les données nettoyées en variables pertinentes pour améliorer la performance des modèles de Machine Learning dans la prédiction du churn client.

## Dataset d’entrée

Le dataset utilisé provient de l’étape précédente `cleaning`. En effet, après le nettoyage, les données sont cohérentes, sans doublons et avec un nombre minimal de valeurs manquantes.

### 🔹 Analyse rapide des colonnes

Séparation des colonnes numériques et catégorielles.

## 1. Création de nouvelles features numériques

### 1.1 Features basiques

#### ➤ MonetaryPerDay

df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)

→ Permet de mesurer la dépense moyenne quotidienne du client par jour.

#### ➤ AvgBasketValue

df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'] + 1)

→ Indique le montant moyen dépensé par transaction. => client VIP vs petit client

#### ➤ PurchaseIntensity

df['PurchaseIntensity'] = df['Frequency'] / (df['CustomerTenureDays'] + 1)

→ Mesure la fréquence d’achat relative à l’ancienneté du client. => client fidèle ou pas

#### ➤ TenureRatio

df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)

→ Permet de détecter les clients inactifs. Permet de comparer ancienneté vs activité récente => détecter churn 

## 2. Distribution des nouvelles features

### 2.1 Histogrammes

![histogramme de MonetaryPerDay](images/histogramme%20MonetaryPerDay.png)

La distribution de MonetaryPerDay montre une forte concentration de faibles valeurs, indiquant que la majorité des clients ont une activité faible, ce qui peut être un indicateur de churn.

![histogramme de AvgBasketValue](images/histogrammeAvgBasketValue.png)

![histogramme de PurchaseIntensity](images/histogrammePurchaseIntensity.png)

![histogramme de TenureRatio](images/histogrammeTenureRatio.png)

✔️ Observations :

▪ Certaines variables présentent une distribution asymétrique.

▪ Les valeurs élevées de TenureRatio indiquent des clients inactifs.

### 2.2 Boxplots vs Churn

![MonetaryPerDay vs Churn](images/MonetaryPerDay%20vs%20Churn.png)

![histogramme de TenureRatio](images/TenureRatio%20vs%20Churn.png)

✔️ Observations :





