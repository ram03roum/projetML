# Data Overview

 Contexte du projet 

Ce projet a pour objectif d’analyser le comportement des clients d’un e-commerce afin de mieux comprendre leurs habitudes et prédire le churn (départ des clients).

L’analyse repose sur un dataset riche contenant des données transactionnelles, comportementales et démographiques.

 Chargement des données

Les données ont été chargées dynamiquement à partir d’un fichier CSV en utilisant une variable d’environnement.

load_dotenv('../.env')
path = os.geten('RAW_DATA_PATH')
df = pd.read_csv(f"../{path}")

→ Cette approche permet :

▪ une meilleure organisation du projet.
▪ une sécurisation des chemins de données.
▪ une flexibilité pour changer de dataset facilement.

 Structure du dataset

df.shape 
df.info()
✔️ Observations :
Le dataset contient un grand nombre de variables (~52 features).
↪ Nombre de variables
52 features au total de types mixtes :
 ● numériques (int, float)
 ● catégorielles (object)
↪ Variable cible : Churn
0 → Client fidèle
1 → Client ayant quitté

 Analyse descriptive
df.describe()
✔️ Observations :
▪ Présence de valeurs extrêmes dans certaines variables (ex : MonetaryTotal)
▪ Certaines variables ont une dispersion importante (écart-type élevé)
▪ Les distributions ne sont pas toujours normales (asymétrie).

→ Cela indique que :

▪ des transformations seront nécessaires (ex : log transformation)

▪ certaines variables devront être normalisées.

⚠️ Problèmes détectés

1. Valeurs manquantes
colonne Age contient un nombre important de valeurs manquantes (~30%)

2. Valeurs aberrantes
▪ Quantités négatives (erreurs ou retours produits)

▪ Valeurs incohérentes dans certaines variables (SupportTickets, ...)

3. Données incohérentes
▪ Formats de dates différents (RegistrationDate)

▪ Données IP non exploitables directement (LastLoginIP)

4. Variables inutiles
Newsletter → valeur constante (aucune information utile)

