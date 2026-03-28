# # ✔ Reprendre ton notebook preprocessing
# # ✔ Le transformer en fonctions propres
# # ✔ Être réutilisable pour le training et le test

# # import pandas as pd
# # import numpy as np

# # from sklearn.metrics import accuracy_score
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.model_selection import train_test_split
# # from imblearn.over_sampling import SMOTE




# # # 1. Charger les données
# # def load_data(path):
# #     df=pd.read_csv(path)
# #     return df

# # # 2. Séparer X / y
# # def split_X_y(df, target_col):
# #     X = df.drop(columns=[target_col]) # X = toutes les colonnes SAUF Churn
# #     y = df[target_col]
# #     return X, y

# # # 3. Train / Test split
# # def split_train_test(X, y, test_size=0.2, random_state=42):
# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X, y, test_size=test_size, random_state=random_state , stratify=y
# #     )
# #     return X_train, X_test, y_train, y_test

# # # 4. Encoding
# # def encode_data(X_train, X_test):
# #     # 🔹 SUPPRESSION DES COLONNES FUITES avant get_dummies
# #     # toutes les colonnes qui contiennent 'ChurnRiskCategory'
# #     # --- Détection automatique des colonnes suspectes ---
# #     # leak_cols = [col for col in X_train.columns if "churn" in col.lower()]
# #     # if leak_cols:
# #     #     print("⚠️ Colonnes fuite détectées :", leak_cols)
# #     #     X_train = X_train.drop(columns=leak_cols)
# #     #     X_test  = X_test.drop(columns=[col for col in leak_cols if col in X_test.columns])
# #     # else:
# #     #     print("✅ pas de fuite détectée !")

# #     leak_cols = ['ChurnRiskCategory', 'CustomerType', 'RFMSegment', 'LoyaltyLevel']
# #     X_train = X_train.drop(columns=[col for col in leak_cols if col in X_train.columns])
# #     X_test  = X_test.drop(columns=[col for col in leak_cols if col in X_test.columns])


# #     # encoder les variables catégorielles restantes
# #     cat_cols = X_train.select_dtypes(include=['object']).columns
# #     X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True) 
# #     X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

# #     # aligner colonnes train/test
# #     # X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
# #     X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)
# #     print("✅ Colonnes de fuite supprimées et encodage terminé !")

# #     return X_train, X_test



# # # 5. Nettoyage final
# # def clean_data(X_train, X_test):

# #     # remplacer inf
# #     X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
# #     X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

# #     # remplir NaN
# #     X_train.fillna(X_train.median(numeric_only=True), inplace=True)
# #     X_test.fillna(X_train.median(numeric_only=True), inplace=True)

# #     return X_train, X_test

# # # 6. Scaling
# # def scale_data(X_train, X_test):
# #     scaler = StandardScaler() 
# #     num_cols = X_train.select_dtypes(include=['int64','float64']).columns

# #     # X_train[num_cols] = scaler.fit_transform(X_train[num_cols]) 
# #     # X_test[num_cols] = scaler.transform(X_test[num_cols]) 
# #     X_train_scaled = X_train.copy()
# #     X_test_scaled = X_test.copy()

# #     X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
# #     X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# #     return X_train_scaled, X_test_scaled, scaler

# # # 7. SMOTE
# # def apply_smote(X_train, y_train):
# #     smote = SMOTE(random_state=42)
# #     return smote.fit_resample(X_train, y_train)

# # # 8. PCA  => pas toujours utilisé
# # #preprocessing.py → pipeline général

# # # 8. Pipeline complet
# # def preprocess_pipeline(path):

# #     print("📥 Chargement des données...")
# #     df = load_data(path)

# #     print("🎯 Séparation X / y...")
# #     X, y = split_X_y(df, target_col='Churn')

# #     print("✂️ Train/Test split...")
# #     X_train, X_test, y_train, y_test = split_train_test(X, y)


# #     print("🔢 Encoding...")
# #     X_train, X_test = encode_data(X_train, X_test)

# #         # ✅ Corrélation uniquement sur les colonnes numériques
# #     numeric_cols = X_train.select_dtypes(include=[np.number]).columns
# #     df_corr = pd.concat([X_train[numeric_cols], y_train], axis=1)
# #     corr = df_corr.corr()['Churn'].sort_values(ascending=False)
# #     print("Colonnes fortement corrélées avec Churn :\n", corr[corr > 0.9])

# #     print("🧹 Nettoyage...")
# #     X_train, X_test = clean_data(X_train, X_test)

# #     print("📏 Scaling...")
# #     X_train, X_test, scaler = scale_data(X_train, X_test)

# #     print("⚖️ SMOTE...")
# #     X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

# #     print("✅ Preprocessing terminé !")

# #     for col in X_train.columns:
# #         if 'ChurnRiskCategory' in col:
# #             print("⚠️ fuite encore présente :", col)
# #     print("✅ pas de fuite détectée !")
    

# #     return X_train, X_test, y_train, y_test, X_train_resampled, y_train_resampled, scaler
    




# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE


# # ─────────────────────────────────────────
# # 1. Chargement
# # ─────────────────────────────────────────
# def load_data(path):
#     df = pd.read_csv(path)
#     print(f"  Shape chargé : {df.shape}")
#     return df


# # ─────────────────────────────────────────
# # 2. Nettoyage des valeurs aberrantes
# #    DOIT être fait AVANT fillna
# # ─────────────────────────────────────────
# def fix_outliers(df):
#     # SupportTickets : -1 et 999 sont des codes d'erreur
#     if 'SupportTicketsCount' in df.columns:
#         df['SupportTicketsCount'] = df['SupportTicketsCount'].replace([-1, 999], np.nan)

#     # Satisfaction : -1 et 99 sont des codes d'erreur
#     if 'SatisfactionScore' in df.columns:
#         df['SatisfactionScore'] = df['SatisfactionScore'].replace([-1, 99], np.nan)

#     # MonetaryTotal : clip les valeurs négatives avant la log
#     if 'MonetaryTotal' in df.columns:
#         df['MonetaryTotal'] = df['MonetaryTotal'].clip(lower=0)

#     # TotalQuantity et MinQuantity : quantités négatives → 0
#     if 'TotalQuantity' in df.columns:
#         df['TotalQuantity'] = df['TotalQuantity'].clip(lower=0)
#     if 'MinQuantity' in df.columns:
#         df['MinQuantity'] = df['MinQuantity'].clip(lower=0)

#     return df


# # ─────────────────────────────────────────
# # 3. Imputation des valeurs manquantes
# #    APRÈS fix_outliers
# # ─────────────────────────────────────────
# def impute_missing(df):
#     df = df.fillna(df.median(numeric_only=True))
#     return df


# # ─────────────────────────────────────────
# # 4. Séparation X / y
# # ─────────────────────────────────────────
# def split_X_y(df, target_col='Churn'):
#     X = df.drop(columns=[target_col])
#     y = df[target_col]
#     return X, y


# # ─────────────────────────────────────────
# # 5. Train / Test split
# # ─────────────────────────────────────────
# def split_train_test(X, y, test_size=0.2, random_state=42):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y,
#         test_size=test_size,
#         random_state=random_state,
#         stratify=y        # préserve la proportion Churn
#     )
#     print(f"  Train : {X_train.shape} | Test : {X_test.shape}")
#     return X_train, X_test, y_train, y_test


# # ─────────────────────────────────────────
# # 6. Suppression des colonnes fuite
# # ─────────────────────────────────────────
# def drop_leaky_cols(X_train, X_test):
#     leak_cols = ['ChurnRiskCategory', 'CustomerType', 'RFMSegment', 'LoyaltyLevel']
#     to_drop = [c for c in leak_cols if c in X_train.columns]
#     if to_drop:
#         print(f"  Colonnes fuite supprimées : {to_drop}")
#     X_train = X_train.drop(columns=to_drop)
#     X_test  = X_test.drop(columns=[c for c in to_drop if c in X_test.columns])
#     return X_train, X_test


# def remove_correlated_features(X_train, X_test, threshold=0.90):
#     numeric_cols = X_train.select_dtypes(include=[np.number]).columns
#     corr_matrix = X_train[numeric_cols].corr().abs()
#     upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#     to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

#     if to_drop:
#         print(f"  Colonnes fortement corrélées (> {threshold}) supprimées : {to_drop}")
#         X_train = X_train.drop(columns=to_drop)
#         X_test = X_test.drop(columns=[c for c in to_drop if c in X_test.columns])
#     else:
#         print(f"  Aucune colonne corrélée à plus de {threshold} détectée.")

#     return X_train, X_test


# # ─────────────────────────────────────────
# # 7. Encodage des variables catégorielles
# #    DOIT être fait AVANT le scaling
# # ─────────────────────────────────────────
# def encode_data(X_train, X_test):
#     cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

#     X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
#     X_test  = pd.get_dummies(X_test,  columns=cat_cols, drop_first=True)

#     # Aligner : X_test prend les colonnes de X_train (manquantes → 0)
#     X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

#     print(f"  Après encodage → {X_train.shape[1]} colonnes")
#     return X_train, X_test


# # ─────────────────────────────────────────
# # 8. Nettoyage final (inf / NaN résiduels)
# # ─────────────────────────────────────────
# def clean_inf_nan(X_train, X_test):
#     X_train = X_train.replace([np.inf, -np.inf], np.nan)
#     X_test  = X_test.replace([np.inf, -np.inf], np.nan)

#     # On remplit avec la médiane de X_train (pas de fuite)
#     train_median = X_train.median(numeric_only=True)
#     X_train = X_train.fillna(train_median)
#     X_test  = X_test.fillna(train_median)

#     return X_train, X_test


# # ─────────────────────────────────────────
# # 9. Scaling
# #    APRÈS encode_data → le scaler connaît
# #    toutes les colonnes (ex. 125)
# # ─────────────────────────────────────────
# def scale_data(X_train, X_test):
#     scaler   = StandardScaler()
#     num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

#     X_train_scaled = X_train.copy()
#     X_test_scaled  = X_test.copy()

#     X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
#     X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])

#     print(f"  Scaler fit sur {len(num_cols)} colonnes numériques")
#     return X_train_scaled, X_test_scaled, scaler


# # ─────────────────────────────────────────
# # 10. SMOTE (sur X_train scalé uniquement)
# # ─────────────────────────────────────────
# def apply_smote(X_train, y_train, random_state=42):
#     smote = SMOTE(random_state=random_state)
#     X_res, y_res = smote.fit_resample(X_train, y_train)
#     print(f"  Après SMOTE : {dict(zip(*np.unique(y_res, return_counts=True)))}")
#     return X_res, y_res


# # ─────────────────────────────────────────
# # 11. Pipeline complète
# # ─────────────────────────────────────────
# def preprocess_pipeline(path, target_col='Churn'):

#     print("─" * 40)
#     print("📥 Chargement...")
#     df = load_data(path)

#     print("🔧 Correction des aberrants...")
#     df = fix_outliers(df)

#     print("🩹 Imputation des NaN...")
#     df = impute_missing(df)

#     print("✂️  Séparation X / y...")
#     X, y = split_X_y(df, target_col)

#     print("📊 Train / Test split...")
#     X_train, X_test, y_train, y_test = split_train_test(X, y)

#     print("🚫 Suppression colonnes fuite...")
#     X_train, X_test = drop_leaky_cols(X_train, X_test)

#     print("� Suppression des features corrélées...")
#     X_train, X_test = remove_correlated_features(X_train, X_test, threshold=0.90)

#     print("�🔢 Encodage (get_dummies)...")
#     X_train, X_test = encode_data(X_train, X_test)

#     print("🧹 Nettoyage inf / NaN résiduels...")
#     X_train, X_test = clean_inf_nan(X_train, X_test)

#     print("📏 Scaling (après encodage)...")
#     X_train, X_test, scaler = scale_data(X_train, X_test)

#     print("⚖️  SMOTE...")
#     X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

#     print("─" * 40)
#     print(f"✅ Pipeline terminée !")
#     print(f"   X_train       : {X_train.shape}")
#     print(f"   X_train SMOTE : {X_train_resampled.shape}")
#     print(f"   X_test        : {X_test.shape}")
#     print("─" * 40)

#     return (
#         X_train,
#         X_test,
#         y_train,
#         y_test,
#         X_train_resampled,
#         y_train_resampled,
#         scaler
#     )

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# ─────────────────────────────────────────
# 1. Chargement
# ─────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path)
    print(f"  Shape chargé : {df.shape}")
    return df


# ─────────────────────────────────────────
# 2. Nettoyage des valeurs aberrantes
#    DOIT être fait AVANT fillna
# ─────────────────────────────────────────
def fix_outliers(df):
    # SupportTickets : -1 et 999 sont des codes d'erreur
    if 'supportticketscount' in df.columns:
        df['supportticketscount'] = df['supportticketscount'].replace([-1, 999], np.nan)

    # Satisfaction : -1 et 99 sont des codes d'erreur
    if 'satisfactionScore' in df.columns:
        df['satisfactionscore'] = df['satisfactionscore'].replace([-1, 99], np.nan)

    # MonetaryTotal : clip les valeurs négatives avant la log
    if 'monetaryTotal' in df.columns:
        df['MonetaryTotal'] = df['onetaryTotal'].clip(lower=0)

    # TotalQuantity et MinQuantity : quantités négatives → 0
    if 'totalQuantity' in df.columns:
        df['TotalQuantity'] = df['TotalQuantity'].clip(lower=0)
    if 'minQuantity' in df.columns:
        df['MinQuantity'] = df['MinQuantity'].clip(lower=0)

    return df


# ─────────────────────────────────────────
# 3. Imputation des valeurs manquantes
#    APRÈS fix_outliers
# ─────────────────────────────────────────
def impute_missing(df):
    df = df.fillna(df.median(numeric_only=True))
    return df


# ─────────────────────────────────────────
# 4. Séparation X / y
# ─────────────────────────────────────────
def split_X_y(df, target_col='Churn'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# ─────────────────────────────────────────
# 5. Train / Test split
# ─────────────────────────────────────────
def split_train_test(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y        # préserve la proportion Churn
    )
    print(f"  Train : {X_train.shape} | Test : {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────
# 6. Suppression des colonnes fuite
# ─────────────────────────────────────────
def drop_leaky_cols(X_train, X_test):
    leak_cols = ['ChurnRiskCategory', 'CustomerType', 'RFMSegment', 'LoyaltyLevel']
    to_drop = [c for c in leak_cols if c in X_train.columns]
    if to_drop:
        print(f"  Colonnes fuite supprimées : {to_drop}")
    X_train = X_train.drop(columns=to_drop)
    X_test  = X_test.drop(columns=[c for c in to_drop if c in X_test.columns])
    return X_train, X_test


def remove_correlated_features(X_train, X_test, threshold=0.90):
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    corr_matrix = X_train[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        print(f"  Colonnes fortement corrélées (> {threshold}) supprimées : {to_drop}")
        X_train = X_train.drop(columns=to_drop)
        X_test = X_test.drop(columns=[c for c in to_drop if c in X_test.columns])
    else:
        print(f"  Aucune colonne corrélée à plus de {threshold} détectée.")

    return X_train, X_test


# ─────────────────────────────────────────
# 7. Encodage des variables catégorielles
#    DOIT être fait AVANT le scaling
# ─────────────────────────────────────────
def encode_data(X_train, X_test):
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test  = pd.get_dummies(X_test,  columns=cat_cols, drop_first=True)

    # Aligner : X_test prend les colonnes de X_train (manquantes → 0)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    print(f"  Après encodage → {X_train.shape[1]} colonnes")
    return X_train, X_test


# ─────────────────────────────────────────
# 8. Nettoyage final (inf / NaN résiduels)
# ─────────────────────────────────────────
def clean_inf_nan(X_train, X_test):
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan)

    # On remplit avec la médiane de X_train (pas de fuite)
    train_median = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_median)
    X_test  = X_test.fillna(train_median)

    return X_train, X_test


# ─────────────────────────────────────────
# 9. Scaling
#    APRÈS encode_data → le scaler connaît
#    toutes les colonnes (ex. 125)
# ─────────────────────────────────────────
def scale_data(X_train, X_test):
    scaler   = StandardScaler()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()

    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])

    print(f"  Scaler fit sur {len(num_cols)} colonnes numériques")
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────
# 10. SMOTE (sur X_train scalé uniquement)
# ─────────────────────────────────────────
def apply_smote(X_train, y_train, random_state=42):
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  Après SMOTE : {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


# ─────────────────────────────────────────
# 11. Pipeline complète
# ─────────────────────────────────────────
def preprocess_pipeline(path, target_col='Churn'):

    print("─" * 40)
    print("📥 Chargement...")
    df = load_data(path)

    print("🔧 Correction des aberrants...")
    df = fix_outliers(df)

    print("🩹 Imputation des NaN...")
    df = impute_missing(df)

    print("✂️  Séparation X / y...")
    X, y = split_X_y(df, target_col)

    print("📊 Train / Test split...")
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    print("🚫 Suppression colonnes fuite...")
    X_train, X_test = drop_leaky_cols(X_train, X_test)

    print("  Suppression des features corrélées...")
    X_train, X_test = remove_correlated_features(X_train, X_test, threshold=0.90)

    print(" 🔢 Encodage (get_dummies)...")
    X_train, X_test = encode_data(X_train, X_test)

    print("🧹 Nettoyage inf / NaN résiduels...")
    X_train, X_test = clean_inf_nan(X_train, X_test)

    print("📏 Scaling (après encodage)...")
    X_train, X_test, scaler = scale_data(X_train, X_test)

    print("⚖️  SMOTE...")
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    print("─" * 40)
    print(f"✅ Pipeline terminée !")
    print(f"   X_train       : {X_train.shape}")
    print(f"   X_train SMOTE : {X_train_resampled.shape}")
    print(f"   X_test        : {X_test.shape}")
    print("─" * 40)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_resampled,
        y_train_resampled,
        scaler
    )


