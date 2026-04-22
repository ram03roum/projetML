import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# -----------------------------------------
# 1. Chargement
# -----------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    print(f"  Shape charg  : {df.shape}")
    return df


# -----------------------------------------
# 2. Nettoyage des valeurs aberrantes
#    DOIT  tre fait AVANT fillna
# -----------------------------------------
def fix_outliers(df):
    # SupportTickets : -1 et 999 sont des codes d'erreur
    if 'supportticketscount' in df.columns:
        df['supportticketscount'] = df['supportticketscount'].replace([-1, 999], np.nan)

    # Satisfaction : -1 et 99 sont des codes d'erreur
    if 'satisfactionscore' in df.columns:
        df['satisfactionscore'] = df['satisfactionscore'].replace([-1, 99], np.nan)

    # MonetaryTotal : clip les valeurs n gatives avant la log
    if 'monetarytotal' in df.columns:
        df['monetarytotal'] = df['monetarytotal'].clip(lower=0)

    # TotalQuantity et MinQuantity : quantit s n gatives   0
    if 'totalquantity' in df.columns:
        df['totalquantity'] = df['totalquantity'].clip(lower=0)
    if 'minquantity' in df.columns:
        df['minquantity'] = df['minquantity'].clip(lower=0)

    return df


# -----------------------------------------
# 3. Imputation des valeurs manquantes
#    APR S fix_outliers
# -----------------------------------------
def impute_missing(df):
    df = df.fillna(df.median(numeric_only=True))
    return df


# -----------------------------------------
# 4. S paration X / y
# -----------------------------------------
def split_X_y(df, target_col='Churn'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# -----------------------------------------
# 5. Train / Test split
# -----------------------------------------
def split_train_test(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y        # pr serve la proportion Churn
    )
    print(f"  Train : {X_train.shape} | Test : {X_test.shape}")
    return X_train, X_test, y_train, y_test


# -----------------------------------------
# 6. Suppression des colonnes fuite
# -----------------------------------------
def drop_leaky_cols(X_train, X_test):
    # Check both cases since column names might vary
    leak_cols_possible = ['ChurnRiskCategory', 'churnriskcategory', 'CustomerType', 'customertype',
                          'RFMSegment', 'rfmsegment', 'LoyaltyLevel', 'loyaltylevel']

    # CRITICAL: Time-based features TOO correlated with churn (correlation > 0.85)
    # These predict if customer left based on time elapsed = definition of churn!
    time_based_leaky = ['recency', 'Recency', 'tenureratio', 'TenureRatio',
                        'customertenuredays', 'CustomerTenureDays',
                        'firstpurchasedaysago', 'FirstPurchaseDaysAgo',
                        'monetaryperday', 'MonetaryPerDay',
                        'monetaryperday_log', 'MonetaryPerDay_log',
                        'preferredmonth', 'PreferredMonth',
                        'favoriteseason', 'FavoriteSeason'
                        ]

    # Also remove any encoded versions (e.g., churnriskcategory_Faible, customertype_Perdu, etc.)
    leak_prefixes = ['churnriskcategory_', 'customertype_', 'rfmsegment_', 'loyaltylevel_']

    to_drop = [c for c in leak_cols_possible if c in X_train.columns]
    to_drop.extend([c for c in time_based_leaky if c in X_train.columns])

    # Find encoded columns that start with leaky prefixes
    encoded_leaky_cols = [col for col in X_train.columns
                          if any(col.lower().startswith(prefix) for prefix in leak_prefixes)]

    to_drop.extend(encoded_leaky_cols)
    to_drop = list(set(to_drop))  # Remove duplicates

    if to_drop:
        print(f"  [CRITICAL] Colonnes fuite supprimees: {len(to_drop)} colonnes")
        time_cols = [c for c in to_drop if any(t in c.lower() for t in ['recency', 'tenure', 'firstpurchase', 'monetaryperday'])]
        if time_cols:
            print(f"    - Time-based (HIGH leakage): {time_cols}")

    X_train = X_train.drop(columns=to_drop)
    X_test  = X_test.drop(columns=[c for c in to_drop if c in X_test.columns])

    # Ensure both have the same columns
    missing_in_test = set(X_train.columns) - set(X_test.columns)
    missing_in_train = set(X_test.columns) - set(X_train.columns)

    if missing_in_test:
        print(f"  [WARN] Colonnes manquantes dans X_test (ajout es avec 0) : {list(missing_in_test)[:5]}")
        for col in missing_in_test:
            X_test[col] = 0

    if missing_in_train:
        print(f"  [WARN] Colonnes manquantes dans X_train : {list(missing_in_train)[:5]}")
        X_test = X_test.drop(columns=list(missing_in_train))

    # Align column order
    X_test = X_test[X_train.columns]

    return X_train, X_test


def remove_correlated_features(X_train, X_test, threshold=0.80):
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    corr_matrix = X_train[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        print(f"  Colonnes fortement corr l es (> {threshold}) supprim es : {to_drop}")
        X_train = X_train.drop(columns=to_drop)
        X_test = X_test.drop(columns=[c for c in to_drop if c in X_test.columns])
    else:
        print(f"  Aucune colonne corr l e   plus de {threshold} d tect e.")

    return X_train, X_test


# -----------------------------------------
# 7. Encodage des variables cat gorielles
#    DOIT  tre fait AVANT le scaling
# -----------------------------------------
def encode_data(X_train, X_test):
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test  = pd.get_dummies(X_test,  columns=cat_cols, drop_first=True)

    # Aligner : X_test prend les colonnes de X_train (manquantes   0)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # Verify alignment
    if set(X_train.columns) != set(X_test.columns):
        missing_in_test = set(X_train.columns) - set(X_test.columns)
        missing_in_train = set(X_test.columns) - set(X_train.columns)
        if missing_in_test:
            print(f"  [WARN] Apr s alignement - colonnes manquantes dans X_test : {list(missing_in_test)[:5]}")
        if missing_in_train:
            print(f"  [WARN] Apr s alignement - colonnes en trop dans X_test : {list(missing_in_train)[:5]}")

    print(f"  Apr s encodage   {X_train.shape[1]} colonnes")
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test


# -----------------------------------------
# 8. Nettoyage final (inf / NaN r siduels)
# -----------------------------------------
def clean_inf_nan(X_train, X_test):
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan)

    # On remplit avec la m diane de X_train (pas de fuite)
    train_median = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_median)
    X_test  = X_test.fillna(train_median)

    return X_train, X_test


# -----------------------------------------
# 9. Scaling
#    APR S encode_data   le scaler conna t
#    toutes les colonnes (ex. 125)
# -----------------------------------------
def scale_data(X_train, X_test):
    scaler   = StandardScaler()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()

    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])

    print(f"  Scaler fit sur {len(num_cols)} colonnes num riques")
    return X_train_scaled, X_test_scaled, scaler


# -----------------------------------------
# 10. SMOTE (sur X_train scal  uniquement)
# -----------------------------------------
def apply_smote(X_train, y_train, random_state=42):
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  Apr s SMOTE : {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


# -----------------------------------------
# 11. Pipeline compl te
# -----------------------------------------
def preprocess_pipeline(path, target_col='churn'):

    feature_log = {}

    print("-" * 40)
    print("  Chargement...")
    df = load_data(path)
    feature_log["chargement"] = df.shape[1]
    print("  Shape initiale :", df.shape)
    print("  Correction des aberrants...")
    df = fix_outliers(df)
    print("  Shape apr s correction aberrants :", df.shape)
    feature_log["correction_aberrants"] = df.shape[1]
    print("  Imputation des NaN...")
    df = impute_missing(df)
    feature_log["imputation_nan"] = df.shape[1]

    print("   S paration X / y...")
    X, y = split_X_y(df, target_col)
    feature_log["split_x_y"] = X.shape[1]

    print("  Train / Test split...")
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    feature_log["train_test_split"] = X_train.shape[1]
    # 🔥 CORRÉLATION APRÈS SPLIT (important)
    corr = X_train.copy()
    corr[target_col] = y_train

    corr = corr.corr(numeric_only=True)[target_col].abs().sort_values(ascending=False)

    print("\n🔍 Corrélation avec la target :")
    print(corr.head(15))
    print("  Suppression colonnes fuite...")
    X_train, X_test = drop_leaky_cols(X_train, X_test)

    print("  Suppression des features corr l es...")
    X_train, X_test = remove_correlated_features(X_train, X_test, threshold=0.80)

    print("   Encodage (get_dummies)...")
    X_train, X_test = encode_data(X_train, X_test)
    feature_log["encodage"] = X_train.shape[1]
    print("  Nettoyage inf / NaN r siduels...")
    X_train, X_test = clean_inf_nan(X_train, X_test)
    feature_log["nettoyage_inf_nan"] = X_train.shape[1]
    print("  Scaling (apr s encodage)...")
    X_train, X_test, scaler = scale_data(X_train, X_test)
    feature_log["scaling"] = X_train.shape[1]
    print("   SMOTE...")
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    # Final verification: ensure X_train and X_test have identical columns
    print("  V rification finale...")
    if list(X_train.columns) != list(X_test.columns):
        print("  [WARN] ERREUR: X_train et X_test ont des colonnes diff rentes!")
        missing_in_test = set(X_train.columns) - set(X_test.columns)
        missing_in_train = set(X_test.columns) - set(X_train.columns)
        if missing_in_test:
            print(f"     Manquantes dans X_test: {missing_in_test}")
        if missing_in_train:
            print(f"     En trop dans X_test: {missing_in_train}")
        # Fix: align X_test to match X_train exactly
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        print(f"  [OK] R alignement effectu : {X_test.shape}")
    else:
        print(f"  [OK] Colonnes identiques: {len(X_train.columns)} features")



    # # résumé final
    # print("\n" + "="*50)
    # print("  RÉSUMÉ FEATURES PAR ÉTAPE")
    # print("="*50)
    # print(f"  Chargement initial     : {df.shape[1]} features")
    # print(f"  Après X/y split        : {X.shape[1]} features")
    # print(f"  Après anti-leakage     : {X_train.shape[1] + len([c for c in X.columns if c not in X_train.columns])} → {X_train.shape[1]} features")
    # print(f"  Après corr > 0.80      : {X_train.shape[1]} features")
    # print(f"  Après encodage OHE     : {X_train.shape[1]} features")
    # print(f"  Lignes train (SMOTE)   : {X_train_resampled.shape[0]} lignes")
    # print("="*50)
    print("\n" + "="*50)
    print("  RÉSUMÉ ÉVOLUTION FEATURES")
    print("="*50)

    for step, n in feature_log.items():
        print(f"{step:<20} : {n} features")

    print("="*50)



    print("-" * 40)
    print(f"[OK] Pipeline termin e !")
    print(f"   X_train       : {X_train.shape}")
    print(f"   X_train SMOTE : {X_train_resampled.shape}")
    print(f"   X_test        : {X_test.shape}")
    print("-" * 40)


    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_resampled,
        y_train_resampled,
        scaler
    )