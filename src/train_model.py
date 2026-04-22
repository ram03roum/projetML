import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import zscore

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    r2_score, mean_absolute_error, mean_squared_error,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from preprocessing import preprocess_pipeline


# ─────────────────────────────────────────
# fonctions d'évaluation
# ─────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy"  : accuracy_score(y_test, y_pred),
        "precision" : precision_score(y_test, y_pred, zero_division=0),
        "recall"    : recall_score(y_test, y_pred, zero_division=0),
        "f1_score"  : f1_score(y_test, y_pred, zero_division=0),
    }


def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "r2"  : r2_score(y_test, y_pred),
        "mae" : mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }


# ─────────────────────────────────────────
# visualisations
# ─────────────────────────────────────────
def save_confusion_matrix(y_true, y_pred, path='reports/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['fidèle (0)', 'churn (1)'],
                yticklabels=['fidèle (0)', 'churn (1)'])
    plt.title('matrice de confusion — random forest')
    plt.ylabel('vraie classe')
    plt.xlabel('classe prédite')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  matrice sauvegardée : {path}")


def save_feature_importance(model, feature_names, top_n=15,
                             path='reports/feature_importance.png'):
    if not hasattr(model, 'feature_importances_'):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 7))
    plt.barh(range(top_n), importances[indices], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.title(f'top {top_n} features importantes')
    plt.xlabel('importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  feature importance sauvegardée : {path}")


def save_clusters_pca(X_scaled, labels, path='reports/clusters_pca.png'):
    pca2 = PCA(n_components=2, random_state=42)
    X_2d = pca2.fit_transform(X_scaled)
    colors = ['#E24B4A', '#378ADD', '#639922', '#BA7517']
    plt.figure(figsize=(10, 7))
    for i in np.unique(labels):
        mask = labels == i
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=colors[i % len(colors)],
                    label=f'cluster {i}', alpha=0.5, s=15)
    plt.title('clusters k-means — projection acp 2d')
    plt.xlabel(f'pc1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'pc2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  clusters pca sauvegardés : {path}")


# ─────────────────────────────────────────
# étape 1 — clustering
# ─────────────────────────────────────────
def run_clustering(df_raw, n_clusters=4):
    print(f"\n{'='*60}")
    print(f"  clustering — k-means avec {n_clusters} clusters")
    print(f"{'='*60}")

    rfm_candidates = [
        'frequency', 'monetarytotal', 'monetaryavg',
        'avgdaysbetweenpurchases', 'totalquantity', 'uniqueproducts',
        'avgquantitypertransaction', 'supportticketscount', 'satisfactionscore',
        'age', 'weekendpurchaseratio', 'avgbasketvalue', 'purchaseintensity',
        'productdiversityratio',
    ]
    rfm_features = [c for c in rfm_candidates if c in df_raw.columns]
    print(f"  features rfm utilisées : {len(rfm_features)}")

    df_clust = df_raw[rfm_features].copy().fillna(df_raw[rfm_features].median())

    # suppression outliers z-score > 4
    z = np.abs(zscore(df_clust, nan_policy='omit'))
    mask = (z < 4).all(axis=1)
    df_clust = df_clust[mask].reset_index(drop=True)
    print(f"  clients après suppression outliers : {len(df_clust)}")

    # normalisation — scaler sauvegardé pour Flask
    sc = StandardScaler()
    X_scaled = sc.fit_transform(df_clust)
    joblib.dump(sc, 'models/kmeans_scaler.pkl')      # ← sauvegarde pour Flask
    joblib.dump(rfm_features, 'models/rfm_features.pkl')  # ← features pour Flask

    # acp 95% variance
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  composantes pca : {pca.n_components_} "
          f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")

    # elbow method
    inertia = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_pca)
        inertia.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 9), inertia, marker='o', color='steelblue')
    plt.title('elbow method — choix du k optimal')
    plt.xlabel('nombre de clusters (k)')
    plt.ylabel('inertie')
    plt.grid()
    plt.tight_layout()
    plt.savefig('reports/elbow_method.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  elbow method sauvegardé : reports/elbow_method.png")

    # k-means final
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n  inertie : {kmeans.inertia_:.2f}")
    print(f"  répartition des clusters :")
    for cid, cnt in zip(unique, counts):
        print(f"    cluster {cid} : {cnt} clients ({cnt/len(labels)*100:.1f}%)")

    save_clusters_pca(X_scaled, labels)

    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    joblib.dump(pca,    'models/pca_model.pkl')
    print("\n  modèles clustering sauvegardés ✅")
    print("    models/kmeans_scaler.pkl ✅")
    print("    models/rfm_features.pkl  ✅")

    return kmeans, labels, rfm_features


# ─────────────────────────────────────────
# étape 2 — classification churn
# ─────────────────────────────────────────
def run_classification(X_train, X_test, y_train, y_test,
                       X_train_smote, y_train_smote):
    print(f"\n{'='*60}")
    print("  classification — prédiction du churn")
    print(f"{'='*60}")

    print(f"\n  distribution des classes (test) :")
    for val, cnt in zip(*np.unique(y_test, return_counts=True)):
        print(f"    classe {val} : {cnt} ({cnt/len(y_test)*100:.1f}%)")

    # modèle 1 : logistic regression
    print(f"\n  entraînement logistic regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, C=0.1,
                            class_weight='balanced')
    lr.fit(X_train_smote, y_train_smote)
    lr_res = evaluate_model(lr, X_test, y_test)
    print(f"    accuracy={lr_res['accuracy']:.4f} | f1={lr_res['f1_score']:.4f}")

    # modèle 2 : random forest de base
    print(f"\n  entraînement random forest...")
    rf = RandomForestClassifier(
        max_depth=8, min_samples_leaf=5, min_samples_split=10,
        n_estimators=200, max_features='sqrt', random_state=42
    )
    rf.fit(X_train_smote, y_train_smote)
    rf_res = evaluate_model(rf, X_test, y_test)
    print(f"    accuracy={rf_res['accuracy']:.4f} | f1={rf_res['f1_score']:.4f}")

    # modèle 3 : gridsearchcv
    print(f"\n  optimisation gridsearchcv...")
    param_grid = {
        'max_depth':         [4, 6, 8, 10],
        'min_samples_leaf':  [4, 8, 12],
        'min_samples_split': [10, 20],
        'n_estimators':      [200, 300],
        'max_features':      ['sqrt', 0.4],
    }
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0
    )
    gs.fit(X_train_smote, y_train_smote)
    best_rf     = gs.best_estimator_
    best_rf_res = evaluate_model(best_rf, X_test, y_test)
    print(f"    meilleurs paramètres : {gs.best_params_}")
    print(f"    score cv             : {gs.best_score_:.4f}")
    print(f"    accuracy test        : {best_rf_res['accuracy']:.4f} | "
          f"f1 : {best_rf_res['f1_score']:.4f}")

    # tableau comparatif
    print(f"\n  {'─'*55}")
    print(f"  {'modèle':<30} {'accuracy':>10} {'f1':>10}")
    print(f"  {'─'*55}")
    print(f"  {'logistic regression':<30} "
          f"{lr_res['accuracy']:>10.4f} {lr_res['f1_score']:>10.4f}")
    print(f"  {'random forest':<30} "
          f"{rf_res['accuracy']:>10.4f} {rf_res['f1_score']:>10.4f}")
    print(f"  {'random forest (gridsearch)':<30} "
          f"{best_rf_res['accuracy']:>10.4f} {best_rf_res['f1_score']:>10.4f}")
    print(f"  {'─'*55}")

    # rapport détaillé
    y_pred = best_rf.predict(X_test)
    print(f"\n  rapport de classification :")
    print(classification_report(y_test, y_pred,
                                target_names=['fidèle (0)', 'churn (1)']))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  matrice de confusion :\n    {cm}")

    save_confusion_matrix(y_test, y_pred)
    save_feature_importance(best_rf, list(X_train.columns))

    return best_rf, "random forest (gridsearch)", best_rf_res


# ─────────────────────────────────────────
# étape 3 — régression monetarytotal
# ─────────────────────────────────────────
def run_regression(data_path):
    print(f"\n{'='*60}")
    print("  régression — prédiction des revenus (monetarytotal)")
    print(f"{'='*60}")

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()

    leak_cols = ['monetaryavg', 'monetarystd', 'monetaryperday',
                 'monetaryperday_log', 'avgbasketvalue',
                 'customerscore', 'customerscorenormalized']
    df = df.drop(columns=[c for c in leak_cols if c in df.columns])

    df = df[(df['monetarytotal'] > 0) & (df['monetarytotal'] < 20000)]
    print(f"  données après filtrage : {df.shape}")
    print(f"  montant moyen : £{df['monetarytotal'].mean():.0f}")

    features_reg = [
        'frequency', 'totalquantity', 'uniqueproducts', 'age',
        'avgdaysbetweenpurchases', 'supportticketscount',
        'satisfactionscore', 'avgquantitypertransaction',
    ]
    features_reg = [f for f in features_reg if f in df.columns]
    print(f"  features utilisées ({len(features_reg)}) : {features_reg}")

    X = df[features_reg].fillna(0)
    y = df['monetarytotal']

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42)

    lr_reg = LinearRegression()
    lr_reg.fit(X_tr, y_tr)
    lr_res = evaluate_regression(lr_reg, X_te, y_te)

    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_tr, y_tr)
    rf_res = evaluate_regression(rf_reg, X_te, y_te)

    print(f"\n  {'─'*55}")
    print(f"  {'modèle':<25} {'r²':>8} {'mae':>12} {'rmse':>12}")
    print(f"  {'─'*55}")
    print(f"  {'linear regression':<25} {lr_res['r2']:>8.4f} "
          f"£{lr_res['mae']:>10.1f} £{lr_res['rmse']:>10.1f}")
    print(f"  {'random forest':<25} {rf_res['r2']:>8.4f} "
          f"£{rf_res['mae']:>10.1f} £{rf_res['rmse']:>10.1f}")
    print(f"  {'─'*55}")

    best_reg = rf_reg if rf_res['r2'] > lr_res['r2'] else lr_reg
    best_res = rf_res if rf_res['r2'] > lr_res['r2'] else lr_res

    y_pred_reg = best_reg.predict(X_te)
    errors     = y_te.values - y_pred_reg

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_te, y_pred_reg, alpha=0.4, color='steelblue', s=10)
    axes[0].plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'r--')
    axes[0].set_title('valeurs réelles vs prédites')
    axes[0].set_xlabel('réel (£)')
    axes[0].set_ylabel('prédit (£)')
    axes[1].hist(errors, bins=50, color='coral', edgecolor='white')
    axes[1].axvline(0, color='black', linestyle='--')
    axes[1].set_title('distribution des erreurs')
    axes[1].set_xlabel('erreur (£)')
    plt.tight_layout()
    plt.savefig('reports/regression_results.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  graphiques régression sauvegardés : reports/regression_results.png")

    joblib.dump(best_reg, 'models/regression_model.pkl')
    print("  modèle régression sauvegardé ✅")

    return best_reg, best_res


# ─────────────────────────────────────────
# étape 4 — modèles Flask
# ─────────────────────────────────────────
def run_flask_models(data_path):
    print(f"\n{'='*60}")
    print("  modèles flask — entraînement")
    print(f"{'='*60}")

    FLASK_CLF_FEATURES = [
        'frequency', 'purchaseintensity', 'uniqueinvoices',
        'regyear', 'regmonth_sin', 'regmonth_cos',
        'uniqueproducts', 'avgdaysbetweenpurchases',
    ]

    FLASK_REG_FEATURES = [
        'frequency', 'uniqueinvoices', 'uniqueproducts',
        'avgdaysbetweenpurchases', 'avgbasketvalue',
        'purchaseintensity', 'satisfactionscore', 'monetarytotal',
    ]

    df_fl = pd.read_csv(data_path)
    df_fl.columns = df_fl.columns.str.lower()

    # nettoyage
    df_fl['satisfactionscore'] = df_fl['satisfactionscore'].replace(
        [-1, 99], np.nan)
    df_fl['satisfactionscore'] = df_fl['satisfactionscore'].fillna(
        df_fl['satisfactionscore'].median())
    df_fl['monetarytotal'] = df_fl['monetarytotal'].clip(lower=0)

    # vérification features disponibles
    FLASK_CLF_FEATURES = [f for f in FLASK_CLF_FEATURES if f in df_fl.columns]
    FLASK_REG_FEATURES = [f for f in FLASK_REG_FEATURES if f in df_fl.columns]

    df_fl = df_fl.dropna(subset=FLASK_CLF_FEATURES + ['churn'])

    # ── classification ──
    print("\n  [1] classification churn...")
    X_clf = df_fl[FLASK_CLF_FEATURES]
    y_clf = df_fl['churn']

    X_clf_tr, X_clf_te, y_clf_tr, y_clf_te = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    scaler_clf_fl = StandardScaler()
    X_clf_tr_sc   = scaler_clf_fl.fit_transform(X_clf_tr)
    X_clf_te_sc   = scaler_clf_fl.transform(X_clf_te)

    sm_fl = SMOTE(random_state=42)
    X_clf_res, y_clf_res = sm_fl.fit_resample(X_clf_tr_sc, y_clf_tr)

    rf_clf_fl = RandomForestClassifier(
        max_depth=8, min_samples_leaf=4, min_samples_split=10,
        n_estimators=200, random_state=42, n_jobs=-1
    )
    rf_clf_fl.fit(X_clf_res, y_clf_res)

    y_clf_pred_fl = rf_clf_fl.predict(X_clf_te_sc)
    print(f"    accuracy  : {accuracy_score(y_clf_te, y_clf_pred_fl):.4f}")
    print(f"    f1-score  : {f1_score(y_clf_te, y_clf_pred_fl, zero_division=0):.4f}")
    print(f"    precision : {precision_score(y_clf_te, y_clf_pred_fl, zero_division=0):.4f}")
    print(f"    recall    : {recall_score(y_clf_te, y_clf_pred_fl, zero_division=0):.4f}")

    # ── régression ──
    print("\n  [2] régression revenu...")
    df_reg_fl = df_fl[
        (df_fl['monetarytotal'] > 0) &
        (df_fl['monetarytotal'] < 20000)
    ].copy().dropna(subset=FLASK_REG_FEATURES)

    X_reg = df_reg_fl[FLASK_REG_FEATURES]
    y_reg = df_reg_fl['monetarytotal']

    X_reg_tr, X_reg_te, y_reg_tr, y_reg_te = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)

    scaler_reg_fl = StandardScaler()
    X_reg_tr_sc   = scaler_reg_fl.fit_transform(X_reg_tr)
    X_reg_te_sc   = scaler_reg_fl.transform(X_reg_te)

    rf_reg_fl = RandomForestRegressor(
        n_estimators=200, max_depth=10,
        min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    rf_reg_fl.fit(X_reg_tr_sc, y_reg_tr)

    y_reg_pred_fl = rf_reg_fl.predict(X_reg_te_sc)
    print(f"    r²   : {r2_score(y_reg_te, y_reg_pred_fl):.4f}")
    print(f"    mae  : £{mean_absolute_error(y_reg_te, y_reg_pred_fl):.1f}")
    print(f"    rmse : £{np.sqrt(mean_squared_error(y_reg_te, y_reg_pred_fl)):.1f}")

    # ── sauvegarde ──
    joblib.dump(rf_clf_fl,     'models/flask_churn_model.pkl')
    joblib.dump(scaler_clf_fl, 'models/flask_scaler_clf.pkl')
    joblib.dump(rf_reg_fl,     'models/flask_reg_model.pkl')
    joblib.dump(scaler_reg_fl, 'models/flask_scaler_reg.pkl')

    flask_config = {
        'clf_features': FLASK_CLF_FEATURES,
        'reg_features': FLASK_REG_FEATURES
    }
    with open('models/flask_config.json', 'w') as f:
        json.dump(flask_config, f, indent=2)

    print("\n  modèles Flask sauvegardés ✅")
    for f in ['flask_churn_model.pkl', 'flask_scaler_clf.pkl',
              'flask_reg_model.pkl', 'flask_scaler_reg.pkl',
              'flask_config.json']:
        print(f"    models/{f}")


# ─────────────────────────────────────────
# main
# ─────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "="*60)
    print("  pipeline machine learning complet")
    print("="*60)

    for folder in ['models', 'reports', 'data/train_test']:
        Path(folder).mkdir(parents=True, exist_ok=True)

    DATA_PATH = 'data/processed/feature_engineering.csv'

    # ── étape 1 : prétraitement ──
    print("\n  étape 1 : prétraitement")
    print("-"*60)
    X_train, X_test, y_train, y_test, \
    X_train_smote, y_train_smote, scaler = preprocess_pipeline(DATA_PATH)

    X_train.to_csv('data/train_test/X_train.csv', index=False)
    X_test.to_csv('data/train_test/X_test.csv',   index=False)
    y_train.to_csv('data/train_test/y_train.csv', index=False)
    y_test.to_csv('data/train_test/y_test.csv',   index=False)
    print(f"  données sauvegardées ({X_train.shape[1]} features) ✅")

    # ── étape 2 : clustering ──
    print("\n  étape 2 : clustering")
    print("-"*60)
    df_raw = pd.read_csv(DATA_PATH)
    df_raw.columns = df_raw.columns.str.lower()
    kmeans, labels, rfm_features = run_clustering(df_raw, n_clusters=4)

    # ── étape 3 : classification ──
    print("\n  étape 3 : classification")
    print("-"*60)
    best_model, best_name, clf_res = run_classification(
        X_train, X_test, y_train, y_test,
        X_train_smote, y_train_smote
    )
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(scaler,     'models/scaler.pkl')
    print("  modèle + scaler sauvegardés ✅")

    # ── étape 4 : régression ──
    print("\n  étape 4 : régression")
    print("-"*60)
    reg_model, reg_res = run_regression(DATA_PATH)

    # ── étape 5 : modèles Flask ──
    print("\n  étape 5 : modèles flask")
    print("-"*60)
    run_flask_models(DATA_PATH)

    # ── résumé final ──
    print("\n" + "="*60)
    print("  résumé des résultats")
    print("="*60)

    print(f"\n  classification :")
    print(f"    modèle    : {best_name}")
    print(f"    accuracy  : {clf_res['accuracy']*100:.2f}%")
    print(f"    f1-score  : {clf_res['f1_score']:.4f}")
    print(f"    precision : {clf_res['precision']:.4f}")
    print(f"    recall    : {clf_res['recall']:.4f}")

    print(f"\n  régression :")
    print(f"    r²   : {reg_res['r2']:.4f} ({reg_res['r2']*100:.1f}%)")
    print(f"    mae  : £{reg_res['mae']:.1f}")
    print(f"    rmse : £{reg_res['rmse']:.1f}")

    print(f"\n  clustering :")
    print(f"    4 segments clients identifiés ✅")

    print(f"\n  fichiers générés :")
    for f in [
        'models/best_model.pkl',        'models/scaler.pkl',
        'models/regression_model.pkl',  'models/kmeans_model.pkl',
        'models/pca_model.pkl',         'models/kmeans_scaler.pkl',
        'models/rfm_features.pkl',      'models/flask_churn_model.pkl',
        'models/flask_scaler_clf.pkl',  'models/flask_reg_model.pkl',
        'models/flask_scaler_reg.pkl',  'models/flask_config.json',
        'data/train_test/X_train.csv',
        'reports/confusion_matrix.png', 'reports/feature_importance.png',
        'reports/clusters_pca.png',     'reports/elbow_method.png',
        'reports/regression_results.png',
    ]:
        print(f"    {'✅' if Path(f).exists() else '❌'} {f}")

    print("\n" + "="*60)
    print("  projet terminé avec succès !")
    print("="*60 + "\n")

    # ── diagnostic overfitting ──
    print("  diagnostic overfitting — best_model :")
    print(f"    Train score      : {best_model.score(X_train_smote, y_train_smote):.4f}")
    print(f"    Test score       : {best_model.score(X_test, y_test):.4f}")
    print(f"    Profondeur max   : {max(t.get_depth() for t in best_model.estimators_)}")
    print(f"    Feuilles moyenne : {np.mean([t.get_n_leaves() for t in best_model.estimators_]):.0f}")
    