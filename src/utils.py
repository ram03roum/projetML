"""
utils.py — fonctions utilitaires réutilisables
utilisé par : preprocessing.py, train_model.py, notebooks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────
# 1. chargement des données
# ─────────────────────────────────────────
def load_data(path):
    """charge un fichier csv et retourne un dataframe."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    print(f"  données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


# ─────────────────────────────────────────
# 2. diagnostic rapide
# ─────────────────────────────────────────
def diagnostic(df):
    """affiche un diagnostic rapide du dataset."""
    print("=" * 50)
    print(f"shape          : {df.shape}")
    print(f"nan totaux     : {df.isnull().sum().sum()}")
    print(f"doublons       : {df.duplicated().sum()}")
    print("=" * 50)

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print("\ncolonnes avec nan :")
        for col, n in missing.items():
            print(f"  {col}: {n} ({n/len(df)*100:.1f}%)")
    else:
        print("aucune valeur manquante ✅")

    return missing


# ─────────────────────────────────────────
# 3. heatmap de corrélation
# ─────────────────────────────────────────
def plot_correlation_heatmap(df, target=None, threshold=0.8,
                              figsize=(14, 10), save_path=None):
    """
    affiche la heatmap de corrélation.
    si target est fourni, affiche aussi la corrélation avec la target.
    """
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    corr = df[num_cols].corr()

    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm',
                center=0, square=True, linewidths=0.5)
    plt.title("heatmap de corrélation")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  heatmap sauvegardée : {save_path}")
    plt.show()

    # corrélation avec la target si fournie
    if target and target in df.columns:
        corr_target = df[num_cols].corr()[target].abs().sort_values(ascending=False)
        print(f"\ntop 10 features corrélées avec '{target}' :")
        print(corr_target.head(11))

    return corr


# ─────────────────────────────────────────
# 4. identifier les paires fortement corrélées
# ─────────────────────────────────────────
def get_high_correlation_pairs(df, threshold=0.8, exclude=None):
    """
    retourne les paires de features avec corrélation > threshold.
    exclude : liste de colonnes à ignorer (ex: ['churn', 'customerid'])
    """
    if exclude is None:
        exclude = []

    num_cols = [c for c in df.select_dtypes(include=['int64', 'float64']).columns
                if c not in exclude]

    corr_matrix = df[num_cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    pairs = []
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if val > threshold:
                pairs.append({
                    'feature_1': idx,
                    'feature_2': col,
                    'correlation': round(val, 4)
                })

    df_pairs = pd.DataFrame(pairs).sort_values('correlation', ascending=False)

    print(f"\npaires avec corrélation > {threshold} : {len(df_pairs)}")
    if len(df_pairs) > 0:
        print(df_pairs.to_string(index=False))

    return df_pairs


# ─────────────────────────────────────────
# 5. parsing de l'adresse ip
# ─────────────────────────────────────────
def parse_ip(ip_series):
    """
    extrait des features depuis une série d'adresses ip.
    retourne un dataframe avec ip_firstoctet et ip_isprivate.
    """
    result = pd.DataFrame()

    # premier octet
    result['ip_firstoctet'] = ip_series.str.extract(r'^(\d+)').astype(float)

    # ip privée (1) ou publique (0)
    def _is_private(ip):
        try:
            parts = str(ip).split('.')
            first  = int(parts[0])
            second = int(parts[1])
            return int(
                first == 10 or
                (first == 172 and 16 <= second <= 31) or
                (first == 192 and second == 168)
            )
        except:
            return 0

    result['ip_isprivate'] = ip_series.apply(_is_private)

    print(f"  ip parsée → ip_firstoctet + ip_isprivate ✅")
    print(f"  ip privées : {result['ip_isprivate'].sum()} / {len(result)}")

    return result


# ─────────────────────────────────────────
# 6. affichage des profils de clusters
# ─────────────────────────────────────────
def print_cluster_profiles(df, cluster_col='cluster',
                            rfm_cols=None, churn_col='churn'):
    """
    affiche le profil rfm de chaque cluster.
    utilisé après k-means pour interpréter les segments.
    """
    if rfm_cols is None:
        rfm_cols = ['recency', 'frequency', 'monetarytotal']

    rfm_cols = [c for c in rfm_cols if c in df.columns]
    if churn_col in df.columns:
        rfm_cols.append(churn_col)

    profiles = df.groupby(cluster_col)[rfm_cols].mean().round(1)
    profiles['nb_clients'] = df[cluster_col].value_counts().sort_index()
    profiles['pct_%']      = (profiles['nb_clients'] / len(df) * 100).round(1)

    if churn_col in profiles.columns:
        profiles['churn_%'] = (profiles[churn_col] * 100).round(1)

    print("\n=== profils des clusters ===")
    print(profiles.to_string())

    return profiles


# ─────────────────────────────────────────
# 7. visualisation distribution churn
# ─────────────────────────────────────────
def plot_churn_distribution(y, title="distribution du churn", save_path=None):
    """affiche la distribution des classes churn."""
    counts = pd.Series(y).value_counts()

    plt.figure(figsize=(6, 4))
    bars = plt.bar(['fidèles (0)', 'partis (1)'], counts.values,
                   color=['steelblue', 'coral'])
    plt.title(title)
    plt.ylabel("nombre de clients")

    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 10,
                 f"{val}\n({val/sum(counts.values)*100:.1f}%)",
                 ha='center', fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  graphique sauvegardé : {save_path}")
    plt.show()


# ─────────────────────────────────────────
# 8. rapport de qualité des données
# ─────────────────────────────────────────
def data_quality_report(df):
    """génère un rapport complet de qualité des données."""
    report = pd.DataFrame({
        'type'        : df.dtypes,
        'nan_count'   : df.isnull().sum(),
        'nan_%'       : (df.isnull().sum() / len(df) * 100).round(2),
        'unique'      : df.nunique(),
        'unique_%'    : (df.nunique() / len(df) * 100).round(2),
    })

    # détecter les colonnes constantes
    report['constante'] = report['unique'] == 1

    # détecter les colonnes quasi-constantes (>95% même valeur)
    report['quasi_constante'] = report['unique_%'] < 5

    print("=== rapport qualité des données ===")
    print(report.to_string())

    print(f"\ncolonnes constantes      : {report['constante'].sum()}")
    print(f"colonnes quasi-constantes : {report['quasi_constante'].sum()}")
    print(f"colonnes avec >30% nan   : {(report['nan_%'] > 30).sum()}")

    return report


# ─────────────────────────────────────────
# 9. vérification data leakage
# ─────────────────────────────────────────
def check_leakage(df, target='churn', threshold=0.8):
    """
    détecte les colonnes trop corrélées avec la target.
    corrélation > threshold → fuite de données probable.
    """
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if target not in num_cols:
        print(f"target '{target}' non trouvée dans les colonnes numériques")
        return []

    corr_target = df[num_cols].corr()[target].abs().sort_values(ascending=False)
    leaky = corr_target[corr_target > threshold].index.tolist()
    leaky = [c for c in leaky if c != target]

    if leaky:
        print(f"\n⚠️ colonnes suspectes (corrélation > {threshold} avec {target}) :")
        for col in leaky:
            print(f"   {col} : {corr_target[col]:.3f}")
    else:
        print(f"✅ aucune fuite détectée (seuil {threshold})")

    return leaky