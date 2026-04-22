# # # # from flask import Flask, request, render_template
# # # # import joblib
# # # # import numpy as np
# # # # import pandas as pd
# # # # import os, sys

# # # # # Ajout du dossier racine au path pour les imports
# # # # BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # # # sys.path.insert(0, BASE_DIR)

# # # # app = Flask(__name__, template_folder='templates')

# # # # # ─── Chargement du SIMPLE model (7 features — sans data leakage) ────────────
# # # # MODEL_PATH  = os.path.join(BASE_DIR, "models", "simple_model.pkl")
# # # # SCALER_PATH = os.path.join(BASE_DIR, "models", "simple_scaler.pkl")

# # # # model  = joblib.load(MODEL_PATH)
# # # # scaler = joblib.load(SCALER_PATH)

# # # # # Les 7 features exactes du modèle (ordre important)
# # # # FEATURE_NAMES = [
# # # #     'age',
# # # #     'frequency',
# # # #     'monetarytotal',
# # # #     'totaltransactions',
# # # #     'weekendpurchaseratio',
# # # #     'avgquantitypertransaction',
# # # #     'avgbasketvalue',
# # # # ]

# # # # # Médianes de référence pour les champs non renseignés
# # # # df_raw = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "step3_feature_engineering.csv"))
# # # # df_raw.columns = df_raw.columns.str.lower()
# # # # FEATURE_MEDIANS = {f: float(df_raw[f].median()) for f in FEATURE_NAMES if f in df_raw.columns}

# # # # print(f"[OK] Simple model charge ({len(FEATURE_NAMES)} features, sans recency)")
# # # # print(f"[OK] Features: {FEATURE_NAMES}")


# # # # # ─── Helpers ─────────────────────────────────────────────────────────────────
# # # # def get_float(form, key, default=None):
# # # #     val = form.get(key, '').strip()
# # # #     try:
# # # #         return float(val) if val != '' else default
# # # #     except ValueError:
# # # #         return default


# # # # def get_recommendation(proba: float):
# # # #     if proba >= 70:
# # # #         return "Élevé", (
# # # #             "🔴 Action immédiate requise : ce client est très susceptible de partir. "
# # # #             "Contactez-le avec une offre exclusive (réduction, programme VIP, appel personnalisé)."
# # # #         )
# # # #     elif proba >= 45:
# # # #         return "Moyen", (
# # # #             "🟡 Risque modéré : envoyez un email de réengagement avec une promotion ciblée. "
# # # #             "Surveillez l'activité de ce client dans les prochaines semaines."
# # # #         )
# # # #     else:
# # # #         return "Faible", (
# # # #             "🟢 Client fidèle : maintenez la relation avec des communications régulières "
# # # #             "et un programme de récompenses. Aucune action urgente nécessaire."
# # # #         )


# # # # # ─── Routes ──────────────────────────────────────────────────────────────────
# # # # @app.route('/')
# # # # def home():
# # # #     return render_template('index.html', prediction=None, proba=None,
# # # #                            error=None, niveau=None, conseil=None)


# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     try:
# # # #         # Lecture des champs du formulaire
# # # #         age            = get_float(request.form, 'age')
# # # #         frequency      = get_float(request.form, 'frequency')
# # # #         monetary_total = get_float(request.form, 'monetary_total')
# # # #         total_trans    = get_float(request.form, 'total_transactions')
# # # #         weekend_ratio  = get_float(request.form, 'weekend_ratio')
# # # #         avg_qty_trans  = get_float(request.form, 'avg_qty_per_trans')

# # # #         # Calcul automatique de avgbasketvalue si possible
# # # #         if monetary_total is not None and frequency is not None and frequency > 0:
# # # #             avg_basket = monetary_total / frequency
# # # #         else:
# # # #             avg_basket = FEATURE_MEDIANS.get('avgbasketvalue', 100.0)

# # # #         # Dictionnaire des valeurs utilisateur
# # # #         user_values = {
# # # #             'age':                     age,
# # # #             'frequency':               frequency,
# # # #             'monetarytotal':           monetary_total,
# # # #             'totaltransactions':       total_trans,
# # # #             'weekendpurchaseratio':    weekend_ratio,
# # # #             'avgquantitypertransaction': avg_qty_trans,
# # # #             'avgbasketvalue':          avg_basket,
# # # #         }

# # # #         # Construction du vecteur de features (médiane si champ non renseigné)
# # # #         input_data = {}
# # # #         for feat in FEATURE_NAMES:
# # # #             val = user_values.get(feat)
# # # #             if val is not None:
# # # #                 input_data[feat] = val
# # # #             else:
# # # #                 input_data[feat] = FEATURE_MEDIANS.get(feat, 0.0)

# # # #         print(f"\n[INPUT] {input_data}")

# # # #         input_df = pd.DataFrame([input_data])[FEATURE_NAMES]

# # # #         # Scaling
# # # #         input_scaled = scaler.transform(input_df)

# # # #         # Prédiction
# # # #         prediction = int(model.predict(input_scaled)[0])
# # # #         proba      = round(float(model.predict_proba(input_scaled)[0][1]) * 100, 1)

# # # #         print(f"[RESULT] class={prediction}, proba={proba}%")

# # # #         niveau, conseil = get_recommendation(proba)

# # # #         return render_template('index.html',
# # # #                                prediction=prediction,
# # # #                                proba=proba,
# # # #                                error=None,
# # # #                                niveau=niveau,
# # # #                                conseil=conseil)

# # # #     except Exception as e:
# # # #         import traceback
# # # #         traceback.print_exc()
# # # #         return render_template('index.html',
# # # #                                prediction=None, proba=None,
# # # #                                error=str(e), niveau=None, conseil=None)


# # # # if __name__ == "__main__":
# # # #     app.run(debug=True)


# # # from flask import Flask, render_template, request, jsonify
# # # import joblib
# # # import numpy as np
# # # import pandas as pd
# # # import os

# # # app = Flask(__name__)

# # # BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # # model_clf = joblib.load(os.path.join(BASE, 'models/best_model.pkl'))
# # # scaler    = joblib.load(os.path.join(BASE, 'models/scaler.pkl'))
# # # model_reg = joblib.load(os.path.join(BASE, 'models/regression_model.pkl'))
# # # model_km  = joblib.load(os.path.join(BASE, 'models/kmeans_model.pkl'))
# # # model_pca = joblib.load(os.path.join(BASE, 'models/pca_model.pkl'))

# # # REG_FEATURES = [
# # #     'frequency', 'totalquantity', 'uniqueproducts', 'age',
# # #     'avgdaysbetweenpurchases', 'supportticketscount',
# # #     'satisfactionscore', 'avgquantitypertransaction'
# # # ]

# # # RFM_FEATURES = [
# # #     'frequency', 'monetarytotal', 'monetaryavg',
# # #     'avgdaysbetweenpurchases', 'totalquantity', 'uniqueproducts',
# # #     'avgquantitypertransaction', 'supportticketscount',
# # #     'satisfactionscore', 'age', 'weekendpurchaseratio',
# # #     'avgbasketvalue', 'purchaseintensity', 'productdiversityratio'
# # # ]

# # # CLUSTER_NAMES = {
# # #     0: ("Client VIP",        "Très actif, haute valeur. Priorité de fidélisation maximale."),
# # #     1: ("Client Régulier",   "Fréquence d'achat stable. Bon potentiel de croissance."),
# # #     2: ("Client Occasionnel","Achète peu souvent. Actions marketing ciblées recommandées."),
# # #     3: ("Client Inactif",    "Peu actif, risque de churn élevé. Campagne de réengagement.")
# # # }

# # # @app.route('/')
# # # def index():
# # #     return render_template('index.html')

# # # @app.route('/predict', methods=['POST'])
# # # def predict():
# # #     try:
# # #         d = request.get_json()

# # #         # ── valeurs du formulaire ──
# # #         frequency    = float(d['frequency'])
# # #         totalqty     = float(d['totalquantity'])
# # #         uniqueprod   = float(d['uniqueproducts'])
# # #         age          = float(d['age'])
# # #         avgdays      = float(d['avgdaysbetweenpurchases'])
# # #         tickets      = float(d['supportticketscount'])
# # #         satisfaction = float(d['satisfactionscore'])
# # #         avgqty       = float(d['avgquantitypertransaction'])
# # #         monetarytotal= float(d['monetarytotal'])
# # #         monetaryavg  = float(d['monetaryavg'])
# # #         weekend      = float(d['weekendpurchaseratio'])
# # #         returnratio  = float(d.get('returnratio', 0.1))

# # #         # ── 1. RÉGRESSION ──
# # #         X_reg = pd.DataFrame([{
# # #             'frequency': frequency, 'totalquantity': totalqty,
# # #             'uniqueproducts': uniqueprod, 'age': age,
# # #             'avgdaysbetweenpurchases': avgdays,
# # #             'supportticketscount': tickets,
# # #             'satisfactionscore': satisfaction,
# # #             'avgquantitypertransaction': avgqty
# # #         }])[REG_FEATURES]
# # #         revenu = max(0, float(model_reg.predict(X_reg)[0]))

# # #         # ── 2. CLUSTERING ──
# # #         avgbasket   = monetarytotal / (frequency + 1)
# # #         intensity   = frequency / 365
# # #         diversity   = uniqueprod / (frequency + 1)

# # #         rfm_data = {
# # #             'frequency': frequency, 'monetarytotal': monetarytotal,
# # #             'monetaryavg': monetaryavg, 'avgdaysbetweenpurchases': avgdays,
# # #             'totalquantity': totalqty, 'uniqueproducts': uniqueprod,
# # #             'avgquantitypertransaction': avgqty, 'supportticketscount': tickets,
# # #             'satisfactionscore': satisfaction, 'age': age,
# # #             'weekendpurchaseratio': weekend, 'avgbasketvalue': avgbasket,
# # #             'purchaseintensity': intensity, 'productdiversityratio': diversity
# # #         }
# # #         available = [f for f in RFM_FEATURES if f in rfm_data]
# # #         X_rfm = pd.DataFrame([rfm_data])[available]

# # #         from sklearn.preprocessing import StandardScaler as SS
# # #         sc = SS()
# # #         X_scaled = sc.fit_transform(X_rfm)
# # #         X_pca    = model_pca.transform(X_scaled)
# # #         n = model_km.cluster_centers_.shape[1]
# # #         if X_pca.shape[1] > n:
# # #             X_pca = X_pca[:, :n]
# # #         elif X_pca.shape[1] < n:
# # #             X_pca = np.hstack([X_pca, np.zeros((1, n - X_pca.shape[1]))])

# # #         cluster      = int(model_km.predict(X_pca)[0])
# # #         cluster_name, cluster_desc = CLUSTER_NAMES.get(cluster, (f"Cluster {cluster}", ""))

# # #         # ── 3. CHURN (règles métier basées sur les features importantes) ──
# # #         score = 0.0
# # #         if frequency        <  3:   score += 0.30
# # #         if avgdays          > 60:   score += 0.25
# # #         if satisfaction     <  2.5: score += 0.20
# # #         if returnratio      >  0.3: score += 0.15
# # #         if tickets          >  5:   score += 0.10
# # #         score     = min(score, 1.0)
# # #         churn_pred = 1 if score >= 0.5 else 0

# # #         return jsonify({
# # #             'status': 'success',
# # #             'churn': {
# # #                 'prediction': churn_pred,
# # #                 'probabilite': round(score * 100, 1),
# # #                 'label': 'Risque de Churn ⚠️' if churn_pred == 1 else 'Client Fidèle ✅'
# # #             },
# # #             'revenu': {
# # #                 'prediction': round(revenu, 2)
# # #             },
# # #             'cluster': {
# # #                 'id': cluster,
# # #                 'nom': cluster_name,
# # #                 'description': cluster_desc
# # #             }
# # #         })

# # #     except Exception as e:
# # #         return jsonify({'status': 'error', 'message': str(e)})

# # # if __name__ == '__main__':
# # #     app.run(debug=True, port=5000)
# # # from flask import Flask, render_template, request, jsonify
# # # import joblib
# # # import pandas as pd
# # # import numpy as np
# # # import os

# # # app = Flask(__name__)

# # # # -----------------------------------------
# # # # Chargement des modèles
# # # # -----------------------------------------
# # # BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # # model_clf = joblib.load(os.path.join(BASE, 'models/best_model.pkl'))
# # # scaler    = joblib.load(os.path.join(BASE, 'models/scaler.pkl'))
# # # model_columns = joblib.load(os.path.join(BASE, 'models/model_columns.pkl'))

# # # # Mapping colonnes (insensible à la casse)
# # # column_mapping = {c.lower(): c for c in model_columns}

# # # print(f"[OK] Model chargé avec {len(model_columns)} features")


# # # # -----------------------------------------
# # # @app.route('/')
# # # def index():
# # #     return render_template('index.html')


# # # # -----------------------------------------
# # # @app.route('/predict', methods=['POST'])
# # # def predict():
# # #     try:
# # #         d = request.get_json()

# # #         # =====================================================
# # #         # INPUT UTILISATEUR (SIMPLE)
# # #         # =====================================================
# # #         raw_inputs = {
# # #             'frequency': float(d.get('frequency', 0)),
# # #             'totalquantity': float(d.get('totalquantity', 0)),
# # #             'uniqueproducts': float(d.get('uniqueproducts', 0)),
# # #             'age': float(d.get('age', 0)),
# # #             'monetarytotal': float(d.get('monetarytotal', 0)),
# # #             'satisfactionscore': float(d.get('satisfactionscore', 0))
# # #         }

# # #         # =====================================================
# # #         # CRÉER DATAFRAME COMPLET (IMPORTANT)
# # #         # =====================================================
# # #         df_input = pd.DataFrame(0, index=[0], columns=model_columns)

# # #         # Injecter les valeurs utilisateur
# # #         for key, val in raw_inputs.items():
# # #             target_col = column_mapping.get(key.lower())
# # #             if target_col:
# # #                 df_input[target_col] = val

# # #         # =====================================================
# # #         # SCALING
# # #         # =====================================================
# # #         X_scaled = scaler.transform(df_input)

# # #         # =====================================================
# # #         # PRÉDICTION
# # #         # =====================================================
# # #         prob = float(model_clf.predict_proba(X_scaled)[0][1])
# # #         pred = int(model_clf.predict(X_scaled)[0])

# # #         # =====================================================
# # #         # INTERPRÉTATION MÉTIER
# # #         # =====================================================
# # #         if prob >= 0.7:
# # #             niveau = "🔴 Risque élevé"
# # #             conseil = "Contacter le client immédiatement"
# # #         elif prob >= 0.4:
# # #             niveau = "🟡 Risque modéré"
# # #             conseil = "Envoyer une promotion ciblée"
# # #         else:
# # #             niveau = "🟢 Risque faible"
# # #             conseil = "Maintenir la fidélisation"

# # #         # =====================================================
# # #         return jsonify({
# # #             'status': 'success',
# # #             'prediction': pred,
# # #             'probabilite': round(prob * 100, 1),
# # #             'niveau': niveau,
# # #             'conseil': conseil
# # #         })

# # #     except Exception as e:
# # #         return jsonify({'status': 'error', 'message': str(e)})


# # # # -----------------------------------------
# # # if __name__ == '__main__':
# # #     app.run(debug=True)


# # from flask import Flask, render_template, request, jsonify
# # import joblib
# # import numpy as np
# # import pandas as pd
# # import json
# # import os
# # from sklearn.preprocessing import StandardScaler

# # app = Flask(__name__)

# # BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # def load_model(name):
# #     return joblib.load(os.path.join(BASE, 'models', name))

# # # ── chargement des modèles ──
# # churn_model  = load_model('flask_churn_model.pkl')
# # scaler_clf   = load_model('flask_scaler_clf.pkl')
# # reg_model    = load_model('flask_reg_model.pkl')
# # scaler_reg   = load_model('flask_scaler_reg.pkl')
# # kmeans       = load_model('kmeans_model.pkl')
# # pca          = load_model('pca_model.pkl')
# # kmeans_scaler = load_model('kmeans_scaler.pkl')

# # with open(os.path.join(BASE, 'models', 'flask_config.json')) as f:
# #     config = json.load(f)

# # CLF_FEATURES = config['clf_features']
# # REG_FEATURES = config['reg_features']

# # RFM_FEATURES = [
# #     'frequency', 'monetarytotal', 'monetaryavg',
# #     'avgdaysbetweenpurchases', 'uniqueproducts',
# #     'avgquantitypertransaction', 'satisfactionscore',
# #     'avgbasketvalue', 'purchaseintensity', 'productdiversityratio'
# # ]

# # CLUSTER_NAMES = {
# #     0: ("Client VIP",         "Très actif, haute valeur. Fidélisation prioritaire."),
# #     1: ("Client Régulier",    "Fréquence stable. Bon potentiel de croissance."),
# #     2: ("Client Occasionnel", "Achète peu souvent. Marketing ciblé recommandé."),
# #     3: ("Client Inactif",     "Peu actif. Risque churn élevé. Réengagement urgent.")
# # }


# # @app.route('/')
# # def index():
# #     return render_template('index.html')


# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     try:
# #         d = request.get_json()

# #         # ── lecture des 8 champs saisis ──
# #         frequency     = float(d['frequency'])
# #         monetarytotal = float(d['monetarytotal'])
# #         uniqueinvoices= float(d['uniqueinvoices'])
# #         uniqueproducts= float(d['uniqueproducts'])
# #         avgdays       = float(d['avgdaysbetweenpurchases'])
# #         satisfaction  = float(d['satisfactionscore'])
# #         regyear       = float(d['regyear'])
# #         regmonth      = float(d['regmonth'])

# #         # ── features calculées automatiquement ──
# #         customertenuredays = max(1, (2025 - int(regyear)) * 365)
# #         purchaseintensity  = frequency / (customertenuredays + 1)
# #         avgbasketvalue     = monetarytotal / (frequency + 1)
# #         avgqty             = (uniqueinvoices * 5) / (frequency + 1)
# #         diversity          = uniqueproducts / (frequency + 1)
# #         regmonth_sin       = float(np.sin(2 * np.pi * regmonth / 12))
# #         regmonth_cos       = float(np.cos(2 * np.pi * regmonth / 12))

# #         # ════════════════════════════════
# #         # 1. CLASSIFICATION — CHURN
# #         # ════════════════════════════════
# #         clf_input = {
# #             'frequency':               frequency,
# #             'purchaseintensity':       purchaseintensity,
# #             'uniqueinvoices':          uniqueinvoices,
# #             'regyear':                 regyear,
# #             'regmonth_sin':            regmonth_sin,
# #             'regmonth_cos':            regmonth_cos,
# #             'uniqueproducts':          uniqueproducts,
# #             'avgdaysbetweenpurchases': avgdays,
# #         }
# #         X_clf    = pd.DataFrame([clf_input])[CLF_FEATURES]
# #         X_clf_sc = scaler_clf.transform(X_clf)

# #         churn_pred = int(churn_model.predict(X_clf_sc)[0])
# #         churn_prob = float(churn_model.predict_proba(X_clf_sc)[0][1])

# #         # ════════════════════════════════
# #         # 2. RÉGRESSION — REVENU
# #         # ════════════════════════════════
# #         reg_input = {
# #             'frequency':               frequency,
# #             'uniqueinvoices':          uniqueinvoices,
# #             'uniqueproducts':          uniqueproducts,
# #             'avgdaysbetweenpurchases': avgdays,
# #             'avgbasketvalue':          avgbasketvalue,
# #             'purchaseintensity':       purchaseintensity,
# #             'satisfactionscore':       satisfaction,
# #             'monetarytotal':           monetarytotal,
# #         }
# #         X_reg    = pd.DataFrame([reg_input])[REG_FEATURES]
# #         X_reg_sc = scaler_reg.transform(X_reg)
# #         revenu   = max(0, float(reg_model.predict(X_reg_sc)[0]))
# #         # ════════════════════════════════
# #         # 3. CLUSTERING — SEGMENT
# #         # ════════════════════════════════
# #         RFM_14_FEATURES = [
# #             'frequency', 'monetarytotal', 'monetaryavg',
# #             'avgdaysbetweenpurchases', 'totalquantity', 'uniqueproducts',
# #             'avgquantitypertransaction', 'supportticketscount', 'satisfactionscore',
# #             'age', 'weekendpurchaseratio', 'avgbasketvalue',
# #             'purchaseintensity', 'productdiversityratio'
# #         ]

# #         rfm_input = {
# #             'frequency':                 frequency,
# #             'monetarytotal':             monetarytotal,
# #             'monetaryavg':               avgbasketvalue,
# #             'avgdaysbetweenpurchases':   avgdays,
# #             'totalquantity':             uniqueinvoices * 5,
# #             'uniqueproducts':            uniqueproducts,
# #             'avgquantitypertransaction': avgqty,
# #             'supportticketscount':       1.0,
# #             'satisfactionscore':         satisfaction,
# #             'age':                       35.0,
# #             'weekendpurchaseratio':      0.30,
# #             'avgbasketvalue':            avgbasketvalue,
# #             'purchaseintensity':         purchaseintensity,
# #             'productdiversityratio':     diversity
# #         }

# #         # X_rfm    = pd.DataFrame([rfm_input])[RFM_14_FEATURES]
# #         # sc_tmp   = StandardScaler()
# #         # X_sc_rfm = sc_tmp.fit_transform(X_rfm)
# #         # X_pca    = pca.transform(X_sc_rfm)

# #         # n = kmeans.cluster_centers_.shape[1]
# #         # if   X_pca.shape[1] > n: X_pca = X_pca[:, :n]
# #         # elif X_pca.shape[1] < n:
# #         #     X_pca = np.hstack([X_pca, np.zeros((1, n - X_pca.shape[1]))])

# #         # cluster = int(kmeans.predict(X_pca)[0])
# #         # cluster_name, cluster_desc = CLUSTER_NAMES.get(
# #         #     cluster, (f"Cluster {cluster}", ""))
# #         X_rfm    = pd.DataFrame([rfm_input])[RFM_14_FEATURES]
# #         X_sc_rfm = kmeans_scaler.transform(X_rfm)  # ← scaler original
# #         X_pca    = pca.transform(X_sc_rfm)

# #         n = kmeans.cluster_centers_.shape[1]
# #         if   X_pca.shape[1] > n: X_pca = X_pca[:, :n]
# #         elif X_pca.shape[1] < n:
# #             X_pca = np.hstack([X_pca, np.zeros((1, n - X_pca.shape[1]))])

# #         cluster = int(kmeans.predict(X_pca)[0])
# #         cluster_name, cluster_desc = CLUSTER_NAMES.get(
# #             cluster, (f"Cluster {cluster}", ""))
# #         return jsonify({
# #             'status': 'success',
# #             'churn': {
# #                 'prediction':  churn_pred,
# #                 'probabilite': round(churn_prob * 100, 1),
# #                 'label': 'Risque de Churn' if churn_pred == 1 else 'Client Fidèle'
# #             },
# #             'revenu': {
# #                 'prediction': round(revenu, 2)
# #             },
# #             'cluster': {
# #                 'id':          cluster,
# #                 'nom':         cluster_name,
# #                 'description': cluster_desc
# #             }
# #         })

# #     except Exception as e:
# #         return jsonify({'status': 'error', 'message': str(e)})


# # if __name__ == '__main__':
# #     app.run(debug=True, port=5000)

# from flask import Flask, render_template, request, jsonify
# import joblib
# import numpy as np
# import pandas as pd
# import json
# import os
# from sklearn.preprocessing import StandardScaler

# app = Flask(__name__)

# BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# def load_model(name):
#     return joblib.load(os.path.join(BASE, 'models', name))

# # ── chargement des modèles ──
# churn_model  = load_model('flask_churn_model.pkl')
# scaler_clf   = load_model('flask_scaler_clf.pkl')
# reg_model    = load_model('flask_reg_model.pkl')
# scaler_reg   = load_model('flask_scaler_reg.pkl')
# kmeans       = load_model('kmeans_model.pkl')
# pca          = load_model('pca_model.pkl')
# kmeans_scaler = load_model('kmeans_scaler.pkl')


# with open(os.path.join(BASE, 'models', 'flask_config.json')) as f:
#     config = json.load(f)

# CLF_FEATURES = config['clf_features']
# REG_FEATURES = config['reg_features']

# RFM_FEATURES = [
#     'frequency', 'monetarytotal', 'monetaryavg',
#     'avgdaysbetweenpurchases', 'uniqueproducts',
#     'avgquantitypertransaction', 'satisfactionscore',
#     'avgbasketvalue', 'purchaseintensity', 'productdiversityratio'
# ]

# CLUSTER_NAMES = {
#     0: ("Client VIP",         "Très actif, haute valeur. Fidélisation prioritaire."),
#     1: ("Client Régulier",    "Fréquence stable. Bon potentiel de croissance."),
#     2: ("Client Occasionnel", "Achète peu souvent. Marketing ciblé recommandé."),
#     3: ("Client Inactif",     "Peu actif. Risque churn élevé. Réengagement urgent.")
# }


# @app.route('/')
# def index():
#     return render_template('index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         d = request.get_json()

#         # ── lecture des 8 champs saisis ──
#         frequency     = float(d['frequency'])
#         monetarytotal = float(d['monetarytotal'])
#         uniqueinvoices= float(d['uniqueinvoices'])
#         uniqueproducts= float(d['uniqueproducts'])
#         avgdays       = float(d['avgdaysbetweenpurchases'])
#         satisfaction  = float(d['satisfactionscore'])
#         regyear       = float(d['regyear'])
#         regmonth      = float(d['regmonth'])

#         # ── features calculées automatiquement ──
#         customertenuredays = max(1, (2025 - int(regyear)) * 365)
#         purchaseintensity  = frequency / (customertenuredays + 1)
#         avgbasketvalue     = monetarytotal / (frequency + 1)
#         avgqty             = (uniqueinvoices * 5) / (frequency + 1)
#         diversity          = uniqueproducts / (frequency + 1)
#         regmonth_sin       = float(np.sin(2 * np.pi * regmonth / 12))
#         regmonth_cos       = float(np.cos(2 * np.pi * regmonth / 12))

#         # ════════════════════════════════
#         # 1. CLASSIFICATION — CHURN
#         # ════════════════════════════════
#         clf_input = {
#             'frequency':               frequency,
#             'purchaseintensity':       purchaseintensity,
#             'uniqueinvoices':          uniqueinvoices,
#             'regyear':                 regyear,
#             'regmonth_sin':            regmonth_sin,
#             'regmonth_cos':            regmonth_cos,
#             'uniqueproducts':          uniqueproducts,
#             'avgdaysbetweenpurchases': avgdays,
#         }
#         X_clf    = pd.DataFrame([clf_input])[CLF_FEATURES]
#         X_clf_sc = scaler_clf.transform(X_clf)

#         churn_pred = int(churn_model.predict(X_clf_sc)[0])
#         churn_prob = float(churn_model.predict_proba(X_clf_sc)[0][1])
#         # ✅ correction seuils
#         prob = churn_prob * 100

#         if prob >= 60:
#             niveau = "🔴 Client à risque élevé"
#             conseil = "Action urgente recommandée"
#         elif prob >= 30:
#             niveau = "🟡 Client à surveiller"
#             conseil = "Relance marketing conseillée"
#         else:
#             niveau = "🟢 Client fidèle"
#             conseil = "Maintenir la relation"
            
#         # ════════════════════════════════
#         # 2. RÉGRESSION — REVENU
#         # ════════════════════════════════
#         reg_input = {
#             'frequency':               frequency,
#             'uniqueinvoices':          uniqueinvoices,
#             'uniqueproducts':          uniqueproducts,
#             'avgdaysbetweenpurchases': avgdays,
#             'avgbasketvalue':          avgbasketvalue,
#             'purchaseintensity':       purchaseintensity,
#             'satisfactionscore':       satisfaction,
#             'monetarytotal':           monetarytotal,
#         }
#         X_reg    = pd.DataFrame([reg_input])[REG_FEATURES]
#         X_reg_sc = scaler_reg.transform(X_reg)
#         revenu   = max(0, float(reg_model.predict(X_reg_sc)[0]))

#         # ════════════════════════════════
#         # 3. CLUSTERING — SEGMENT
#         # ════════════════════════════════
#         RFM_14_FEATURES = [
#             'frequency', 'monetarytotal', 'monetaryavg',
#             'avgdaysbetweenpurchases', 'totalquantity', 'uniqueproducts',
#             'avgquantitypertransaction', 'supportticketscount', 'satisfactionscore',
#             'age', 'weekendpurchaseratio', 'avgbasketvalue',
#             'purchaseintensity'
#         ]

#         rfm_input = {
#             'frequency':                 frequency,
#             'monetarytotal':             monetarytotal,
#             'monetaryavg':               avgbasketvalue,
#             'avgdaysbetweenpurchases':   avgdays,
#             'totalquantity':             uniqueinvoices * 5,
#             'uniqueproducts':            uniqueproducts,
#             'avgquantitypertransaction': avgqty,
#             'supportticketscount':       1.0,
#             'satisfactionscore':         satisfaction,
#             'age':                       35.0,
#             'weekendpurchaseratio':      0.30,
#             'avgbasketvalue':            avgbasketvalue,
#             'purchaseintensity':         purchaseintensity,
#         }

#         # X_rfm    = pd.DataFrame([rfm_input])[RFM_14_FEATURES]
#         # sc_tmp   = StandardScaler()
#         # X_sc_rfm = sc_tmp.fit_transform(X_rfm)
#         # X_pca    = pca.transform(X_sc_rfm)
#         X_rfm    = pd.DataFrame([rfm_input])[RFM_14_FEATURES]
#         X_sc_rfm = kmeans_scaler.transform(X_rfm)  # ← scaler original
#         X_pca    = pca.transform(X_sc_rfm)
#         n = kmeans.cluster_centers_.shape[1]
#         if   X_pca.shape[1] > n: X_pca = X_pca[:, :n]
#         elif X_pca.shape[1] < n:
#             X_pca = np.hstack([X_pca, np.zeros((1, n - X_pca.shape[1]))])

#         cluster = int(kmeans.predict(X_pca)[0])
#         cluster_name, cluster_desc = CLUSTER_NAMES.get(
#             cluster, (f"Cluster {cluster}", ""))

#         return jsonify({
#             'status': 'success',
#             # 'churn': {
#             #     'prediction':  churn_pred,
#             #     'probabilite': round(churn_prob * 100, 1),
#             #     'label': 'Risque de Churn' if churn_pred == 1 else 'Client Fidèle'
#             # },
#             'churn': {
#                 'prediction':  churn_pred,
#                 'probabilite': round(prob, 1),
#                 'niveau': niveau,
#                 'conseil': conseil
#             },
#             'revenu': {
#                 'prediction': round(revenu, 2)
#             },
#             'cluster': {
#                 'id':          cluster,
#                 'nom':         cluster_name,
#                 'description': cluster_desc
#             }
#         })

#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)})


# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import json
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model(name):
    return joblib.load(os.path.join(BASE, 'models', name))

# ── chargement des modèles ──
churn_model  = load_model('flask_churn_model.pkl')
scaler_clf   = load_model('flask_scaler_clf.pkl')
reg_model    = load_model('flask_reg_model.pkl')
scaler_reg   = load_model('flask_scaler_reg.pkl')
kmeans       = load_model('kmeans_model.pkl')
pca          = load_model('pca_model.pkl')
kmeans_scaler = load_model('kmeans_scaler.pkl')


with open(os.path.join(BASE, 'models', 'flask_config.json')) as f:
    config = json.load(f)

CLF_FEATURES = config['clf_features']
REG_FEATURES = config['reg_features']

RFM_FEATURES = [
    'frequency', 'monetarytotal', 'monetaryavg',
    'avgdaysbetweenpurchases', 'uniqueproducts',
    'avgquantitypertransaction', 'satisfactionscore',
    'avgbasketvalue', 'purchaseintensity', 'productdiversityratio'
]

CLUSTER_NAMES = {
    0: ("Client VIP",         "Très actif, haute valeur. Fidélisation prioritaire."),
    1: ("Client Régulier",    "Fréquence stable. Bon potentiel de croissance."),
    2: ("Client Occasionnel", "Achète peu souvent. Marketing ciblé recommandé."),
    3: ("Client Inactif",     "Peu actif. Risque churn élevé. Réengagement urgent.")
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        d = request.get_json()

        # ── lecture des 8 champs saisis ──
        frequency     = float(d['frequency'])
        monetarytotal = float(d['monetarytotal'])
        uniqueinvoices= float(d['uniqueinvoices'])
        uniqueproducts= float(d['uniqueproducts'])
        avgdays       = float(d['avgdaysbetweenpurchases'])
        satisfaction  = float(d['satisfactionscore'])
        regyear       = float(d['regyear'])
        regmonth      = float(d['regmonth'])

        # ── features calculées automatiquement ──
        customertenuredays = max(1, (2025 - int(regyear)) * 365)
        purchaseintensity  = frequency / (customertenuredays + 1)
        avgbasketvalue     = monetarytotal / (frequency + 1)
        avgqty             = (uniqueinvoices * 5) / (frequency + 1)
        diversity          = uniqueproducts / (frequency + 1)
        regmonth_sin       = float(np.sin(2 * np.pi * regmonth / 12))
        regmonth_cos       = float(np.cos(2 * np.pi * regmonth / 12))

        # ════════════════════════════════
        # 1. CLASSIFICATION — CHURN
        # ════════════════════════════════
        clf_input = {
            'frequency':               frequency,
            'purchaseintensity':       purchaseintensity,
            'uniqueinvoices':          uniqueinvoices,
            'regyear':                 regyear,
            'regmonth_sin':            regmonth_sin,
            'regmonth_cos':            regmonth_cos,
            'uniqueproducts':          uniqueproducts,
            'avgdaysbetweenpurchases': avgdays,
        }
        X_clf    = pd.DataFrame([clf_input])[CLF_FEATURES]
        X_clf_sc = scaler_clf.transform(X_clf)

        churn_pred = int(churn_model.predict(X_clf_sc)[0])
        churn_prob = float(churn_model.predict_proba(X_clf_sc)[0][1])
        # ✅ correction seuils
        prob = churn_prob * 100

        if prob >= 60:
            niveau = "🔴 Client à risque élevé"
            conseil = "Action urgente recommandée"
            cluster_name = "Client Inactif"
        elif prob >= 40:
            niveau = "🟡 Client à surveiller"
            conseil = "Relance marketing conseillée"
            cluster_name = "Client requlier"
        else :
            niveau = "🟢 Client fidèle"
            conseil = "Maintenir la relation"
            cluster_name = "Client VIP"

        # ════════════════════════════════
        # 2. RÉGRESSION — REVENU
        # ════════════════════════════════
        reg_input = {
            'frequency':               frequency,
            'uniqueinvoices':          uniqueinvoices,
            'uniqueproducts':          uniqueproducts,
            'avgdaysbetweenpurchases': avgdays,
            'avgbasketvalue':          avgbasketvalue,
            'purchaseintensity':       purchaseintensity,
            'satisfactionscore':       satisfaction,
            'monetarytotal':           monetarytotal,
        }
        X_reg    = pd.DataFrame([reg_input])[REG_FEATURES]
        X_reg_sc = scaler_reg.transform(X_reg)
        revenu   = max(0, float(reg_model.predict(X_reg_sc)[0]))

        # ════════════════════════════════
        # 3. CLUSTERING — SEGMENT
        # ════════════════════════════════
        RFM_14_FEATURES = [
            'frequency', 'monetarytotal', 'monetaryavg',
            'avgdaysbetweenpurchases', 'totalquantity', 'uniqueproducts',
            'avgquantitypertransaction', 'supportticketscount', 'satisfactionscore',
            'age', 'weekendpurchaseratio', 'avgbasketvalue',
            'purchaseintensity'
        ]

        rfm_input = {
            'frequency':                 frequency,
            'monetarytotal':             monetarytotal,
            'monetaryavg':               avgbasketvalue,
            'avgdaysbetweenpurchases':   avgdays,
            'totalquantity':             uniqueinvoices * 5,
            'uniqueproducts':            uniqueproducts,
            'avgquantitypertransaction': avgqty,
            'supportticketscount':       1.0,
            'satisfactionscore':         satisfaction,
            'age':                       35.0,
            'weekendpurchaseratio':      0.30,
            'avgbasketvalue':            avgbasketvalue,
            'purchaseintensity':         purchaseintensity,
        }

        # X_rfm    = pd.DataFrame([rfm_input])[RFM_14_FEATURES]
        # sc_tmp   = StandardScaler()
        # X_sc_rfm = sc_tmp.fit_transform(X_rfm)
        # X_pca    = pca.transform(X_sc_rfm)
        X_rfm    = pd.DataFrame([rfm_input])[RFM_14_FEATURES]
        X_sc_rfm = kmeans_scaler.transform(X_rfm)  # ← scaler original
        X_pca    = pca.transform(X_sc_rfm)
        n = kmeans.cluster_centers_.shape[1]
        if   X_pca.shape[1] > n: X_pca = X_pca[:, :n]
        elif X_pca.shape[1] < n:
            X_pca = np.hstack([X_pca, np.zeros((1, n - X_pca.shape[1]))])

        cluster = int(kmeans.predict(X_pca)[0])
        cluster_desc = CLUSTER_NAMES.get(
            cluster, (f"Cluster {cluster}", ""))

        return jsonify({
            'status': 'success',
            # 'churn': {
            #     'prediction':  churn_pred,
            #     'probabilite': round(churn_prob * 100, 1),
            #     'label': 'Risque de Churn' if churn_pred == 1 else 'Client Fidèle'
            # },
            'churn': {
                'prediction':  churn_pred,
                'probabilite': round(prob, 1),
                'niveau': niveau,
                'conseil': conseil
            },
            'revenu': {
                'prediction': round(revenu, 2)
            },
            'cluster': {
                'id':          cluster,
                'nom':         cluster_name,
                'description': cluster_desc
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)