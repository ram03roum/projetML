"""
train_simple_model_custom.py — Train simple model with specific 7 features

Features used:
- age
- frequency  
- monetarytotal
- totaltransactions
- weekendpurchaseratio
- avgquantitypertransaction
- recency

Usage:
    python src/train_simple_model_custom.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE


# Exactly the 7 features you specified
CUSTOM_FEATURES = [
    'age',
    'frequency',
    'monetarytotal',
    'totaltransactions',
    'weekendpurchaseratio',
    'avgquantitypertransaction',
    'recency'
]


def load_and_prepare_custom_data(data_path):
    """Load data and prepare custom feature set."""
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()
    
    print(f"Original data shape: {df.shape}")
    print(f"Churn distribution:\n{df['churn'].value_counts()}")
    
    # Check which features are available
    available_features = [f for f in CUSTOM_FEATURES if f in df.columns]
    missing_features = [f for f in CUSTOM_FEATURES if f not in df.columns]
    
    print(f"\nCustom features requested: {len(CUSTOM_FEATURES)}")
    print(f"Available features ({len(available_features)}): {available_features}")
    if missing_features:
        print(f"Missing features: {missing_features}")
        print("⚠️  Training will continue with available features only!")
    
    # Use available features only
    X = df[available_features].copy()
    y = df['churn'].copy()
    
    # Handle missing values with median imputation
    print(f"\nHandling missing values:")
    for col in X.columns:
        missing_count = X[col].isnull().sum()
        if missing_count > 0:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            print(f"  ✅ Filled {missing_count} missing values in {col} with median: {median_val:.2f}")
        else:
            print(f"  ✅ {col}: No missing values")
    
    return X, y, available_features


def create_custom_model(X, y, features_used):
    """Create and train the custom simple model."""
    print(f"\n{'='*60}")
    print("Training Custom Simple Model")
    print(f"{'='*60}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset split:")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    print(f"  Train churn distribution: {np.bincount(y_train)} (0: Loyal, 1: Churn)")
    print(f"  Test churn distribution: {np.bincount(y_test)} (0: Loyal, 1: Churn)")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n✅ Features scaled using StandardScaler")
    
    # Apply SMOTE to balance classes
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"✅ SMOTE applied - Balanced train distribution: {np.bincount(y_train_smote)}")
    
    # Train Random Forest model
    print(f"\nTraining Random Forest with {len(features_used)} features...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train_smote, y_train_smote)
    print(f"✅ Model training completed")
    
    # Predictions
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate model
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0)
    }
    
    print(f"\n{'='*50}")
    print("CUSTOM MODEL PERFORMANCE")
    print(f"{'='*50}")
    print(f"Features used: {len(features_used)}")
    print(f"Feature list: {features_used}")
    print(f"\nAccuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Loyal  Churn")
    print(f"Actual Loyal   {cm[0,0]:3d}    {cm[0,1]:3d}")
    print(f"      Churn    {cm[1,0]:3d}    {cm[1,1]:3d}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': features_used,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance (sorted):")
    for i, (_, row) in enumerate(importance_df.iterrows(), 1):
        print(f"  {i}. {row['feature']:25s}: {row['importance']:.4f}")
    
    return rf_model, scaler, metrics, y_test, y_pred, importance_df


def save_custom_confusion_matrix(y_true, y_pred, features_used, path='reports/custom_simple_model_confusion_matrix.png'):
    """Save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Loyal (0)', 'Churn (1)'],
                yticklabels=['Loyal (0)', 'Churn (1)'])
    plt.title(f'Simple Model Confusion Matrix\n({len(features_used)} features)')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {path}")


def save_custom_models(model, scaler, features_used, metrics, importance_df):
    """Save the custom model and all related files."""
    # Create models directory if it doesn't exist
    Path("models").mkdir(exist_ok=True)
    
    # Save model and scaler (overwrite existing simple model)
    joblib.dump(model, 'models/simple_model.pkl')
    joblib.dump(scaler, 'models/simple_scaler.pkl')
    
    # Save detailed feature information
    with open('models/simple_model_features.txt', 'w') as f:
        f.write("Simple Model Features for Flask App\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total features: {len(features_used)}\n\n")
        f.write("Features (in order):\n")
        for i, feature in enumerate(features_used, 1):
            f.write(f"{i}. {feature}\n")
        f.write(f"\nPerformance Metrics:\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
        f.write(f"\nFeature Importance (sorted):\n")
        for i, (_, row) in enumerate(importance_df.iterrows(), 1):
            f.write(f"{i}. {row['feature']:25s}: {row['importance']:.4f}\n")
    
    # Save Python list format for easy copy-paste to Flask app
    with open('models/flask_feature_list.py', 'w') as f:
        f.write("# Feature list for Flask app (app_simple.py)\n")
        f.write("# Copy this list to FEATURE_NAMES in your Flask app\n\n")
        f.write("FEATURE_NAMES = [\n")
        for feature in features_used:
            f.write(f"    '{feature}',\n")
        f.write("]\n")
    
    print(f"\n✅ Custom models saved:")
    print(f"   - models/simple_model.pkl")
    print(f"   - models/simple_scaler.pkl")
    print(f"   - models/simple_model_features.txt")
    print(f"   - models/flask_feature_list.py")


def test_custom_predictions(model, scaler, features_used):
    """Test the model with sample predictions."""
    print(f"\n{'='*50}")
    print("TESTING CUSTOM MODEL PREDICTIONS")
    print(f"{'='*50}")
    
    # Test cases with your exact features
    test_cases = [
        {
            'name': 'High Churn Risk Customer',
            'values': {
                'age': 25,
                'frequency': 2, 
                'monetarytotal': 150,
                'totaltransactions': 3,
                'weekendpurchaseratio': 0.1,
                'avgquantitypertransaction': 1.2,
                'recency': 250
            }
        },
        {
            'name': 'Loyal VIP Customer', 
            'values': {
                'age': 45,
                'frequency': 15,
                'monetarytotal': 2500,
                'totaltransactions': 20,
                'weekendpurchaseratio': 0.3,
                'avgquantitypertransaction': 3.5,
                'recency': 30
            }
        },
        {
            'name': 'Average Customer',
            'values': {
                'age': 35,
                'frequency': 6,
                'monetarytotal': 800,
                'totaltransactions': 8,
                'weekendpurchaseratio': 0.2,
                'avgquantitypertransaction': 2.1,
                'recency': 90
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        
        # Create DataFrame with available features only
        test_data = {}
        for feature in features_used:
            if feature in test_case['values']:
                test_data[feature] = test_case['values'][feature]
                print(f"  {feature:25s}: {test_case['values'][feature]}")
            else:
                # This shouldn't happen with our current setup
                test_data[feature] = 0  
        
        test_df = pd.DataFrame([test_data])
        
        # Scale and predict
        test_scaled = scaler.transform(test_df)
        prediction = model.predict(test_scaled)[0]
        probability = model.predict_proba(test_scaled)[0][1]
        
        print(f"  {'→ Prediction':25s}: {'🔴 CHURN' if prediction == 1 else '🟢 LOYAL'}")
        print(f"  {'→ Churn Probability':25s}: {probability:.1%}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CUSTOM SIMPLE MODEL TRAINING")
    print("="*60)
    print("Features to be used:")
    for i, feature in enumerate(CUSTOM_FEATURES, 1):
        print(f"  {i}. {feature}")
    
    # Create reports directory
    Path("reports").mkdir(exist_ok=True)
    
    # Load and prepare data
    DATA_PATH = 'data/processed/step3_feature_engineering.csv'
    X, y, features_used = load_and_prepare_custom_data(DATA_PATH)
    
    # Create and train model
    model, scaler, metrics, y_test, y_pred, importance_df = create_custom_model(X, y, features_used)
    
    # Save confusion matrix
    save_custom_confusion_matrix(y_test, y_pred, features_used)
    
    # Save models and related files
    save_custom_models(model, scaler, features_used, metrics, importance_df)
    
    # Test model with sample predictions
    test_custom_predictions(model, scaler, features_used)
    
    print(f"\n{'='*60}")
    print("CUSTOM SIMPLE MODEL TRAINING COMPLETED!")
    print("="*60)
    print(f"\n✅ Success! Model trained with {len(features_used)} features")
    print(f"📊 Final accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"\n🚀 Next steps:")
    print(f"1. The simple_model.pkl has been updated")
    print(f"2. Run: python app/app_simple.py")
    print(f"3. Open: http://localhost:5000")
    print(f"\n📝 Note: Flask app will be updated automatically to use these features:")
    for i, feature in enumerate(features_used, 1):
        print(f"   {i}. {feature}")
    print("="*60 + "\n")