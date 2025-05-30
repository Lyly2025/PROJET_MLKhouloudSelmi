# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def load_and_preprocess_data(filepath):
    """Charge et prétraite les données"""
    try:
        df = pd.read_csv(filepath, sep=';', decimal=',', encoding='latin1')
        print("Données chargées avec succès.")
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {e}", file=sys.stderr)
        return None
    
    # Nettoyage des données
    df.columns = df.columns.str.strip()
    
    # Filtrage des données
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]  # Exclure les retours
    df = df.dropna(subset=['CustomerID'])  # Garder seulement les clients connus
    df = df[(df['Quantity'] > 0) & (df['Quantity'] < 1000)]  # Quantités raisonnables
    df = df[(df['UnitPrice'] > 0) & (df['UnitPrice'] < 100)]  # Prix raisonnables
    
    # Conversion des types
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    
    # Feature engineering
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDay'] = df['InvoiceDate'].dt.day
    df['InvoiceMonth'] = df['InvoiceDate'].dt.month
    df['InvoiceWeekday'] = df['InvoiceDate'].dt.weekday
    df['InvoiceHour'] = df['InvoiceDate'].dt.hour
    df['DescriptionLength'] = df['Description'].str.len()
    df['IsWeekend'] = df['InvoiceWeekday'].isin([5, 6]).astype(int)
    
    print(f"Données après prétraitement: {df.shape[0]} lignes")
    return df

def feature_engineering(df):
    """Crée les features pour le modèle"""
    print("\nCréation des features...")
    
    # Features
    text_features = 'Description'
    categorical_features = ['Country', 'InvoiceMonth', 'InvoiceWeekday']
    numerical_features = ['Quantity', 'InvoiceDay', 'InvoiceHour', 
                         'DescriptionLength', 'IsWeekend']
    
    # Préprocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=500, stop_words='english', 
                                   ngram_range=(1, 2)), text_features),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ])
    
    X = preprocessor.fit_transform(df)
    y = df['UnitPrice']
    
    print(f"Shape des features: {X.shape}")
    return X, y, preprocessor

def train_model(X, y):
    """Entraîne et évalue le modèle"""
    print("\nDébut de l'entraînement...")
    
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    
    # Modèle avec optimisation des hyperparamètres
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    model = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=2  # Affiche plus d'infos pendant l'entraînement
    )
    
    # Entraînement
    print("Recherche des meilleurs hyperparamètres...")
    model.fit(X_train, y_train)
    
    # Meilleurs paramètres
    print("\nMeilleurs paramètres trouvés:")
    print(model.best_params_)
    
    # Évaluation
    y_pred = model.best_estimator_.predict(X_test)
    
    print("\nPerformance du modèle:")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    
    # Visualisation des résultats
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
    plt.xlabel('Prix réel')
    plt.ylabel('Prix prédit')
    plt.title('Comparaison des prix réels et prédits')
    plt.savefig('results.png')  # Sauvegarde le graphique
    plt.show()
    
    return model.best_estimator_

def save_artifacts(model, preprocessor):
    """Sauvegarde les artefacts du modèle"""
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    joblib.dump(pipeline, 'price_predictor_pipeline.joblib')
    print("\nPipeline sauvegardé:")
    print("- price_predictor_pipeline.joblib")
    print("- results.png (graphique)")

if __name__ == "__main__":
    # 1. Chargement des données
    print("Chargement et prétraitement des données...")
    df = load_and_preprocess_data('Online_Retail.csv')
    
    if df is not None:
        # 2. Feature engineering
        X, y, preprocessor = feature_engineering(df)
        
        # 3. Entraînement
        model = train_model(X, y)
        
        # 4. Sauvegarde
        save_artifacts(model, preprocessor)
    else:
        print("Erreur lors du chargement des données - arrêt du programme.")
        sys.exit(1)