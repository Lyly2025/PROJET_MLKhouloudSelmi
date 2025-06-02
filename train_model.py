# train_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ===== IMPORTS KERAS =====
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense

# ===== 1. FONCTION DE CONSTRUCTION KERAS =====
def build_keras_model(input_dim):
    """Crée un modèle séquentiel Keras pour la régression"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1)  # Couche de sortie linéaire pour la régression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ===== 2. CHARGEMENT ET NETTOYAGE DES DONNÉES (EXISTANT) =====
df = pd.read_excel('Online Retail.xlsx')
# [Vos étapes de nettoyage existantes...]

# ===== 3. FEATURE ENGINEERING (EXISTANT) =====
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
# [Vos autres transformations...]

# ===== 4. PRÉPARATION DES DONNÉES =====
features = ['Quantity', 'UnitPrice', 'InvoiceHour', 'InvoiceDayOfWeek', 'DescriptionLength', 'Country']
X = df[features]
y = df['TotalPrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== 5. PRÉTRAITEMENT (EXISTANT) =====
num_features = ['Quantity', 'UnitPrice', 'InvoiceHour', 'InvoiceDayOfWeek', 'DescriptionLength']
cat_features = ['Country']
preprocessor = ColumnTransformer([...])

# ===== 6. PIPELINE AVEC KERAS =====
keras_regressor = KerasRegressor(
    build_fn=lambda: build_keras_model(len(num_features) + len(cat_features)),
    epochs=20,
    batch_size=32,
    verbose=1
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', keras_regressor)  # Remplace RandomForestRegressor
])

# ===== 7. ENTRAÎNEMENT =====
print("\nEntraînement du modèle Keras...")
history = pipeline.fit(X_train, y_train)

# ===== 8. ÉVALUATION =====
y_pred = pipeline.predict(X_test)
print("\nPerformance du modèle Keras:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# ===== 9. SAUVEGARDE =====
# Sauvegarde Keras (.h5)
keras_model = pipeline.named_steps['regressor'].model
save_model(keras_model, 'sales_forecast.h5')

# Sauvegarde du préprocesseur (optionnel)
joblib.dump(preprocessor, 'preprocessor.joblib')

print("\n✅ Modèle Keras sauvegardé sous sales_forecast.h5")
print("✅ Préprocesseur sauvegardé sous preprocessor.joblib")