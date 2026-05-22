import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Chargement des données nettoyées ──
df = pd.read_csv('data/idf_clean.csv')

print(f'Dimensions : {df.shape}')
print(f'\nAperçu :\n{df.head()}')

# ── Étape 1 : sélectionner les colonnes utiles ──
features = ['Surface reelle bati', 'Nombre pieces principales', 'Type local', 'Code departement', 'Code postal']

# ── Étape 2 : séparer features et cible ──
X = df[features]               # les entrées du modèle
y = df['Valeur fonciere']      # ce qu'on veut prédire — le prix

# ── Étape 3 : séparation train / test ──
# 80% pour entraîner le modèle, 20% pour l'évaluer
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f'\nEntraînement : {len(X_train)} exemples')
print(f'Test         : {len(X_test)} exemples')

# ── Étape 4 : entraîner le modèle ──
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Étape 5 : évaluer sur les données de test ──
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f'\nRésultats Random Forest :')
print(f'MAE  : {mae:,.0f}€')
print(f'RMSE : {rmse:,.0f}€')
print(f'R²   : {r2:.2f}')

# ── Étape 6 : feature importance ──
importances = pd.Series(model.feature_importances_, index=features)
importances_triees = importances.sort_values(ascending=False)

print(f'\nImportance des features :')
print(importances_triees)

importances_triees.plot(kind='barh', color='steelblue')
plt.title('Importance des features — Random Forest Prix Immobilier')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('data/feature_importance.png')
plt.show()

# ── Étape 7 : exemples de prédictions ──
nouveaux_biens = pd.DataFrame([
    # Appartement, 50m², 2 pièces, Paris (75)
    {'Surface reelle bati': 50, 'Nombre pieces principales': 2, 'Type local': 0, 'Code departement': 75, 'Code postal': 75011},
    # Maison, 120m², 5 pièces, Seine-et-Marne (77)
    {'Surface reelle bati': 120, 'Nombre pieces principales': 5, 'Type local': 1, 'Code departement': 77, 'Code postal': 77100},
    # Appartement, 30m², 1 pièce, Seine-Saint-Denis (93)
    {'Surface reelle bati': 30, 'Nombre pieces principales': 1, 'Type local': 0, 'Code departement': 93, 'Code postal': 93100},
])

predictions = model.predict(nouveaux_biens)

descriptions = [
    'Appartement 50m² 2p - Paris 11ème',
    'Maison 120m² 5p - Seine-et-Marne',
    'Appartement 30m² 1p - Seine-Saint-Denis',
]

print('\nExemples de prédictions :')
for desc, prix in zip(descriptions, predictions):
    print(f'{desc} → {prix:,.0f}€')