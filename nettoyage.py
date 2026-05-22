import pandas as pd
from sklearn.model_selection import train_test_split
 
 
# 1. Charger le dataset
df = pd.read_csv("data/idf_2024.csv")
 
 
# 2. Garder seulement les biens utiles : Appartement et Maison
df_local = df[
    (df["Type local"] == "Appartement") |
    (df["Type local"] == "Maison")
].copy()
 
 
# 3. Nettoyer la colonne cible : Valeur fonciere
df_local["Valeur fonciere"] = (
    df_local["Valeur fonciere"]
    .str.replace(",", ".")
    .astype(float)
)
 
 
# 4. Gerer les valeurs manquantes
df_local = df_local.dropna()
 
 
# 5. Supprimer les colonnes inutiles
colonnes_a_supprimer = ["Commune", "Nature mutation"]
df_local = df_local.drop(columns=colonnes_a_supprimer)
 
 
# 6. Encoder la colonne texte utile : Type local
df_local = pd.get_dummies(df_local, columns=["Type local"], dtype=int)
 
 
# 7. Transformer la date
df_local["Date mutation"] = pd.to_datetime(df_local["Date mutation"], dayfirst=True)
df_local["annee"] = df_local["Date mutation"].dt.year
df_local["mois"] = df_local["Date mutation"].dt.month
 
 
# On supprime ensuite la date brute
df_local = df_local.drop(columns=["Date mutation"])
 
 
# 8. Separer X et y
y = df_local["Valeur fonciere"]
X = df_local.drop(columns=["Valeur fonciere"])
 
 
# 9. Faire le train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
 
# 10. Sauvegarder le dataset propre
df_local.to_csv("data/idf_clean.csv", index=False)
 
 
# Verifications
print(df_local.head())
print(df_local.dtypes)
print(df_local.isnull().sum())
print(X.head())
print(y.head())
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)