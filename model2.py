import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


# 1. Charger le dataset propre
df = pd.read_csv("data/idf_clean.csv")


# 2. Separer X et y
y = df["Valeur fonciere"]
X = df.drop(columns=["Valeur fonciere"])


# 3. Faire le train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Creer et entrainer le modele
model = LinearRegression()
model.fit(X_train, y_train)


# 5. Faire les predictions
y_pred = model.predict(X_test)


# 6. Evaluer le modele
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# 7. Afficher les resultats
print("MAE :", mae)
print("RMSE :", rmse)
print("R2 :", r2)