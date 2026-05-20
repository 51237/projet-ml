import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Chargement ──
df = pd.read_csv('data/idf_2024.csv')

# ── Vue d'ensemble ──
print(f'Dimensions : {df.shape}')
print(f'\nTypes :\n{df.dtypes}')
print(f'\nAperçu :\n{df.head()}')

# ── Statistiques de base ──
print(f'\nStatistiques :\n{df.describe()}')

# ── Valeurs manquantes ──
print(f'\nValeurs manquantes :\n{df.isnull().sum()}')

# ── Types de biens présents ──
print(f'\nTypes de biens :\n{df["Type local"].value_counts()}')

# ── Nature des mutations ──
print(f'\nNature mutations :\n{df["Nature mutation"].value_counts()}')