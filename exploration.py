import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement
df = pd.read_csv('data/idf_clean.csv')

# Vue d'ensemble
print(f'Dimensions : {df.shape}')
print(f'\nTypes :\n{df.dtypes}')
print(f'\nAperçu :\n{df.head()}')

# Statistiques de base
print(f'\nStatistiques :\n{df.describe()}')

#Valeurs manquantes
print(f'\nValeurs manquantes :\n{df.isnull().sum()}')

#Types de biens présents
print(f'\nTypes de biens :\n{df["Type local"].value_counts()}')

# Nature des mutations
print(f'\nNature mutations :\n{df["Nature mutation"].value_counts()}')

# VISUALISATIONS

# Préparer les données pour les visualisations
df_viz = df[df['Type local'].isin(['Appartement', 'Maison'])].copy()
df_viz['Valeur fonciere'] = df_viz['Valeur fonciere'].str.replace(',', '.').astype(float)
df_viz = df_viz.dropna(subset=['Surface reelle bati', 'Valeur fonciere'])

# Visualisation 1 : Répartition des types de biens
df['Type local'].value_counts().plot(kind='bar', color='steelblue')
plt.title('Répartition des types de biens - Île-de-France 2024')
plt.xlabel('Type de bien')
plt.ylabel('Nombre de transactions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/type_biens.png')
plt.show()

# Visualisation 2 : Distribution des prix
df_viz[df_viz['Valeur fonciere'] < 2_000_000]['Valeur fonciere'].hist(bins=50, color='steelblue')
plt.title('Distribution des prix - Île-de-France 2024')
plt.xlabel('Prix (€)')
plt.ylabel('Nombre de transactions')
plt.tight_layout()
plt.savefig('data/distribution_prix.png')
plt.show()

# Visualisation 3 : Prix médian par département
prix_dept = df_viz.groupby('Code departement')['Valeur fonciere'].median().sort_values()
prix_dept.plot(kind='barh', color='steelblue')
plt.title('Prix médian par département - Île-de-France 2024')
plt.xlabel('Prix médian (€)')
plt.tight_layout()
plt.savefig('data/prix_departement.png')
plt.show()

# Visualisation 4 : Heatmap de corrélation
colonnes_num = ['Valeur fonciere', 'Surface reelle bati', 'Nombre pieces principales']
sns.heatmap(df_viz[colonnes_num].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Corrélation entre les variables numériques')
plt.tight_layout()
plt.savefig('data/heatmap.png')
plt.show()

print('Visualisations sauvegardées !')