import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

# Charger le dataset
print("Chargement du dataset...")
df = pd.read_csv('Expresso_churn_dataset.csv')

# Afficher les informations générales sur le dataset
print("\n--- Informations générales sur le dataset ---")
print(f"Nombre de lignes: {df.shape[0]}, Nombre de colonnes: {df.shape[1]}")
print("\nAperçu des premières lignes:")
print(df.head())
print("\nInformations sur les colonnes:")
print(df.info())
print("\nStatistiques descriptives:")
print(df.describe())
print("\nValeurs manquantes par colonne:")
print(df.isnull().sum())

# Créer un rapport de profilage pandas
print("\nCréation du rapport de profilage...")
profile = ProfileReport(df, title="Expresso Churn Dataset Profiling Report", explorative=True)
profile.to_file("expresso_churn_profiling_report.html")
print("Rapport de profilage enregistré sous 'expresso_churn_profiling_report.html'")

# Vérifier et supprimer les doublons
print("\n--- Gestion des doublons ---")
duplicates = df.duplicated().sum()
print(f"Nombre de lignes dupliquées: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Doublons supprimés. Nouveau nombre de lignes: {df.shape[0]}")

# Fonction pour mapper TENURE en valeurs numériques
def map_tenure(tenure):
    if pd.isna(tenure):
        return 0
    numbers = re.findall(r'\d+', tenure)
    return int(numbers[0]) if numbers else 0

# Mapper TENURE à TENURE_NUM
df['TENURE_NUM'] = df['TENURE'].apply(map_tenure)
df = df.drop('TENURE', axis=1)

# Définir les features (X) et la cible (y)
X = df.drop(['user_id', 'CHURN'], axis=1)
y = df['CHURN']

# Définir les colonnes numériques et catégoriques
numerical_cols = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 
                  'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'ZONE1', 'ZONE2', 
                  'REGULARITY', 'FREQ_TOP_PACK', 'TENURE_NUM']
categorical_cols = ['REGION', 'MRG', 'TOP_PACK']

# Détection et traitement des valeurs aberrantes pour les colonnes numériques
print("\n--- Détection et traitement des valeurs aberrantes ---")
for col in numerical_cols:
    if col in X.columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
        print(f"Colonne {col}: {outliers} valeurs aberrantes détectées")
        
        # Méthode de plafonnement pour traiter les valeurs aberrantes
        X[col] = np.where(X[col] < lower_bound, lower_bound, X[col])
        X[col] = np.where(X[col] > upper_bound, upper_bound, X[col])

# Visualiser la distribution des caractéristiques numériques après traitement
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols[:min(9, len(numerical_cols))]):
    if col in X.columns:
        plt.subplot(3, 3, i+1)
        sns.histplot(X[col], kde=True)
        plt.title(f'Distribution de {col}')
plt.tight_layout()
plt.savefig('distribution_features.png')
print("Visualisation des distributions enregistrée sous 'distribution_features.png'")

# Préprocessing pour les données numériques avec imputation et standardisation
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Préprocessing pour les données catégoriques
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combiner les transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ])

# Définir le pipeline avec preprocessing et modèle
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    # Pour un test plus rapide, réduisez le nombre d'estimateurs:
    # ('classifier', RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1))
])

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dimensions de X_train: {X_train.shape}, X_test: {X_test.shape}")
print("Début de l'entraînement du modèle...")

# Entraîner le modèle
pipeline.fit(X_train, y_train)

print("Entraînement terminé.")

# Évaluer le modèle
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
print(f"Précision sur l'ensemble d'entraînement: {train_score:.4f}")
print(f"Précision sur l'ensemble de test: {test_score:.4f}")

# Sauvegarder le modèle
joblib.dump(pipeline, 'model.pkl')

print("Modèle entraîné et sauvegardé sous 'model.pkl'.")