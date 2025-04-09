import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import joblib
import re

# Charger le dataset
df = pd.read_csv('Expresso_churn_dataset.csv')

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

# Préprocessing pour les données numériques
num_transformer = SimpleImputer(strategy='constant', fill_value=0)

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

# Sauvegarder le modèle
joblib.dump(pipeline, 'model.pkl')

print("Modèle entraîné et sauvegardé sous 'model.pkl'.")