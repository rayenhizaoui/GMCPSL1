import streamlit as st
import pandas as pd
import joblib
import re

# Charger le pipeline entraîné
pipeline = joblib.load('model.pkl')

# Charger le dataset pour obtenir les valeurs uniques des dropdowns
df = pd.read_csv('Expresso_churn_dataset.csv')

# Fonction pour mapper TENURE en valeurs numériques
def map_tenure(tenure):
    numbers = re.findall(r'\d+', tenure)
    return int(numbers[0]) if numbers else 0

# Interface Streamlit
st.title('Prédiction de Churn Expresso')

# Champs de saisie
region = st.selectbox('Région', options=sorted(df['REGION'].fillna('Unknown').unique()))
tenure = st.selectbox('Durée (Tenure)', options=sorted(df['TENURE'].unique()))
montant = st.number_input('Montant', min_value=0.0, value=0.0)
frequence_rech = st.number_input('Fréquence de Recharge', min_value=0.0, value=0.0)
revenue = st.number_input('Revenu', min_value=0.0, value=0.0)
arpu_segment = st.number_input('Segment ARPU', min_value=0.0, value=0.0)
frequence = st.number_input('Fréquence', min_value=0.0, value=0.0)
data_volume = st.number_input('Volume de Données', min_value=0.0, value=0.0)
on_net = st.number_input('Appels On-Net', min_value=0.0, value=0.0)
orange = st.number_input('Appels Orange', min_value=0.0, value=0.0)
tigo = st.number_input('Appels Tigo', min_value=0.0, value=0.0)
zone1 = st.number_input('Appels Zone 1', min_value=0.0, value=0.0)
zone2 = st.number_input('Appels Zone 2', min_value=0.0, value=0.0)
mrg = st.selectbox('MRG', options=['NO', 'YES', 'Unknown'])
regularity = st.number_input('Régularité', min_value=0.0, value=0.0)
top_pack = st.selectbox('Top Pack', options=sorted(df['TOP_PACK'].fillna('Unknown').unique()))
freq_top_pack = st.number_input('Fréquence Top Pack', min_value=0.0, value=0.0)

# Bouton de validation
if st.button('Prédire'):
    # Créer un dataframe avec les données saisies
    input_data = pd.DataFrame({
        'REGION': [region],
        'MONTANT': [montant],
        'FREQUENCE_RECH': [frequence_rech],
        'REVENUE': [revenue],
        'ARPU_SEGMENT': [arpu_segment],
        'FREQUENCE': [frequence],
        'DATA_VOLUME': [data_volume],
        'ON_NET': [on_net],
        'ORANGE': [orange],
        'TIGO': [tigo],
        'ZONE1': [zone1],
        'ZONE2': [zone2],
        'MRG': [mrg],
        'REGULARITY': [regularity],
        'TOP_PACK': [top_pack],
        'FREQ_TOP_PACK': [freq_top_pack],
        'TENURE_NUM': [map_tenure(tenure)]
    })
    
    # Prédire la probabilité de churn
    prob = pipeline.predict_proba(input_data)[:, 1][0]
    st.write(f'**Probabilité de Churn :** {prob:.2f}')
