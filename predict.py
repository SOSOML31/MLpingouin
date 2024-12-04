import pandas as pd
from joblib import load

# Charger le modèle préalablement sauvegardé
model = load('app/model/penguin_model.joblib')

def predict_species(data):
    """
    Fonction qui prend en entrée les caractéristiques du manchot et retourne l'espèce prédite.
    """
    # Si `data` est un objet Pydantic (PenguinInput), le convertir en dictionnaire
    if hasattr(data, 'dict'):
        data = data.dict()

    # Vérifier la structure des données après conversion en dictionnaire
    print(f"Data reçu après conversion: {data}")

    # Convertir les données d'entrée en DataFrame
    input_data = pd.DataFrame([data], columns=['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'])
    
    # Vérifier la forme de `input_data` pour débogage
    print(f"Data transformée en DataFrame: {input_data}")

    # Prédire l'espèce du manchot
    prediction = model.predict(input_data)
    return prediction[0]  # Retourner l'espèce prédite
