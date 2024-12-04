import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Dossier où enregistrer les fichiers : utiliser le dossier du script
script_dir = os.path.dirname(os.path.realpath(__file__))  # Dossier où le script est exécuté
dataset_dir = script_dir  # Les fichiers seront sauvegardés dans le même répertoire que le script

# Télécharger le dataset depuis Kaggle
# Assurez-vous que l'API Kaggle est configurée dans votre environnement (fichier API 'kaggle.json')
os.system(f"kaggle datasets download -d parulpandey/palmer-archipelago-antarctica-penguin-data -p {dataset_dir} --unzip")

print("Dataset téléchargé et extrait dans le dossier :", dataset_dir)

# Chemins vers les fichiers extraits
file_size = os.path.join(dataset_dir, "penguins_size.csv")
file_lter = os.path.join(dataset_dir, "penguins_lter.csv")

# Charger les données
penguins_size = pd.read_csv(file_size)
penguins_lter = pd.read_csv(file_lter)

# Aperçu des deux datasets
print("Dataset penguins_size :")
print(penguins_size.head())
print("\nDataset penguins_lter :")
print(penguins_lter.head())

# Nettoyage des données (Suppression des lignes avec des valeurs manquantes)
penguins_size_cleaned = penguins_size.dropna()
penguins_lter_cleaned = penguins_lter.dropna()

# Afficher des informations sur les datasets nettoyés
print("\nInformations sur le dataset penguins_size :")
print(penguins_size_cleaned.info())

print("\nInformations sur le dataset penguins_lter :")
print(penguins_lter_cleaned.info())

# Vérification des valeurs manquantes avant nettoyage
print("\nValeurs manquantes dans penguins_size avant nettoyage :")
print(penguins_size.isnull().sum())

print("\nValeurs manquantes dans penguins_lter avant nettoyage :")
print(penguins_lter.isnull().sum())

# Vérification après nettoyage
print("\nValeurs manquantes dans penguins_size après nettoyage :")
print(penguins_size_cleaned.isnull().sum())

print("\nValeurs manquantes dans penguins_lter après nettoyage :")
print(penguins_lter_cleaned.isnull().sum())

# Sélectionner les caractéristiques et la cible
features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
target = 'species'

# Créer les variables X et y
X = penguins_size_cleaned[features]  # Caractéristiques
y = penguins_size_cleaned[target]    # Cible (espèce)

# Diviser les données en train et test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fusionner les caractéristiques et la cible pour chaque ensemble
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Sauvegarder les datasets dans des fichiers CSV dans le même dossier que le script
train_data.to_csv(os.path.join(script_dir, 'train.csv'), index=False)
test_data.to_csv(os.path.join(script_dir, 'test.csv'), index=False)

print("Les fichiers train.csv et test.csv ont été créés et sauvegardés dans le dossier :", script_dir)

# Charger et vérifier les fichiers CSV créés
train_data = pd.read_csv(os.path.join(script_dir, 'train.csv'))
test_data = pd.read_csv(os.path.join(script_dir, 'test.csv'))

print("\nEnsemble d'entraînement :")
print(train_data.head())

print("\nEnsemble de test :")
print(test_data.head())
