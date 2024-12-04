from flask import Flask, request, jsonify  # Flask pour créer l'API, request pour gérer les requêtes HTTP, jsonify pour formater les réponses en JSON
from flask_cors import CORS  # Flask-CORS pour permettre les requêtes cross-origin
import torch  # PyTorch pour charger et utiliser le modèle d'apprentissage automatique
from torchvision import transforms  # Transformations pour préparer les images pour le modèle
from PIL import Image  # Bibliothèque pour manipuler les images
import io  # Pour lire les données binaires de l'image
from sqlalchemy import create_engine, Column, Integer, String  # SQLAlchemy pour gérer la base de données
from sqlalchemy.ext.declarative import declarative_base  # Pour définir les tables de la base de données
from sqlalchemy.orm import sessionmaker  # Pour gérer les sessions avec la base de données
import os 

DB_NAME = "DATAML"
DB_HOST = "localhost"
DB_USER = "postgres"
DB_PASSWORD = "931752"
DB_PORT = "5432"

# Création de l'URL de connexion pour SQLAlchemy
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Création de l'objet Engine pour interagir avec la base de données
engine = create_engine(DATABASE_URL)

# Déclarer la base pour définir les modèles de tables
Base = declarative_base()

# Définition de la table `predictions`
class PredictionResult(Base):
    __tablename__ = "predictions"  # Nom de la table
    id = Column(Integer, primary_key=True, autoincrement=True)  # ID unique auto-incrémenté
    image_name = Column(String, nullable=True)  # Nom du fichier image envoyé
    result = Column(String, nullable=False)  # Résultat de la prédiction (texte)

# Créer la table dans la base de données si elle n'existe pas
Base.metadata.create_all(engine)

# Configuration des sessions pour interagir avec la base
Session = sessionmaker(bind=engine)
session = Session()

# === Configuration de Flask ===

# Initialiser l'application Flask
app = Flask(__name__)

# Activer CORS pour permettre les requêtes depuis n'importe quelle origine
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Limiter la taille des fichiers à 16 MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# === Chargement du modèle d'apprentissage automatique ===

# Chemin vers le modèle PyTorch enregistré
model_path = "best_penguin_model.pth"

# Vérifier si le modèle existe
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le modèle '{model_path}' est introuvable.")

# Charger le modèle sur le périphérique approprié (GPU si disponible, sinon CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model.eval()  # Mettre le modèle en mode évaluation (pas d'entraînement)

# === Définition des classes ===
# Les différentes classes de pingouins que le modèle peut prédire
class_names = ['Adelie Penguin', 'Chinstrap Penguin', 'Emperor Penguin', 'Gentoo Penguin']

# === Transformations pour préparer les images ===
# Appliquer des transformations nécessaires pour que l'image soit au bon format pour le modèle
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionner l'image à 224x224 pixels
    transforms.ToTensor(),  # Convertir l'image en tenseur PyTorch
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normaliser les couleurs selon les poids pré-entraînés
])

# === Route pour effectuer des prédictions ===

@app.route('/predict', methods=['POST'])
def predict():
    # Vérifier si un fichier a été envoyé avec la requête
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400

    file = request.files['file']  # Récupérer le fichier envoyé
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400

    try:
        # Récupérer le nom du fichier
        image_name = file.filename

        # Charger et transformer l'image
        image = Image.open(io.BytesIO(file.read())).convert("RGB")  # Lire l'image et la convertir en RGB
        image = transform(image).unsqueeze(0).to(device)  # Appliquer les transformations et ajouter une dimension pour le batch

        # Faire une prédiction
        with torch.no_grad():  # Désactiver les calculs de gradients pour accélérer le processus
            outputs = model(image)  # Passer l'image dans le modèle pour obtenir les prédictions
            _, preds = torch.max(outputs, 1)  # Récupérer l'indice de la classe avec la plus grande probabilité
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]  # Convertir les scores en probabilités

        # Récupérer le nom de la classe prédite et la confiance associée
        predicted_class = class_names[preds[0]]  # Classe prédite
        confidence = probabilities[preds[0]].item() * 100  # Probabilité en pourcentage
        is_penguin = confidence > 60  # Vérifier si c'est un pingouin avec plus de 50% de confiance

        # Construire le texte du résultat
        result_text = f"_ ({predicted_class}) à {confidence:.2f}%" if is_penguin else f"Prend une meilleurs photo"

        # Enregistrer le résultat dans la base de données
        prediction = PredictionResult(image_name=image_name, result=result_text)
        session.add(prediction)
        session.commit()

        # Retourner la réponse JSON
        return jsonify({'image_name': image_name, 'result': result_text})

    except Exception as e:
        # Annuler les modifications si une erreur survient
        session.rollback()
        return jsonify({'error': str(e)}), 500

# === Route pour récupérer toutes les prédictions enregistrées ===

@app.route('/results', methods=['GET'])
def get_results():
    # Récupérer toutes les entrées de la table `predictions`
    results = session.query(PredictionResult).all()
    # Transformer les résultats en format JSON
    return jsonify([
        {"id": result.id, "image_name": result.image_name, "result": result.result} for result in results
    ])

# === Point d'entrée principal ===
if __name__ == '__main__':
    # Lancer l'application Flask sur toutes les interfaces réseau, port 5000
    app.run(host='0.0.0.0', port=5000)