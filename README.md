# TUNUSIA_LICENSE_PLATE-RECOGNITION
# 🚗 Reconnaissance de Plaques d'Immatriculation

Ce projet implémente un modèle de réseau de neurones convolutionnels (CNN) pour la reconnaissance de plaques d'immatriculation.
L'objectif est de détecter et de classer les caractères présents sur les plaques, 
ce qui peut être utile dans diverses applications, telles que le contrôle d'accès et la surveillance.

## ✨ Fonctionnalités

- 🔍 Reconnaissance automatique des caractères sur les plaques d'immatriculation.
- 🔤 Prise en charge de 29 classes différentes de caractères.
- 📸 Prédictions basées sur des images en niveaux de gris.

## 🛠️ Technologies Utilisées

- 🐍 Python
- 🔷 Keras
- 🔢 NumPy
- 📊 scikit-learn

## 📥 Installation
git clone git@github.com:muhammedehab35/TUNUSIA_LICENSE_PLATE-RECOGNITION.git

cd TUNUSIA_LICENSE_PLATE-RECOGNITION

pip install -r requirements.txt


## ⚙️ Utilisation
python MODEL_SCRIPT.py


## 📊 Prétraitement des Données

est une étape cruciale pour assurer la qualité des images et améliorer la précision de la reconnaissance des caractères. Les principales techniques utilisées incluent :

# Extraction des Images :

Lecture des images et des étiquettes depuis les répertoires spécifiés.
# Filtrage Homomorphe :


Amélioration du contraste et réduction du bruit à l'aide de transformations dans le domaine fréquentiel.
# Suppression des Petits Objets :

Élimination des éléments indésirables dans les images binaires.
# Segmentation des Caractères :

Isolation des caractères présents sur les plaques pour un traitement ultérieur.
# Amélioration de l'Image :

Techniques d'égalisation d'histogramme et filtrage adaptatif pour améliorer la qualité visuelle.
# Augmentation des Données : 

Application de transformations (rotation, retournement) pour enrichir le jeu de données.
# Sauvegarde des Images Prétraitées : 

Conservation des images traitées pour une utilisation ultérieure dans le modèle.



## 🤝 Contributions
Indique comment les autres peuvent contribuer à ton projet.
