import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import numpy as np
import cv2

# --- 1. CONFIGURATION ---
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 5

# --- 2. CHARGEMENT ET PRÉPARATION DES DONNÉES ---
print("Chargement du dataset Cats vs Dogs...")
# On charge 10% du dataset pour que l'entraînement soit rapide sur ton PC
ds_train, ds_test = tfds.load('cats_vs_dogs', 
                              split=['train[:10%]', 'train[90%:]'], 
                              as_supervised=True)

def dataset_to_numpy(ds, img_size):
    X, y = [], []
    for image, label in ds:
        image = image.numpy()
        # Redimensionnement pour que toutes les images aient la même taille
        image = cv2.resize(image, (img_size, img_size))
        X.append(image)
        y.append(label.numpy())
    
    # Normalisation : on passe de [0, 255] à [0, 1]
    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int32)
    return X, y

print("Conversion des données en tableaux NumPy...")
X_train, y_train = dataset_to_numpy(ds_train, IMG_SIZE)
X_test, y_test = dataset_to_numpy(ds_test, IMG_SIZE)

# --- 3. CRÉATION DU MODÈLE (L'ARCHITECTURE) ---
# Ici, on remplace tes classes manuelles par des blocs TensorFlow optimisés
model = models.Sequential([
    # Équivalent de ta classe Conv(8 filtres, 3x3) + ReLU
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # Équivalent de ta classe Pooling(2x2)
    layers.MaxPooling2D((2, 2)),
    
    # Deuxième couche de Conv/Pool pour mieux détecter les formes complexes
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Équivalent de ton .flatten() : passage de 3D à 1D
    layers.Flatten(),
    
    # Équivalent de tes classes Dense
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax') # 2 sorties : Chat (0) ou Chien (1)
])

# --- 4. COMPILATION ---
# On définit l'optimiseur (Adam) et la fonction de perte (Cross-Entropy)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- 5. ENTRAÎNEMENT ---
print("\nDébut de l'entraînement...")
model.fit(X_train, y_train, 
          epochs=EPOCHS, 
          batch_size=BATCH_SIZE, 
          validation_data=(X_test, y_test))

# --- 6. TEST DE PRÉDICTION ---
print("\nTest sur une image...")
prediction = model.predict(X_test[:1])
classe_predite = np.argmax(prediction)
print(f"Classe réelle : {y_test[0]} | Classe prédite par le CNN : {classe_predite}")
