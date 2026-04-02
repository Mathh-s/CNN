import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
IMG_SIZE = 64

ds_train, ds_test = tfds.load('cats_vs_dogs',split=['train[:80%]', 'train[80%:]'],as_supervised=True)

def dataset_to_numpy(ds, img_size):
    X, y = [], []
    for image, label in ds:
        image = image.numpy()
        label = label.numpy()
        image = cv2.resize(image, (img_size, img_size))
        X.append(image)
        y.append(label)
    X = np.array(X, dtype=np.float32) / 255.0  # Normalisation
    y = np.array(y, dtype=np.float32)
    return X, y
X_train, y_train = dataset_to_numpy(ds_train, IMG_SIZE)
X_demo, y_demo = dataset_to_numpy(ds_test, IMG_SIZE)

def relu(x):
    return np.maximum(x,0)

def relu_backward(gradient, x):
    dX = np.array(gradient, copy=True)
    dX[x <= 0] = 0
    return dX
def softmax(x):
    shift_x = x - np.max(x)
    exp_x = np.exp(shift_x)
    somme = np.sum(exp_x, axis=1) # Résultat de forme (N,)
    somme = somme.reshape(-1, 1)  # On force la forme (N, 1)
    
    return exp_x / somme

def cross_loss(probs, y_true):
    probs = probs.flatten()
    prob_classe_reelle = probs[int(y_true)]
    loss = -np.log(prob_classe_reelle + 1e-8)
    return loss  

def train(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=64):

    pass

class Conv:
    def __init__(self, n_filters, filter_size, n_channels, lr=0.01):
        #Nombre de filtres
        self.n_filters = n_filters

        #Taille d'un filtre 
        self.filter_size = filter_size

        #Nombre de canaux d'entrée
        self.n_channels = n_channels

        #Taux d'apprentissage
        self.lr = lr
        
        #Initialisation des filtres (0.1 comme petite variance)
        self.filters = np.random.randn(n_filters, n_channels, filter_size, filter_size) * 0.1
        #Biais de chaque filtre initialisé à 0
        self.biases = np.zeros(n_filters)

    def forward(self, entree):
        #Sauvegarde pour backward (gradient)
        self.entree = entree

        #Dimensions : canaux, hauteur, largeur (1 seule image à la fois)
        n_channels, H, W = entree.shape

        #Calcul de la taille de la sortie après convolution sans padding
        #Formule : taille_sortie = taille_entrée - taille_filtre + 1
        H_out = H - self.filter_size + 1
        W_out = W - self.filter_size + 1

        #Allocation du tenseur de sortie rempli de zéros
        #Forme : (n_filters, H_out, W_out)
        sortie = np.zeros((self.n_filters, H_out, W_out))

        #Boucle sur chaque filtre (chaque filtre produit une feature map)
        for f in range(self.n_filters):
            #Boucle sur chaque position verticale de la fenêtre glissante
            for i in range(H_out):
                #Boucle sur chaque position horizontale de la fenêtre glissante
                for j in range(W_out):
                    #Région de l'image couverte par le filtre
                    #Forme : (n_channels, filter_size, filter_size)
                    region = entree[:, i:i+self.filter_size, j:j+self.filter_size]

                    #Convolution : produit élément par élément entre la région et le filtre, puis somme de tous les résultats + biais
                    sortie[f, i, j] = np.sum(region * self.filters[f]) + self.biases[f]

        return sortie

    def backward(self, gradient):
        n_filters, H_out, W_out = gradient.shape
        n_channels, H, W = self.entree.shape
        
        d_filters = np.zeros(self.filters.shape)
        d_biases = np.zeros(self.biases.shape)
        d_entree = np.zeros(self.entree.shape)

        for f in range(n_filters):
            for i in range(H_out):
                for j in range(W_out):
                    region = self.entree[:, i:i+self.filter_size, j:j+self.filter_size]
                    
                    d_filters[f] += region * gradient[f, i, j]
                    d_biases[f] += gradient[f, i, j]
                    d_entree[:, i:i+self.filter_size, j:j+self.filter_size] += gradient[f, i, j] * self.filters[f]

        self.filters -= self.lr * d_filters
        self.biases -= self.lr * d_biases

        return d_entree


class Pooling:
    def __init__(self, pool_size=2):
        #Taille de la fenêtre 
        self.pool_size = pool_size

    def forward(self, entree):
        #Sauvegarde pour backward (quels pixels étaient les max)
        self.entree = entree  
        
        #Dimensions
        channels, H, W = entree.shape

        #Taille fenêtre
        p = self.pool_size

        #Sortie divisée par la taille de la fenêtre dans chaque dimension 
        H_out = H // p
        W_out = W // p

        #Tenseur de sortie
        sortie = np.zeros((channels, H_out, W_out))

        #Pour 1 seule image
        #Boucle sur chaque canal 
        for c in range(channels):
            #Boucle sur chaque position verticale de la fenêtre
            for i in range(H_out):
                #Boucle sur chaque position horizontale de la fenêtre
                for j in range(W_out):
                    #Fenêtre p×p dans le canal c
                    #i*p:(i+1)*p sélectionne les lignes de la fenêtre
                    #j*p:(j+1)*p sélectionne les colonnes de la fenêtre
                    region = entree[c, i*p:(i+1)*p, j*p:(j+1)*p]#Max Pooling
                    sortie[c, i, j] = np.max(region)

        return sortie
        
    def backward(self, gradient):
        channels, H_out, W_out = grandient.shape
        p = self.pool_size
        d_entree = np.zeros(self.entree.shape)
        for c in range(channels):
            for i in range(H_out):
                for j in range(W_out):
                    region = self.entree[c, i*p:(i+1)*p, j*p:(j+1)*p]
                    mask = (region == np.max(region))
                    d_entree[c, i*p:(i+1)*p, j*p:(j+1)*p] = mask * gradient[c, i, j]
        return d_entree

class Dense:
    def __init__(self, nentree, nsortie, lr=0.01):
        #Nombre de neurones d'entrée
        self.nentree = nentree

        #Nombre de neurones de sortie
        self.nsortie = nsortie

        #Taux d'apprentissage
        self.lr = lr

        #Initialisation He pour ReLU : variance=2/nentree
        #Forme matrice de poids : (nentree, nsortie)
        self.W = np.random.randn(nentree, nsortie) * np.sqrt(2.0 / nentree)

        #Biais de chaque neurone de sortie initialisé à 0
        self.b = np.zeros(nsortie)

    def forward(self, X):
        #Sauvegarde pour le backward
        self.entree = X  
        #Forme X : (1, nentree)
        #Forme W : (nentree, nsortie)
        #Forme sortie : (1, nsortie)
        return X @ self.W + self.b

    def backward(self, gradient):
        dW = self.entree.T @ gradient # Calcule l'erreur associée aux poids
        db = np.sum(gradient, axis=0) # Calcule l'erreur associée aux biais
        dX = gradient @ self.W.T # Calcule l'erreur de l'entrée pour la passer à la couche d'avant
        self.W -= self.lr * dW # Modifie les poids
        self.b -= self.lr * db # Modifie les biais
        return dX

class CNN:
    def __init__(self, lr=0.001):
        #Taux d'apprentissage
        self.lr = lr
        #Architecture : Conv, Pooling, Dense, Dense
        self.conv = Conv(n_filters=8, filter_size=3, n_channels=1, lr=lr)
        self.pool = Pooling(pool_size=2)
        self.dense1 = Dense(nentree=8 * 13 * 13, nsortie=128, lr=lr)
        self.dense2 = Dense(nentree=128, nsortie=10, lr=lr)

    def forward(self, X):
        #Convolution
        out = self.conv.forward(X)

        #ReLU
        out = np.maximum(0, out)

        #Max Pooling
        out = self.pool.forward(out)



    def backward(self, gradientscore):
        # 1. Retour à travers la couche Dense 2
        grad = self.dense2.backward(gradientscore)
        
        # 2. Retour à travers la ReLU de la couche Dense
        grad = relu_backward(grad, self.out_dense1)
        
        # 3. Retour à travers la couche Dense 1
        grad = self.dense1.backward(grad)
        
        # 4. "Unflatten" : On redonne au gradient la forme (8, 13, 13)
        grad = grad.reshape(8, 13, 13)
        
        # 5. Retour à travers le Pooling
        grad = self.pool.backward(grad)
        
        # 6. Retour à travers la ReLU de la Conv
        grad = relu_backward(grad, self.out_conv)
        
        # 7. Retour à travers la Conv (Mise à jour des filtres)
        grad = self.conv.backward(grad)
        
        return grad

    def predict(self, X):
        """
        Prend une image X et retourne l'indice de la classe prédite
        """
        # On récupère les scores bruts (logits)
        logits = self.forward(X)
        
        # On transforme en probabilités (en utilisant ta fonction softmax)
        probas = softmax(logits)
        
        # On retourne l'indice du maximum (ex: 3 si c'est le chiffre 3)
        return np.argmax(probas)






#hhdhhddnd
