import numpy as np
import matplotlib.pyplot as plt

def download_mnist(path="./mnist"):
    #Télécharge data
    #Retourne X_train, y_train & X_test, y_test
    pass

def relu(x):
    return np.maximum(0, x)

def relu_backward(gradient, x):
    #Rétropropagation
    #dL/dx = dL/dy * 1 si x > 0, sinon 0
    pass

def softmax(x):
    #Convertit logits en proba
    #softmax(x_i) = exp(x_i) / somme exp(x_j)
    pass

def cross_loss(resultatss, y_true):
    #Calcul loss
    #L = -1/N * somme log(p[classe_vraie])
    pass

def train(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=64):
    #Boucle d'entraînement
    #Pour chaque epoch :
        #Shuffle
        #Découpage en mini-batches de taille `batch_size`
        #Pour chaque mini-batch :
               #Forward, Softmax, Calcul loss, Backward, MAJ
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
        pass


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
        #Transforme gradient de la forme (channels, H_out, W_out) en (channels, H, W)
        pass


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
        out = self.conv.forward(X)           #(8, 26, 26)

        #ReLU
        out = np.maximum(0, out)             #(8, 26, 26)

        #Max Pooling
        out = self.pool.forward(out)         #(8, 13, 13)

        #Aplatissement (flatten)
        batch = out.shape[0]
        out = out.reshape(batch, -1)         #(8*13*13 = 1352)

        #Couche Dense
        out = self.dense1.forward(out)       #(1, 128)
        #ReLU
        out = np.maximum(0, out)            

        #Couche de sortie (logits)
        out = self.dense2.forward(out)       #(1, 10)

        return out  #scores bruts (avant softmax)

    def backward(self, gradientscore):
        pass

    def predict(self, X):
        #Forward pour logits
        scores = self.forward(X)

        #Indice avec le score le plus élevé (classe prédite pour chaque image du batch)
        return np.argmax(scores, axis=1)
