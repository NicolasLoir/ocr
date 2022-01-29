# import des packages
from tensorflow.keras.datasets import mnist
import numpy as np


def charger_az_dataset(Chemin):
    # initialise les listes de données et les labels
    data = []
    labels = []
    # boucle sur les lignes du dataset A-Z mascuscrit
    for row in open(Chemin):
        # sépare les labels et images de la ligne
        row = row.split(",")
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype="uint8")

    # les images sont représentés ici sur un seul channel(grayscale)
    # en  28x28 = 784 pixels.
    # Nous devons prendre cette représentation "applatie" et la
    # convertir en matrice utilisable
    image = image.reshape((28, 28))
    # met a joru les listes de données et labels
    data.append(image)
    labels.append(label)
    # convertir les données et labels en tables NumPy
    data = np.array(data, dtype="float32")
    labels = np.array(labels, dtype="int")
    # retourner un couple comportant les données et labels
    return (data, labels)


def charger_mnist_dataset():
    # Chargement du dataset et séparation, mise en relation
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
    data = np.vstack([trainData, testData])
    labels = np.hstack([trainLabels, testLabels])
    # retourner un couple comportant les données et labels
    return (data, labels)
