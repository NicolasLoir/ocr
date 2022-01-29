# Permet à matplotlib d'enregistrer les courbes
import argparse
import numpy as np
import matplotlib.pyplot as plt
from imutils import build_montages
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from loaders import charger_az_dataset
from loaders import charger_mnist_dataset
from resnet import ResNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib
matplotlib.use("Agg")
# importer les packages


# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--az", required=True,
                help="chemin vers A-Z dataset")
ap.add_argument("-m", "--model", type=str, required=True,
                help="chemin vers la sortie de résultat ")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="chemin vers les courbes")
args = vars(ap.parse_args())

# Initialisation des epochs d'entrainement, learning rate,
# et batch size
EPOCHS = 1
INIT_LR = 1e-1
BS = 128
# Charger les datasets
print("[INFO] Chargement des datasets...")
(azData, azLabels) = charger_az_dataset(args["az"])
(digitsData, digitsLabels) = charger_mnist_dataset()

# Les labels du dataset MNIST étant de 0 à 9 nous allons ajouter 10
# aux labels A-Z afin d'éviter des problèmes de labellisation
# On ne veut pas confondre "1" et "B"...
azLabels += 10
# On empile nos données et labels
data = np.vstack([azData, digitsData])
labels = np.hstack([azLabels, digitsLabels])
# Avant de connaitre notre architecture nous avions harmoniser nos
# images en 28x28 pixels, toutefois ResNet utilise des images
# en 32x32 pixels, nous devons donc les redimensionner
data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")
# Ajontons un channel a nos images et changeons l'intensité des pixels
# Nous passons d'un greyscale à noir et blanc [0,255]->[0,1]
data = np.expand_dims(data, axis=-1)
data /= 255.0

# conversion des labels d'entier vers vecteur
le = LabelBinarizer()
labels = le.fit_transform(labels)
counts = labels.sum(axis=0)
# prise en compte des erreurs de labilisation initiales
classTotals = labels.sum(axis=0)
classWeight = {}
# Boucle d'application de poids des classes
for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]
# separation des données d'entrainement et de test avec un ration 80%/20%
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.20, random_state=42)


# construction d'un générateur d'image afin d'augmenter notre dataset
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    fill_mode="nearest")

# Initialisation et compilation du réseau
print("[INFO] Compilation du modele...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3),
                     (64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print("[INFO] Entrainement du réseau...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS,
    class_weight=classWeight,
    verbose=1)
# Définition des lsites de labels
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]
print("[INFO] Evaluation du réseau...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=labelNames))


print("[INFO] Enregistrement du réseau...")
model.save(args["model_OCR"], save_format="h5")
# construction de la courbe et sauvegarde de celle-ci
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])


print("\nTeest des images en dessous 12345678azerty\n")

# Initialisation de notre liste d'images
images = []
# selection aleatoire d'images de caracteres
for i in np.random.choice(np.arange(0, len(testY)), size=(49,)):
    # On essai de classifier
    probs = model.predict(testX[np.newaxis, i])
    prediction = probs.argmax(axis=1)
    label = labelNames[prediction[0]]
    # On décide d'ajouter le label sur l'image
    image = (testX[i] * 255).astype("uint8")
    color = (0, 255, 0)
    # Si le label prédéfini et celui découvert son différent
    # on change la couleur (Rappel : BGR )
    if prediction[0] != np.argmax(testY[i]):
        color = (0, 0, 255)
    # On combine les channels de l'image afin qu'un être humain puisse
    # l'observer correctement puis on rajoute le label
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                color, 2)
    # On rajoute notre image à la liste en sortie de fonction
    images.append(image)
# Construction de l'image finale de résulat
montage = build_montages(images, (96, 96), (7, 7))[0]
# Affichage du résultat
cv2.imshow("OCR Results", montage)
cv2.waitKey(0)
