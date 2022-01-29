# imports des packages
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
import os

# argument parser
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                 help="Chemin vers l'image a classifier")
# ap.add_argument("-m", "--model", type=str, required=True,
#                 help="Chemin vers votre modele")
# args = vars(ap.parse_args())

# Chargment du modèle d'ocr
print("[INFO] Chargement du modèle...")
# model = load_model(args["model"])
model = load_model('./readchar.model')

print(os.getcwd())

# Chargment de l'image, mise en niveau de gris et
# flou gaussien pour attenuer le bruit
image = cv2.imread('./image.png')
# image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Detection de bords et contours
edged = cv2.Canny(blurred, 30, 150)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]
# Initialisation de la liste des contours qui seront
# utilisés pour la reconnaissance
chars = []


# cv2.imshow('Classique', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow('Gray', gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow('blurred', blurred)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('cnts', edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# ---------------------------

# Boucler sur les contours
for c in cnts:
    # On trouve les dimensions et position du contour
    (x, y, w, h) = cv2.boundingRect(c)
    # on filtre les contours, on ignore les contours
    # trop grand ou trop petit
    # !!! Attention, il faudra peut être modifier ces valeurs
    # selon votre photo !!!
    if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
        # On extrait le caracteres et applique un threshold
        # de façon a faire apparaitre le caractère en blanc sur un
        # fond noir puis on récupère nos dimensions
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape
        # On redimensionne l'image si la largeur est supérieur à la hauteur
        if tW > tH:
            thresh = imutils.resize(thresh, width=32)
        # Sinon on redimensionne par la hauteur
        # Cela pour en lien avec notre modele fonctionnant par
        # limite de 32x32 pixels
        else:
            thresh = imutils.resize(thresh, height=32)

        # On récupère de nouveau nos dimensions
        # On determine de combien de pixel on doit completer
        # l'image. Nous avons déjà une dimension en 32 pixels
        # maintenant l'image doit faire 32x32
        (tH, tW) = thresh.shape
        dX = int(max(0, 32 - tW) / 2.0)
        dY = int(max(0, 32 - tH) / 2.0)
        # On rempli l'image pour faire 32x32
        padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                    left=dX,
                                    right=dX, borderType=cv2.BORDER_CONSTANT,
                                    value=(0, 0, 0))
        padded = cv2.resize(padded, (32, 32))
        # On réajuste l'intensite et on ajoute une dimension
        padded = padded.astype("float32") / 255.0
        padded = np.expand_dims(padded, axis=-1)
        # On mets enfin à jour notre array
        chars.append((padded, (x, y, w, h)))

# On extrait les "boites" de nos caracteres
boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")
# On effectue nos predictions (liste)
preds = model.predict(chars)
# On defini notre liste de label
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

# On boucle sur nos prédictions
for (pred, (x, y, w, h)) in zip(preds, boxes):
    # On trouve l'index du lebel ayant la probabilité la plus
    # de notre prédiction puis on extrait le label
    i = np.argmax(pred)
    prob = pred[i]
    label = labelNames[i]
    # On dessine la "boite" et notre label sur l'image
    # pas obligatoire mais pratique pour le TP
    print("[INFO] {} - {:.2f}%".format(label, prob * 100))
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    # On affiche l'image et attend un appui sur un touche
    # pour passer à la suite
    cv2.imshow("Image", image)
    cv2.waitKey(0)
