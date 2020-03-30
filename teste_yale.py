import cv2
import os
import numpy as np
from PIL import Image

detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

reconhecedorEigen = cv2.face.EigenFaceRecognizer_create()
reconhecedorEigen.read("classificadorEigenYale.yml")

reconhecedorFisher = cv2.face.FisherFaceRecognizer_create()
reconhecedorFisher.read("classificadorFisherYale.yml")

reconhecedorLBPH = cv2.face.LBPHFaceRecognizer_create()
reconhecedorLBPH.read("classificadorLBPHYale.yml")

f = open('avaliacao_yale.txt', 'w+')
f.write("# Avalicao de Confiabilidade de Algoritmos\n")
f.write('====================================================================================\n')

caminhos = [os.path.join('yalefaces/teste', f) for f in os.listdir('yalefaces/teste')]

for i in range(1, 4):
    totalAcertos = 0
    percentualAcerto = 0.0
    totalConfianca = 0.0

    if i == 1: 
        reconhecedor = reconhecedorEigen
        f.write("# Algortimo Eigen Faces:\n\n")
    elif i == 2: 
        reconhecedor = reconhecedorFisher
        f.write("# Algortimo Fisher Faces:\n\n")
    else: 
        reconhecedor = reconhecedorLBPH
        f.write("# Algortimo LBPH Faces:\n\n")

    for caminhoImagem in caminhos:
        imagemFace = Image.open(caminhoImagem).convert('L')
        imagemFaceNP = np.array(imagemFace, 'uint8')
        facesDetectadas = detectorFace.detectMultiScale(imagemFaceNP)
        for (x, y, l, a) in facesDetectadas:
            idprevisto, confianca = reconhecedor.predict(imagemFaceNP)
            idatual = int(os.path.split(caminhoImagem)[1].split(".")[0].replace("subject", ""))
            print(str(idatual) + " foi classificado como " + str(idprevisto) + " - " + str(confianca))        
            f.write(str(idatual) + " foi classificado como " + str(idprevisto) + " - " + str(confianca) + "\n")
            if idprevisto == idatual:
                totalAcertos += 1
                totalConfianca += confianca

    percentualAcerto = (totalAcertos / 30) * 100
    totalConfianca = totalConfianca / totalAcertos
    print("Total de acertos: " + str(totalAcertos) + " de 30")
    print("Percentual de acerto: " + str(percentualAcerto) + "%")
    print("Total confiança: " + str(totalConfianca))

    f.write('====================================================================================\n')
    f.write("Total de acertos: " + str(totalAcertos) + " de 30\n")
    f.write("Percentual de acerto: " + str(percentualAcerto) + "%\n")
    f.write("Total confianca: " + str(totalConfianca) + "\n")
    f.write('====================================================================================\n\n')

f.close()