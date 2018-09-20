import random
import cv2
import numpy as np
import time
import sys
import os
#from matplotlib import pyplot as plt

def kmeans(img):
    '''Executa o k-means com k=2 a fim de segmentar uma imagem'''
    centroid1 = 0
    centroid2 = 0

    media_centroide1 = 0
    media_centroide2 = 0

    image_data = np.asarray(img)
    hist = cv2.calcHist([image_data],[0],None,[256],[0,256])
    rand_points = [random.randint(0, 255) for i in range(2)]
    width, height = img.shape[1], img.shape[0]

    # plt.hist(image_data.ravel(),256,[0,256])
    # plt.title('Histogram for gray scale picture')
    # plt.show()

    matrix_points = np.zeros([2,len(hist)])

    for k in range(0, 15):
        if k == 0:
            cent1 = rand_points[0]
            cent2 = rand_points[1]
        else:
            cent1 = media_centroide1
            cent2 = media_centroide2
        pontos_centroide1 = []
        pontos_centroide2 = []
        valores_centroide1 = []
        valores_centroide2 = []
        soma1 = 0
        soma2 = 0
        for p, val in enumerate(hist):
            if  abs(p - cent1) <  abs(p - cent2):
                pontos_centroide1.append(p)
                valores_centroide1.append(val)
                soma1 = soma1 + (p * val)
            else:
                pontos_centroide2.append(p)
                valores_centroide2.append(val)
                soma2 = soma2 + (p * val)
        media_centroide1 = int(soma1)/sum(valores_centroide1)
        media_centroide2 = int(soma2)/sum(valores_centroide2)
    return [pontos_centroide1,pontos_centroide2]

def kmeans4groups(img):
    '''Executa o kmeans com k=4 para segmentar imagens'''
    centroid1 = 0
    centroid2 = 0
    centroid3 = 0
    centroid4 = 0

    media_centroide1 = 0
    media_centroide2 = 0
    media_centroide3 = 0
    media_centroide4 = 0

    image_data = np.asarray(img)
    hist = cv2.calcHist([image_data],[0],None,[256],[0,256])
    rand_points = [random.randint(0, 255) for i in range(4)]
    width, height = img.shape[1], img.shape[0]

    #Visualizar o historigrama, se necessario
    # plt.hist(image_data.ravel(),256,[0,256])
    # plt.title('Histogram for gray scale picture')
    # plt.show()

    for k in range(0, 15):
        if k == 0:
            cent1 = rand_points[0]
            cent2 = rand_points[1]
            cent3 = rand_points[2]
            cent4 = rand_points[3]
        else:
            cent1 = media_centroide1
            cent2 = media_centroide2
            cent3 = media_centroide3
            cent4 = media_centroide4
        pontos_centroide1 = []
        pontos_centroide2 = []
        pontos_centroide3 = []
        pontos_centroide4 = []
        valores_centroide1 = []
        valores_centroide2 = []
        valores_centroide3 = []
        valores_centroide4 = []
        soma1 = 0
        soma2 = 0
        soma3 = 0
        soma4 = 0
        for p, val in enumerate(hist):
            arg1 = abs(p - cent1)
            arg2 = abs(p - cent2)
            arg3 = abs(p - cent3)
            arg4 = abs(p - cent4)
            if arg1 == min(arg1, arg2, arg3, arg4):
                pontos_centroide1.append(p)
                valores_centroide1.append(val)
                soma1 = soma1 + (p * val)
            elif arg2 == min(arg2, arg3, arg4):
                pontos_centroide2.append(p)
                valores_centroide2.append(val)
                soma2 = soma2 + (p * val)
            elif arg3 == min(arg3, arg4):
                pontos_centroide3.append(p)
                valores_centroide3.append(val)
                soma3 = soma3 + (p * val)
            else:
                pontos_centroide4.append(p)
                valores_centroide4.append(val)
                soma4 = soma4 + (p * val)

        media_centroide1 = int(soma1)/sum(valores_centroide1)
        media_centroide2 = int(soma2)/sum(valores_centroide2)
        media_centroide3 = int(soma3)/sum(valores_centroide3)
        media_centroide4 = int(soma4)/sum(valores_centroide4)
    return [pontos_centroide1,pontos_centroide2,pontos_centroide3, pontos_centroide4]

def get_median(list):
    '''Obtém a mediana de um Array.'''
    sortedlist = sorted(list)
    tamanhoLista = len(sortedlist)
    meio = int(tamanhoLista / 2)
    if tamanhoLista % 2 == 0:
        medianA = sortedlist[meio]
        medianB = sortedlist[meio-1]
        median = np.double(medianA + medianB) / 2
    else:
        median = sortedlist[meio + 1]
    return median

def getFiltro(tamanhoJanela):
    filtro = np.zeros((tamanhoJanela, tamanhoJanela), int)
    return filtro

def median_filter(img, tamFiltro):
    '''Aplica o filtro de mediada'''
    imgDest = np.zeros_like(img)
    filtro = getFiltro(tamFiltro)
    width, height = img.shape[1], img.shape[0]
    filter_width, filter_height = filtro.shape[0], filtro.shape[1]
    intensidades = []

    for y in range(height):
        for x in range(width):

            for filterY in range(int(-(filter_height / 2)), filter_height - 1):
                for filterX in range(int(-(filter_width / 2)), filter_width - 1):

                    pixel_y = y - filterY
                    pixel_x = x - filterX
                    pixel = img[filterY, filterX]

                    if (pixel_y >= 0) and (pixel_y < height) and (pixel_x >= 0) and (pixel_x < width):
                        pixel = img[pixel_y, pixel_x]

                    intensidades.append(pixel)

            mediana = get_median(intensidades)
            imgDest[y, x] = mediana
            intensidades = []
    return imgDest

def edge_detector(img):
    '''Deteção de borda utilizando o filtro de Sobel'''
    height, width = img.shape[0], img.shape[1]
    resultado = np.zeros_like(img, dtype=np.float)
    soma = np.zeros_like(img)
    GX = np.zeros_like(img)
    GY = np.zeros_like(img)
    hor = np.zeros((3, 3), int)
    '''
    -1 -2 -1
     0  0  0
     1  2  1
    '''
    hor[0,0] = -1
    hor[0,1] = -2
    hor[0,2] = -1
    hor[1,0] = 0
    hor[1,1] = 0
    hor[1,2] = 0
    hor[2,0] = 1
    hor[2,1] = 2
    hor[2,2] = 1

    ver = np.zeros((3, 3), int)
    '''
    -1 0 1;
    -2 0 2;
    -1 0 1
    '''
    ver[0,0] = -1
    ver[0,1] = 0
    ver[0,2] = 1
    ver[1,0] = -2
    ver[1,1] = 0
    ver[1,2] = 2
    ver[2,0] = -1
    ver[2,1] = 0
    ver[2,2] = 1

    for i in range(height - 1):
        for j in range(width - 1):
            GX[i, j] = abs(hor[0, 0] * np.double(img[i - 1, j - 1]) + hor[0, 1] * np.double(img[i - 1, j]) + hor[0, 2] *
                           np.double(img[i - 1, j + 1]) + hor[1, 0] * np.double(img[i, j - 1]) + hor[1, 1] *
                           np.double(img[i, j]) + hor[1, 2] * np.double(img[i, j + 1]) + hor[2, 0] *
                           np.double(img[i + 1, j - 1]) + hor[2, 1] * np.double(img[i + 1, j]) + hor[2, 2] *
                           np.double(img[i + 1, j + 1]))
            GY[i, j] = abs(ver[0, 0] * np.double(img[i - 1, j - 1]) + ver[0, 1] * np.double(img[i - 1, j]) + ver[0, 2] *
                           np.double(img[i - 1, j + 1]) + ver[1, 0] * np.double(img[i, j - 1]) + ver[1, 1] *
                           np.double(img[i, j]) + ver[1, 2] * np.double(img[i, j + 1]) + ver[2, 0] *
                           np.double(img[i + 1, j - 1]) + ver[2, 1] * np.double(img[i + 1, j]) + ver[2, 2] *
                           np.double(img[i + 1, j + 1]))
            
            soma[i, j] = abs(np.double(GX[i, j]) + np.double(GY[i, j]))
            if soma[i, j] > 100:
                resultado[i, j] = 1
            else:
                resultado[i, j] = 0
    #cv2.imshow("Detector de Bordas Sobel", resultado)
    return resultado

def process_edge_detection(imgName, filter_with_median):
    '''Abre a imagem, executa a detecção de borda e grava a nova imagem em disco'''
    start_time = time.time()
    desImgName = imgName
    imgName = sys.path[0] + '\\imgs\\' + imgName
    processedFolder = sys.path[0] + '\\' + 'processed'

    if not os.path.exists(processedFolder):
        os.makedirs(processedFolder)

    img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)
    print("Processing image '{0}'...".format(imgName))

    if (filter_with_median):
        img = median_filter(img, 7)

    destImgName = processedFolder + "\\" + desImgName
    imgDest = edge_detector(img)
    cv2.imwrite(destImgName, imgDest.astype('uint8') * 255)
    elapsed_time = time.time() - start_time
    print("Done! Elapsed Time: {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

def make_segmented_image(groups, img):
    '''Transforma os 2 grupos achados pelo k-means em 2 novas imagens binarizadas'''
    destImg = np.zeros_like(img)
    height, width = img.shape[0], img.shape[1]
    for y in range(height):
        for x in range(width):
            if groups[1] > groups[0]:
                if (img[y][x] in groups[1]):
                    destImg[y][x] = int(0)

                else:
                    destImg[y][x] = int(1)
            else:
                if (img[y][x] in groups[1]):
                    destImg[y][x] = int(1)

                else:
                    destImg[y][x] = int(0)
    return destImg

def split_kmeans(groups, img):
    '''Transforma os 4 grupos achados pelo k-means em 4 imagens'''
    destImg0 = np.zeros_like(img)
    destImg1 = np.zeros_like(img)
    destImg2 = np.zeros_like(img)
    destImg3 = np.zeros_like(img)


    height, width = img.shape[0], img.shape[1]
    for y in range(height):
        for x in range(width):
            destImg0[y][x] = 1
            destImg1[y][x] = 1
            destImg2[y][x] = 1
            destImg3[y][x] = 1

    for y in range(height):
        for x in range(width):

            if (img[y][x] in groups[0]):
                destImg0[y][x] = int(0)
            elif (img[y][x] in groups[1]):
                destImg1[y][x] = int(0)
            elif (img[y][x] in groups[2]):
                destImg2[y][x] = int(0)
            elif (img[y][x] in groups[3]):
                destImg3[y][x] = int(0)

    return [destImg0, destImg1, destImg2, destImg3]

def process_kmeans(imgName, filter_with_median):
    '''Executa o algoritmo k-means com k=2 a fim de binarizar a imagem'''
    start_time = time.time()
    desImgName = imgName
    imgName = sys.path[0] + '\\imgs\\' + imgName
    processedFolder = sys.path[0] + '\\' + 'processed'

    if not os.path.exists(processedFolder):
        os.makedirs(processedFolder)

    img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)
    if (filter_with_median):
        img = median_filter(img, 7)

    print("Processing image '{0}'...".format(imgName))

    destImgName = processedFolder + "\\" + desImgName
    groups = kmeans(img)
    imgDest = make_segmented_image(groups, img)

    cv2.imwrite(destImgName, imgDest.astype('uint8') * 255)
    elapsed_time = time.time() - start_time
    print("Done! Elapsed Time: {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

def process_kmeans4(imgName, filter_with_median):
    '''Executa o k-means com k=4, ou seja, 4 centróides, resultado em 4 grupos do histograma da imagem'''
    start_time = time.time()
    desImgName = imgName
    imgName = sys.path[0] + '\\imgs\\' + imgName
    processedFolder = sys.path[0] + '\\' + 'processed'

    if not os.path.exists(processedFolder):
        os.makedirs(processedFolder)

    img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)
    print("Processing image '{0}'...".format(imgName))
    if (filter_with_median):
        img = median_filter(img, 7)
        imgName = "with_median_" + imgName

    groups = kmeans4groups(img)
    # print(groups[0])
    # print(groups[1])
    # print(groups[2])
    # print(groups[3])
    imgs = split_kmeans(groups, img)

    destImgName = processedFolder + "\\1_" + desImgName
    cv2.imwrite(destImgName, imgs[0].astype('uint8') * 255)

    destImgName = processedFolder + "\\2_" + desImgName
    cv2.imwrite(destImgName, imgs[1].astype('uint8') * 255)

    destImgName = processedFolder + "\\3_" + desImgName
    cv2.imwrite(destImgName, imgs[2].astype('uint8') * 255)

    destImgName = processedFolder + "\\4_" + desImgName
    cv2.imwrite(destImgName, imgs[3].astype('uint8') * 255)

    elapsed_time = time.time() - start_time
    print("Done! Elapsed Time: {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

''' Programa Principal '''
process_edge_detection("image_(1).jpg", False)
process_edge_detection("Image_(1a).jpg", True)
process_edge_detection("Image_(2a).jpg", False)
process_kmeans("Image_(3a).jpg", False)
process_kmeans("Image_(3b).jpg", True)
process_edge_detection("Image_(4).jpg", False)
process_kmeans("Image_(4).jpg", False) #tentativa
process_kmeans4("Image_(5).jpg", False)