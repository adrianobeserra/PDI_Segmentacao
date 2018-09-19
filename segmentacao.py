import cv2
import numpy as np
import time
import sys
import os

'''Implementação do filtro de mediana.'''
def get_median(list):
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
    #img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

''' Programa Principal '''
process_edge_detection("image_(1).jpg", False)
process_edge_detection("Image_(1a).jpg", True)
