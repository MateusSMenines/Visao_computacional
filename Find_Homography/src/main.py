# TRABALHO 2 DE VISÃO COMPUTACIONAL
# Nome: Mateus Sobrinho Menines

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import matplotlib
import math
import random
import cv2 as cv


# Função para normalizar pontos
def normalize_points(pts):


    #centroid
    c = np.mean(pts,0)


    # T = ([s 0 -xc*s] [0 s -yc*s] [0 0 1]) onde s = sqrt(2)/desvio padrão
    s = np.sqrt(2)/(np.std(pts))

    T = np.array([[s, 0, (-c[0]*s)],
                  [0, s, (-c[1]*s)], 
                  [0, 0, 1]])


    return T


# Função para montar a matriz A do sistema de equações do DLT
def compute_A(corrs, T1, T2):


    A = []


    #normalizando os pontos da correspondecia
    for i in range(len(corrs)):

        pts1 = np.array([[corrs[i][0], corrs[i][1], 1]]).T
        pts2 = np.array([[corrs[i][2], corrs[i][3], 1]]).T

        pts1_n = np.dot(T1,pts1)
        pts2_n = np.dot(T2,pts2)

        x1 = pts1_n[0][0]
        y1 = pts1_n[1][0]
        x2 = pts2_n[0][0]
        y2 = pts2_n[1][0]

        A.append([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])
        A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])

    A = np.array(A)


    return A


# Função do DLT Normalizado
def compute_normalized_dlt(A,T1,T2):


    # Calculando SVD
    U, S, V = np.linalg.svd(A)


    # Homografia H
    H_n = np.reshape(V[8],(3, 3))
    

    # Denormaliza H_normalizada 
    H = np.dot(np.linalg.inv(T2),np.dot(H_n,T1))


    return H


# função para calculo de distancia
def distance(corrs, H):


    pts1 = np.matrix([corrs[0], corrs[1], 1]).T
    pts2 = np.matrix([corrs[2], corrs[3], 1]).T

    estimate = np.dot(H,pts1)
    estimate = (estimate/estimate[2])

    error = pts2 - estimate

    dis = np.linalg.norm(error)

    return dis


# função para transformar pontos em array 
def array(pts1, pts2):


    pts1_list = []
    pts2_list = []


    #transformando dados em arrays 
    for i in range(len(pts1)):

        pts1_list.append([pts1[i][0][0],pts1[i][0][1]])
        pts2_list.append([pts2[i][0][0],pts2[i][0][1]])


    pts1 = np.array(pts1_list) # array dos pontos 1 
    pts2 = np.array(pts2_list) # array dos pontos 2

    return pts1, pts2


# função Ransac
def RANSAC(pts1, pts2):


    dis_threshold = 0.9
    N = 999999999
    N_done = 0
    Ninl = []
    inlier = []
    p = 0.99

    corrs_list = []


    # função de transfomação dos dados em um array
    pts1, pts2 = array(pts1,pts2)


    # função para obter matriz T e T'
    T1 = normalize_points(pts1)
    T2 = normalize_points(pts2)


    # criando array de correspondencia
    for i in range(len(pts1)):

        corrs_list.append([pts1[i][0],pts1[i][1],pts2[i][0],pts2[i][1]])

    corrs = np.array(corrs_list) # array de correspondencia


    # metodo iterativo de RANSAC
    while N > N_done:

        inlier = []

        pts_r1 = corrs[random.randrange(0, len(corrs))]
        pts_r2 = corrs[random.randrange(0, len(corrs))]
        pts_r3 = corrs[random.randrange(0, len(corrs))]
        pts_r4 = corrs[random.randrange(0, len(corrs))]

        pts_r = np.vstack((pts_r1,pts_r2))
        pts_r = np.vstack((pts_r,pts_r3))
        pts_r = np.vstack((pts_r,pts_r4))


        # calculando matriz A
        A = compute_A(pts_r, T1, T2)


        # calculando matriz H
        H = compute_normalized_dlt(A, T1, T2)
        

        # calculo do erro e contagem de inlier
        for i in range(len(corrs)):
            
            dis = distance(corrs[i],H)
            
            if dis < 5:
                inlier.append(corrs[i])


        # verificando condição dos inliers
        if len(inlier) > len(Ninl):
            
            Ninl = inlier
            finalH = H

        print (f"corrs size: {len(corrs)}     inliers found: {len(inlier)}      max inliers found: {len(Ninl)}     N: {int(N)}     N_done: {N_done}")
    

        if len(Ninl) > (len(corrs)*dis_threshold):
            break


        # atualizando o numero de interação
        e = 1 - len(Ninl)/len(corrs)
        N = (math.log(1-p,10))/(math.log(1-math.pow(1-e,3),10))
        N_done = N_done + 1



    # calculando A com o novo array de inliers
    inlier = np.array(Ninl)
    A = compute_A(inlier, T1, T2)


    # calculando H com o novo array de inlier
    H = compute_normalized_dlt(A, T1, T2)


    return H


########################################################################################################################
# Exemplo de Teste da função de homografia usando o SIFT


MIN_MATCH_COUNT = 10
img1 = cv.imread('../images/minions03.jpg', 0)   # queryImage
img2 = cv.imread('../images/minions01a.jpg', 0)        # trainImage


# Inicialização do SIFT
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


# FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)


    #################################################
    M = RANSAC(src_pts, dst_pts)
    #################################################


    img4 = cv.warpPerspective(img1, M, (img1.shape[1], img1.shape[0])) 

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

fig, axs = plt.subplots(2, 2, figsize=(30, 15))
fig.add_subplot(2, 2, 1)
plt.imshow(img3, 'gray')
fig.add_subplot(2, 2, 2)
plt.title('Primeira imagem')
plt.imshow(img1, 'gray')
fig.add_subplot(2, 2, 3)
plt.title('Segunda imagem')
plt.imshow(img2, 'gray')
fig.add_subplot(2, 2, 4)
plt.title('Primeira imagem após transformação')
plt.imshow(img4, 'gray')
plt.show()

########################################################################################################################
