import cv2
import numpy as np



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
def compute_A(pts1, pts2, T1, T2):


    A = []

    #normalizando os pontos da correspondecia
    for i in range(4):

        apts1 = np.array([[pts1[i][0], pts1[i][1], 1]]).T
        
        apts2 = np.array([[pts2[i][0], pts2[i][1], 1]]).T

        pts1_n = np.dot(T1,apts1)
        pts2_n = np.dot(T2,apts2)

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


def main(aruco_dict, arucoParameters,img):


    cap = cv2.VideoCapture(0)
    [l,c,ch] = np.shape(img)

    pts_src = np.array([[0,0],[c,0],[c,l],[0,l]])
    T1 = normalize_points(pts_src)


    while True:

        ret, frame = cap.read()  

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers( gray, aruco_dict, parameters=arucoParameters)
        image = cv2.aruco.drawDetectedMarkers(frame, corners)

        if len(corners) > 0:

            pts_dst = np.array(corners[0][0])

            T2 = normalize_points(pts_dst)

            A = compute_A(pts_src,pts_dst,T1,T2)

            h = compute_normalized_dlt(A, T1, T2)

            #h, status = cv2.findHomography(pts_src, pts_dst)
            warped_image = cv2.warpPerspective(img, h, (frame.shape[1],frame.shape[0]))

            mask = np.zeros([frame.shape[0], frame.shape[1]])
            cv2.fillConvexPoly(mask, np.int32([pts_dst]), (255, 255, 255), cv2.LINE_AA)

            element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            mask = cv2.erode(mask, element, iterations=3)

            mask3 = np.zeros_like(warped_image)

            for i in range(0, 3):

                mask3[:,:,i] = mask/255

            warped_image_masked = cv2.multiply(warped_image, mask3)
            frame_masked = cv2.multiply(frame, 1-mask3)
            
            im_out = cv2.add(warped_image_masked, frame_masked)

            cv2.imshow("image", im_out)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else: 

            cv2.imshow('image',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    cap.release()
    cv2.destroyAllWindows()
   

if __name__ == '__main__':

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParameters = cv2.aruco.DetectorParameters_create()

    img = cv2.imread('image.jpg')

    main(aruco_dict, arucoParameters, img)
