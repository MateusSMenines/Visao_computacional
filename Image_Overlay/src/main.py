import cv2
import numpy as np


def normalize_points(pts):

    c = np.mean(pts, 0)
    s = np.sqrt(2) / (np.std(pts))
    T = np.array([[s, 0, (-c[0] * s)], [0, s, (-c[1] * s)], [0, 0, 1]])
    return T


def compute_A(pts1, pts2, T1, T2):

    A = []
    for i in range(4):
        apts1 = np.array([[pts1[i][0], pts1[i][1], 1]]).T
        apts2 = np.array([[pts2[i][0], pts2[i][1], 1]]).T
        pts1_n = np.dot(T1, apts1)
        pts2_n = np.dot(T2, apts2)
        x1 = pts1_n[0][0]
        y1 = pts1_n[1][0]
        x2 = pts2_n[0][0]
        y2 = pts2_n[1][0]
        A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
    A = np.array(A)
    return A


def compute_normalized_dlt(A, T1, T2):

    U, S, V = np.linalg.svd(A)
    H_n = np.reshape(V[8], (3, 3))
    H = np.dot(np.linalg.inv(T2), np.dot(H_n, T1))
    return H


def main(aruco_dict, arucoParameters, img):

    cap = cv2.VideoCapture(0)
    [l, c, ch] = np.shape(img)
    pts_src = np.array([[0, 0], [c, 0], [c, l], [0, l]])
    T1 = normalize_points(pts_src)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            image=gray,
            dictionary=aruco_dict,
            parameters=arucoParameters,
        )
        frame = cv2.aruco.drawDetectedMarkers(frame, corners)
        if ids is not None:
            for id, corner in zip(ids, corners):
                pts_dst = corner[0]
                T2 = normalize_points(pts_dst)
                A = compute_A(pts_src, pts_dst, T1, T2)
                h = compute_normalized_dlt(A, T1, T2)
                mask = np.zeros([frame.shape[0], frame.shape[1]])
                warped_image = cv2.warpPerspective(
                    src=img,
                    M=h,
                    dsize=(frame.shape[1], frame.shape[0]),
                )
                cv2.fillConvexPoly(
                    img=mask,
                    points=np.int32([pts_dst]),
                    color=(255, 255, 255),
                    lineType=cv2.LINE_AA,
                )
                element = cv2.getStructuringElement(
                    shape=cv2.MORPH_RECT,
                    ksize=(3, 3),
                )
                mask = cv2.erode(
                    src=mask,
                    kernel=element,
                    iterations=3,
                )
                mask3 = np.zeros_like(warped_image)
                for i in range(0, 3):
                    mask3[:, :, i] = mask / 255
                warped_image_masked = cv2.multiply(warped_image, mask3)
                frame_masked = cv2.multiply(frame, 1 - mask3)
                im_out = cv2.add(warped_image_masked, frame_masked)
                cv2.imshow("image", im_out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            cv2.imshow("image", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParameters = cv2.aruco.DetectorParameters()
    img = cv2.imread("img/image.jpg")
    main(aruco_dict, arucoParameters, img)
