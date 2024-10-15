import cv2
import numpy as np


class ArucoMarkerWarpApp:
    def __init__(
        self,
        aruco_dict: cv2.aruco.Dictionary,
        aruco_params: cv2.aruco.DetectorParameters,
        img: np.ndarray,
    ):
        self.aruco_dict = aruco_dict
        self.aruco_params = aruco_params
        self.img = img
        self.cap = cv2.VideoCapture(0)
        self.src_points = self.get_src_points()

    def get_src_points(self) -> np.ndarray:
        height, width = self.img.shape[:2]
        return np.array(
            [
                [0, 0],
                [width, 0],
                [width, height],
                [0, height],
            ]
        )

    def normalize_points(self, pts: np.ndarray) -> np.ndarray:
        c = np.mean(pts, axis=0)
        s = np.sqrt(2) / np.std(pts)
        T = np.array(
            [
                [s, 0, -c[0] * s],
                [0, s, -c[1] * s],
                [0, 0, 1],
            ]
        )
        return T

    def compute_A_matrix(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        T1: np.ndarray,
        T2: np.ndarray,
    ) -> np.ndarray:
        A = []
        for i in range(4):
            pts1_n = np.dot(T1, np.append(src_points[i], 1))
            pts2_n = np.dot(T2, np.append(dst_points[i], 1))
            x1, y1 = pts1_n[0], pts1_n[1]
            x2, y2 = pts2_n[0], pts2_n[1]
            A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])
            A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        return np.array(A)

    def compute_homography(
        self, src_points: np.ndarray, dst_points: np.ndarray
    ) -> np.ndarray:
        T1 = self.normalize_points(src_points)
        T2 = self.normalize_points(dst_points)
        A = self.compute_A_matrix(src_points, dst_points, T1, T2)
        _, _, V = np.linalg.svd(A)
        H_normalized = np.reshape(V[-1], (3, 3))
        H = np.dot(np.linalg.inv(T2), np.dot(H_normalized, T1))
        return H

    def apply_perspective_warp(
        self, frame: np.ndarray, homography: np.ndarray, dst_points: np.ndarray
    ) -> np.ndarray:
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        warped_image = cv2.warpPerspective(
            self.img,
            homography,
            (frame.shape[1], frame.shape[0]),
        )
        cv2.fillConvexPoly(
            img=mask,
            points=np.int32([dst_points]),
            color=(255, 255, 255),
            lineType=cv2.LINE_AA,
        )
        mask_eroded = cv2.erode(
            src=mask,
            kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=3,
        )
        mask3 = np.zeros_like(warped_image)
        for i in range(0, 3):
            mask3[:, :, i] = mask_eroded / 255
        warped_image_masked = warped_image * mask3
        frame_masked = frame * (1 - mask3)
        return cv2.add(warped_image_masked, frame_masked)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            image=gray_frame,
            dictionary=self.aruco_dict,
            parameters=self.aruco_params,
        )
        frame = cv2.aruco.drawDetectedMarkers(frame, corners)

        if ids is not None:
            for corner in corners:
                dst_points = corner[0]
                homography = self.compute_homography(self.src_points, dst_points)
                return self.apply_perspective_warp(frame, homography, dst_points)

        return frame

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_with_warp = self.process_frame(frame)
            cv2.imshow("Warped Image", frame_with_warp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    aruco_params = cv2.aruco.DetectorParameters()
    img = cv2.imread("img/image.jpg")
    app = ArucoMarkerWarpApp(aruco_dict, aruco_params, img)
    app.run()
