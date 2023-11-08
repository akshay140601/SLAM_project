import cv2
import numpy as np

class minimize_reprojection_error():
    def __init__(self) -> None:
        pass

    def least_squares(self, pts_3d, kp1, intrinsic_params, distortion_coeff):

        _, rvec, tvec = cv2.solvePnP(pts_3d, kp1, intrinsic_params, distortion_coeff, flags=cv2.SOLVEPNP_EPNP)

        rot_matrix, _ = cv2.Rodrigues(rvec)
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = rot_matrix
        cam_pose[:3, 3] = tvec.flatten()

        return rot_matrix, tvec, cam_pose