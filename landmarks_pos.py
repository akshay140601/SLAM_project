import numpy as np

class get_landmarks_coordinates():
    def __init__(self) -> None:
        pass

    def compute(self, pts_3d, extrinsic_mat):

        pts_3d_world_frame = []

        for pts in pts_3d:
            pts_3d_homo = np.array([pts[0], pts[1], pts[2], 1])
            pts_transformed_homo = np.linalg.inv(extrinsic_mat) @ pts_3d_homo
            pts_3d_world_frame.append(pts_transformed_homo[:3])

        return np.array(pts_3d_world_frame)