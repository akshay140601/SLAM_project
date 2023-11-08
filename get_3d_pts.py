import numpy as np

class get_three_dim_pts():
    def __init__(self) -> None:
        pass

    def compute(self, kp1, kp2, matches, intrinsic_param, baseline):

        # Extract focal lengths and optical centres from intrinsic param matrix
        fx = intrinsic_param[0, 0]
        fy = intrinsic_param[1, 1]
        cx = intrinsic_param[0, 2]
        cy = intrinsic_param[1, 2]
        focal_length = (fx + fy) / 2

        pts_3d = []
        pts_2d = []
        
        for match in matches:
            left_2d = kp1[match.queryIdx].pt
            right_2d = kp2[match.queryIdx].pt

            pts_2d.append(kp2[match.trainIdx].pt)

            # Disparity = x - x'
            disparity = abs(left_2d[0] - right_2d[0])

            #if (disparity != 0):

                # Depth
            z = (focal_length * baseline) / (disparity + 0.0000001)
            x = ((left_2d[0] - cx) * z) / fx
            y = ((left_2d[1] - cy) * z) / fy

            pts_3d.append([x, y, z])

        return np.array(pts_3d), np.array(pts_2d)

