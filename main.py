import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from PnP import minimize_reprojection_error

from data_loader import data_loading
from feature_detection_and_mapping import feat_det_mat
from get_3d_pts import get_three_dim_pts
from landmarks_pos import get_landmarks_coordinates
from plot import AnimatedMap

# Define any helper functions we want

if __name__ == '__main__':

    # Calculating total number of frames
    num_frames = 5000 # Change it

    left_cam_intrinsic = np.block([[718.856, 0, 606.1928],
                                   [0, 718.856, 0],
                                   [0, 0, 1]])
    
    baseline = 431.52705

    distortion_coeff = np.zeros((4, 1))

    #fig, ax = plt.subplots()

    for frame_number in range(num_frames):

        kitti_data = data_loading()
        imgl, imgr = kitti_data.frames(frame_number)
        feature_mapping = feat_det_mat(method='ORB')
        kp1, kp2, matches, img_match = feature_mapping.detect_and_match(imgl=imgl, imgr=imgr)

        cam_frame_3d_pts = get_three_dim_pts()
        pts_3d, pts_2d = cam_frame_3d_pts.compute(kp1, kp2, matches, left_cam_intrinsic, baseline)
        extrin_matrix = minimize_reprojection_error()
        rotation_matrix, translation_vector, extrinsic_matrix = extrin_matrix.least_squares(pts_3d, pts_2d, left_cam_intrinsic, distortion_coeff)
        landmarks_loc = get_landmarks_coordinates()
        landmarks_3d = landmarks_loc.compute(pts_3d, extrinsic_matrix)
        x = landmarks_3d[:,0]
        y = landmarks_3d[:,1]
        z = landmarks_3d[:,2]
        print(x)
        viz = AnimatedMap()
        viz.update_scatter_plot(x, y, z)

        #img_match_rgb = cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB)
        '''ax.clear()
        ax.imshow(img_match_rgb)
        plt.title(f'Matches for frame {frame_number}')
        plt.axis('off')
        plt.draw()
        plt.pause(0.01)'''
    #plt.show()
