import cv2
import os
import numpy as np
import pandas as pd
import yaml


with open("config/initial_config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        print(error)

main_path = config['data']['main_path']
pose_path = config['data']['pose_path']


class DataLoader(object):
    def __init__(self, sequence, low_memory=True):
        self.sequence_dir = main_path.format(sequence)
        self.poses_dir = pose_path.format(sequence)
        poses = pd.read_csv(self.poses_dir, delimiter=' ', header=None)
        self.left_camera_images = sorted(
            os.listdir(self.sequence_dir + 'image_0'))
        self.right_camera_images = sorted(
            os.listdir(self.sequence_dir + 'image_1'))

        self.frames = len(self.left_camera_images)

        calibration = pd.read_csv(
            self.sequence_dir + 'calib.txt', delimiter=' ', header=None, index_col=0)

        self.P0 = np.array(calibration.loc['P0:']).reshape((3, 4))
        self.P1 = np.array(calibration.loc['P1:']).reshape((3, 4))

        self.times = np.array(pd.read_csv(self.sequence_dir + 'times.txt',
                                          delimiter=' ',
                                          header=None))

        self.ground_truth = np.zeros((len(poses), 3, 4))
        for i in range(len(poses)):
            self.ground_truth[i] = np.array(poses.iloc[i]).reshape((3, 4))

        self.reset_frames()

        self.first_image_left = cv2.imread(self.sequence_dir + 'image_0/'
                                            + self.left_camera_images[0])
        self.first_image_right = cv2.imread(self.sequence_dir + 'image_1/'
                                            + self.right_camera_images[0])
        self.second_image_left = cv2.imread(self.sequence_dir + 'image_0/'
                                            + self.left_camera_images[1])
        self.second_image_right = cv2.imread(self.sequence_dir + 'image_1/'
                                                + self.right_camera_images[1])

        self.image_height = self.first_image_left.shape[0]
        self.image_width = self.first_image_left.shape[1]

    def reset_frames(self):
        self.left_images = (cv2.imread(self.sequence_dir + 'image_0/' + left, 0)
                            for left in self.left_camera_images)
        self.right_images = (cv2.imread(self.sequence_dir + 'image_1/' + right, 0)
                             for right in self.right_camera_images)
