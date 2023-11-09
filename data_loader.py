import cv2
import os
import numpy as np

class data_loading():
    def __init__(self) -> None:
        pass
        
    def frames(self, frame_number):
        
        dataset_seq_00 = "dataset/sequences/00/"
        frame_number = '{:06d}'.format(frame_number)
        left_frame_path = os.path.join(dataset_seq_00, "image_0",
                                        f"{frame_number}.png")
        right_frame_path = os.path.join(dataset_seq_00,"image_1",
                                    f"{frame_number}.png")
        
        right_frame = cv2.imread(right_frame_path)
        left_frame = cv2.imread(left_frame_path)
        return right_frame, left_frame
    
    def ground_truth(self, frame_number, sequence):
        sequence = '{:02d}'.format(sequence)
        path = f"data_odometry_poses\\dataset\\poses\\00.txt"
        file = open(path,'r')
        ground_truth = []
        for line in file:
            line_nums = line.split()
            ground_truth.append(line_nums)
        return ground_truth[frame_number]
        
