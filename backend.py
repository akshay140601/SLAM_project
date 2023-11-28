import gtsam
import matplotlib.pyplot as plt
import numpy as np
from gtsam.symbol_shorthand import X

'''def gtsam_isam2(initial_poses):

    poses_initial = [gtsam.Pose3(gtsam.Rot3(pose[:3, :3]), gtsam.Point3(pose[:, 3])) for pose in initial_poses]
    updated_poses = np.zeros((initial_poses.shape[0], initial_poses.shape[1], initial_poses.shape[2]))

    # Create the iSAM2 object
    parameters = gtsam.ISAM2Params()
    parameters.setRelinearizeThreshold(100)
    parameters.relinearizeSkip = 10000
    isam = gtsam.ISAM2(parameters)

    # Creating the factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    for i, pose in enumerate(poses_initial):
        initial_estimate.insert(X(i), pose.compose(gtsam.Pose3(
            gtsam.Rot3.Rodrigues(-0.1, 0.2, 0.25), gtsam.Point3(0, 0, 0))))
        
        if i == 0:
            # Add a prior
            pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(
                [0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))  # 30cm std on x, y, z, 0.1 rad on roll, pitch, yaw
            graph.push_back(gtsam.PriorFactorPose3(X(0), poses_initial[0], pose_noise))

        else:
            odometry_factor = gtsam.BetweenFactorPose3(X(i - 1), X(i), poses_initial[i - 1].between(poses_initial[i]),
                                                        gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 3000000, 300000000, 3000000000])))
            graph.push_back(odometry_factor)
            # Updation
            isam.update(graph, initial_estimate)
            isam.update()

            current_estimate = isam.calculateEstimate()
            ith_pose = current_estimate.atPose3(X(i)).matrix()
            updated_poses[i] = ith_pose[:3, :]

            # Clear the factor graph
            graph.resize(0)
            initial_estimate.clear()

    return updated_poses
'''

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from gtsam.symbol_shorthand import X

def levenberg_marquardt_optimization(initial_poses, loop_closure_frames):

    poses_initial = [gtsam.Pose3(gtsam.Rot3(pose[:3, :3]), gtsam.Point3(pose[:, 3])) for pose in initial_poses]
    updated_poses = np.zeros((initial_poses.shape[0], initial_poses.shape[1], initial_poses.shape[2]))

    # Create the factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    for i, pose in enumerate(poses_initial):
        initial_estimate.insert(X(i), pose.compose(gtsam.Pose3(
            gtsam.Rot3.Rodrigues(-0.1, 0.2, 0.25), gtsam.Point3(0, 0, 0))))
        
        if i == 0:
            # Add a prior
            pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(
                [0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))  # 30cm std on x, y, z, 0.1 rad on roll, pitch, yaw
            graph.add(gtsam.PriorFactorPose3(X(0), poses_initial[0], pose_noise))

        else:
            odometry_factor = gtsam.BetweenFactorPose3(X(i - 1), X(i), poses_initial[i - 1].between(poses_initial[i]),
                                                        gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])))
            graph.add(odometry_factor)

    len_list = len(loop_closure_frames) / 2
    loop_closure_frames = np.array(loop_closure_frames).reshape(len_list, 2)
    # Adding the loop closures
    for i in range(loop_closure_frames.shape[0]):
        first_frame = loop_closure_frames[i, 0]
        second_frame = loop_closure_frames[i, 1]
        loop_closure_factor = gtsam.BetweenFactorPose3(X(first_frame), X(second_frame), poses_initial[first_frame].between(poses_initial[second_frame]),
                                                       gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])))
        graph.add(loop_closure_factor)
    
    # Optimize using Levenberg-Marquardt
    parameters = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, parameters)
    result = optimizer.optimize()
    
    for i in range(initial_poses.shape[0]):
        ith_pose = result.atPose3(X(i)).matrix()
        updated_poses[i] = ith_pose[:3, :]

    return updated_poses
