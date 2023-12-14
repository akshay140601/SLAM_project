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
        roll = np.arctan2(initial_poses[i][1][0], initial_poses[i][0][0])
        pitch = np.arctan2(-initial_poses[i][2][0], np.sqrt((initial_poses[i][2][1])**2 + (initial_poses[i][2][2])**2))
        yaw = np.arctan2(initial_poses[i][2][1], initial_poses[i][2][2])
        tx = initial_poses[i][0][3]
        ty = initial_poses[i][1][3]
        tz = initial_poses[i][2][3]
        initial_estimate.insert(X(i), pose.compose(gtsam.Pose3(
            gtsam.Rot3.Rodrigues(roll, pitch, yaw), gtsam.Point3(tx, ty, tz))))
        
        if i == 0:
            # Add a prior
            pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(
                [0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))  # 30cm std on x, y, z, 0.1 rad on roll, pitch, yaw
            graph.add(gtsam.PriorFactorPose3(X(0), poses_initial[0], pose_noise))

        else:
            odometry_factor = gtsam.BetweenFactorPose3(X(i - 1), X(i), poses_initial[i - 1].between(poses_initial[i]),
                                                        gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])))
            graph.add(odometry_factor)

    if len(loop_closure_frames) != 0:
        len_list = int(len(loop_closure_frames) / 2)
        #print('Starting loop closure')
        loop_closure_frames = np.array(loop_closure_frames).reshape(len_list, 2)
        #print(loop_closure_frames)
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


def Powells_dog_leg_optimization(initial_poses, loop_closure_frames):

    poses_initial = [gtsam.Pose3(gtsam.Rot3(pose[:3, :3]), gtsam.Point3(pose[:, 3])) for pose in initial_poses]
    updated_poses = np.zeros((initial_poses.shape[0], initial_poses.shape[1], initial_poses.shape[2]))

    # Create the factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    for i, pose in enumerate(poses_initial):
        '''roll = np.arctan2(initial_poses[i][1][0], initial_poses[i][0][0])
        pitch = np.arctan2(-initial_poses[i][2][0], np.sqrt((initial_poses[i][2][1])**2 + (initial_poses[i][2][2])**2))
        yaw = np.arctan2(initial_poses[i][2][1], initial_poses[i][2][2])
        tx = initial_poses[i][0][3]
        ty = initial_poses[i][1][3]
        tz = initial_poses[i][2][3]'''
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

    if len(loop_closure_frames) != 0:
        len_list = int(len(loop_closure_frames) / 2)
        #print('Starting loop closure')
        loop_closure_frames = np.array(loop_closure_frames).reshape(len_list, 2)
        #print(loop_closure_frames)
        # Adding the loop closures
        for i in range(loop_closure_frames.shape[0]):
            first_frame = loop_closure_frames[i, 0]
            second_frame = loop_closure_frames[i, 1]
            loop_closure_factor = gtsam.BetweenFactorPose3(X(first_frame), X(second_frame), poses_initial[first_frame].between(poses_initial[second_frame]),
                                                        gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])))
            graph.add(loop_closure_factor)
    
    # Optimize using Levenberg-Marquardt
    parameters = gtsam.DoglegParams()
    optimizer = gtsam.DoglegOptimizer(graph, initial_estimate, parameters)
    result = optimizer.optimize()
    
    for i in range(initial_poses.shape[0]):
        ith_pose = result.atPose3(X(i)).matrix()
        updated_poses[i] = ith_pose[:3, :]

    return updated_poses

def ISAM2(initial_poses, loop_closure_frames):
    """Perform 3D SLAM given ground truth poses as well as simple
    loop closure detection."""
    lc_frames = loop_closure_frames.copy()
    updated_poses = np.zeros((initial_poses.shape[0], initial_poses.shape[1], initial_poses.shape[2]))

    # Declare the 3D translational standard deviations of the prior factor's Gaussian model, in meters.
    prior_xyz_sigma = 0.0001

    # Declare the 3D rotational standard deviations of the prior factor's Gaussian model, in degrees.
    prior_rpy_sigma = 0.0001

    # Declare the 3D translational standard deviations of the odometry factor's Gaussian model, in meters.
    odometry_xyz_sigma = 0.01

    # Declare the 3D rotational standard deviations of the odometry factor's Gaussian model, in degrees.
    odometry_rpy_sigma = 0.1

    # Although this example only uses linear measurements and Gaussian noise models, it is important
    # to note that iSAM2 can be utilized to its full potential during nonlinear optimization. This example
    # simply showcases how iSAM2 may be applied to a Pose2 SLAM problem.
    PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_rpy_sigma*np.pi/180,
                                                                prior_rpy_sigma*np.pi/180,
                                                                prior_rpy_sigma*np.pi/180,
                                                                prior_xyz_sigma,
                                                                prior_xyz_sigma,
                                                                prior_xyz_sigma]))
    ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([odometry_rpy_sigma*np.pi/180,
                                                                odometry_rpy_sigma*np.pi/180,
                                                                odometry_rpy_sigma*np.pi/180,
                                                                odometry_xyz_sigma,
                                                                odometry_xyz_sigma,
                                                                odometry_xyz_sigma]))

    # Create a Nonlinear factor graph as well as the data structure to hold state estimates.
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # Create iSAM2 parameters which can adjust the threshold necessary to force relinearization and how many
    # update calls are required to perform the relinearization.
    parameters = gtsam.ISAM2Params()
    parameters.setRelinearizeThreshold(0.01)
    parameters.relinearizeSkip = 1
    isam = gtsam.ISAM2(parameters)

    # Create the ground truth poses of the robot trajectory.
    true_poses = [gtsam.Pose3(pose) for pose in initial_poses]

    # Create the ground truth odometry transformations, xyz translations, and roll-pitch-yaw rotations
    # between each robot pose in the trajectory.
    odometry_tf = [true_poses[i-1].transformPoseTo(true_poses[i]) for i in range(1, len(true_poses))]
    odometry_xyz = [(odometry_tf[i].x(), odometry_tf[i].y(), odometry_tf[i].z()) for i in range(len(odometry_tf))]
    odometry_rpy = [odometry_tf[i].rotation().rpy() for i in range(len(odometry_tf))]

    # Corrupt xyz translations and roll-pitch-yaw rotations with gaussian noise to create noisy odometry measurements.
    noisy_measurements = [np.random.multivariate_normal(np.hstack((odometry_rpy[i],odometry_xyz[i])), \
                                                        ODOMETRY_NOISE.covariance()) for i in range(len(odometry_tf))]

    # Add the prior factor to the factor graph, and poorly initialize the prior pose to demonstrate
    # iSAM2 incremental optimization.
    graph.push_back(gtsam.PriorFactorPose3(1, true_poses[0], PRIOR_NOISE))
    initial_estimate.insert(1, true_poses[0].compose(gtsam.Pose3(
        gtsam.Rot3.Rodrigues(0, 0, 0), gtsam.Point3(0, 0, 0))))

    # Initialize the current estimate which is used during the incremental inference loop.
    current_estimate = initial_estimate
    for i in range(len(odometry_tf)):

        # Obtain the noisy translation and rotation that is received by the robot and corrupted by gaussian noise.
        noisy_odometry = noisy_measurements[i]

        # Compute the noisy odometry transformation according to the xyz translation and roll-pitch-yaw rotation.
        noisy_tf = gtsam.Pose3(gtsam.Rot3.RzRyRx(noisy_odometry[:3]), noisy_odometry[3:6].reshape(-1,1))

        # Add a binary factor in between two existing states if loop closure is detected.
        # Otherwise, add a binary factor between a newly observed state and the previous state.
        if len(lc_frames) != 0:
            if i == lc_frames[0]:
                graph.push_back(gtsam.BetweenFactorPose3(i + 1, lc_frames[1] + 1, noisy_tf, ODOMETRY_NOISE))
                del lc_frames[0:3]
                graph.push_back(gtsam.BetweenFactorPose3(i + 1, i + 2, noisy_tf, ODOMETRY_NOISE))
                noisy_estimate = current_estimate.atPose3(i + 1).compose(noisy_tf)
                initial_estimate.insert(i + 2, noisy_estimate)
            else:
                graph.push_back(gtsam.BetweenFactorPose3(i + 1, i + 2, noisy_tf, ODOMETRY_NOISE))

                # Compute and insert the initialization estimate for the current pose using a noisy odometry measurement.
                noisy_estimate = current_estimate.atPose3(i + 1).compose(noisy_tf)
                initial_estimate.insert(i + 2, noisy_estimate)

        else:
            graph.push_back(gtsam.BetweenFactorPose3(i + 1, i + 2, noisy_tf, ODOMETRY_NOISE))

            # Compute and insert the initialization estimate for the current pose using a noisy odometry measurement.
            noisy_estimate = current_estimate.atPose3(i + 1).compose(noisy_tf)
            initial_estimate.insert(i + 2, noisy_estimate)

        # Perform incremental update to iSAM2's internal Bayes tree, optimizing only the affected variables.
        isam.update(graph, initial_estimate)
        current_estimate = isam.calculateEstimate()
        initial_estimate.clear()

    '''params = gtsam.DoglegParams()
    params.setMaxIterations(150)
    params.setRelativeErrorTol(1e-3)

    optimizer = gtsam.gtsam.DoglegOptimizer(graph, initial_estimate, params)
    current_estimate = optimizer.optimize()'''
    #current_estimate = isam.calculateEstimate()
    #print(current_estimate)
    for i in range(1, initial_poses.shape[0]+1):
        #current_estimate = isam.calculateEstimate()
        ith_pose = current_estimate.atPose3(i).matrix()
        updated_poses[i-1] = ith_pose[:3, :]

    return updated_poses

def LM(initial_poses, loop_closure_frames):
    """Perform 3D SLAM given ground truth poses as well as simple
    loop closure detection."""
    lc_frames = loop_closure_frames.copy()
    updated_poses = np.zeros((initial_poses.shape[0], initial_poses.shape[1], initial_poses.shape[2]))

    prior_xyz_sigma = 1e-4
    prior_rpy_sigma = 1e-6
    odometry_xyz_sigma = 0.001
    odometry_rpy_sigma = 0.0001
    PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_rpy_sigma*np.pi/180,
                                                                prior_rpy_sigma*np.pi/180,
                                                                prior_rpy_sigma*np.pi/180,
                                                                prior_xyz_sigma,
                                                                prior_xyz_sigma,
                                                                prior_xyz_sigma]))
    ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([odometry_rpy_sigma*np.pi/180,
                                                                odometry_rpy_sigma*np.pi/180,
                                                                odometry_rpy_sigma*np.pi/180,
                                                                odometry_xyz_sigma,
                                                                odometry_xyz_sigma,
                                                                odometry_xyz_sigma]))

    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    true_poses = [gtsam.Pose3(pose) for pose in initial_poses]

    odometry_tf = [true_poses[i-1].transformPoseTo(true_poses[i]) for i in range(1, len(true_poses))]
    odometry_xyz = [(odometry_tf[i].x(), odometry_tf[i].y(), odometry_tf[i].z()) for i in range(len(odometry_tf))]
    odometry_rpy = [odometry_tf[i].rotation().rpy() for i in range(len(odometry_tf))]

    noisy_measurements = [np.random.multivariate_normal(np.hstack((odometry_rpy[i],odometry_xyz[i])), \
                                                        ODOMETRY_NOISE.covariance()) for i in range(len(odometry_tf))]


    graph.push_back(gtsam.PriorFactorPose3(1, true_poses[0], PRIOR_NOISE))
    initial_estimate.insert(1, true_poses[0].compose(gtsam.Pose3(
        gtsam.Rot3.Rodrigues(0, 0, 0), gtsam.Point3(0, 0, 0))))

    current_estimate = initial_estimate
    for i in range(len(odometry_tf)):

        noisy_odometry = noisy_measurements[i]

        noisy_tf = gtsam.Pose3(gtsam.Rot3.RzRyRx(noisy_odometry[:3]), noisy_odometry[3:6].reshape(-1,1))

        #noisy_tf = gtsam.Pose3(gtsam.Rot3.RzRyRx(odometry_rpy[i]), odometry_xyz[i])

        if len(lc_frames) != 0:
            if i == lc_frames[0]:
                graph.push_back(gtsam.BetweenFactorPose3(i + 1, lc_frames[1] + 1, noisy_tf, ODOMETRY_NOISE))
                del lc_frames[0:3]
                graph.push_back(gtsam.BetweenFactorPose3(i + 1, i + 2, noisy_tf, ODOMETRY_NOISE))
                noisy_estimate = current_estimate.atPose3(i + 1).compose(noisy_tf)
                initial_estimate.insert(i + 2, noisy_estimate)
            else:
                graph.push_back(gtsam.BetweenFactorPose3(i + 1, i + 2, noisy_tf, ODOMETRY_NOISE))
                noisy_estimate = current_estimate.atPose3(i + 1).compose(noisy_tf)
                initial_estimate.insert(i + 2, noisy_estimate)

        else:
            graph.push_back(gtsam.BetweenFactorPose3(i + 1, i + 2, noisy_tf, ODOMETRY_NOISE))
            noisy_estimate = current_estimate.atPose3(i + 1).compose(noisy_tf)
            initial_estimate.insert(i + 2, noisy_estimate)

    '''parameters = gtsam.LevenbergMarquardtParams()
    parameters.setRelativeErrorTol(1e-5)
    parameters.setAbsoluteErrorTol(1e-5)

    parameters.setlambdaInitial(1e-1)
    parameters.setlambdaFactor(100.0)
    #parameters.setlambdaFactorFailed(100.0)
    parameters.setlambdaUpperBound(1e100)
    parameters.setlambdaLowerBound(1e-100)
    parameters.setMaxIterations(200)'''
    parameters = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, parameters)
    current_estimate = optimizer.optimize()

    for i in range(1, initial_poses.shape[0]+1):
        ith_pose = current_estimate.atPose3(i).matrix()
        updated_poses[i-1] = ith_pose[:3, :]

    return updated_poses
