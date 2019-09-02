import numpy as np

def QuinticTimeScaling(Tf, t):
    # t in [0,1], return ratio of scaled time in [0,1]
    return 10 * (1.0 * t/Tf) ** 3 - 15 * (1.0 * t/Tf) ** 4 \
        + 6 * (1.0 * t/Tf) ** 5

def p2pTrajectory(startPoint, endPoint, resolution=1e-3):
    # start and end points are points in joint space. np array of shape (nv, ) (nv is num of joints)
    # resolution is used to calculate num of steps for trajectory
    # 
    # return bool ret and list traj
    # if max distance in joint space coordinate is smaller than 5*resolution, ret = False and no trajectory
    # will be returned. Otherwise return ret = True and smapled trajectory
    # traj length nStep+1, each element is ndarray of shape (nv, )
    #
    nv = startPoint.shape[0]
    dist = np.abs(np.subtract(startPoint, endPoint, dtype = np.float64))
    nStep = int(np.amax(dist) / resolution)
    if nStep > 5:
        ret = True
        scaledTime = []
        traj = [startPoint]
        for i in range(nStep+1):
            ST = QuinticTimeScaling(nStep, i)
            scaledTime.append(ST)
            traj.append(startPoint + ST * (endPoint - startPoint))
        return ret, traj
    else:
        ret = False
        return ret, []

def velocityGeneration(traj, timeStep):
    # given traj of length nStep+1, calculate velocity for each time step
    nStep = len(traj) -1
    velocityCommand = []
    for i in range(nStep):
        velocityCommand.append((traj[i+1] - traj[i]) / timeStep)
    return velocityCommand


