import numpy as np
import numpy.linalg as la
import math


# Checks if a matrix is a valid rotation matrix.
def isRotm(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Convert quaternion to rotation matrix
def quaternion2Rotation(q):
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    R = np.array([[1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                  [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
                  [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]])
    if isRotm(R):
        return R
    else:
        raise Exception('R is not a rotation matrix, please check your quaternions')


# Convert rotation matrix to quaternion
def rotation2Quaternion(R):
    assert(isRotm(R))
    r11 = R[0][0]
    r12 = R[0][1]
    r13 = R[0][2]
    r21 = R[1][0]
    r22 = R[1][1]
    r23 = R[1][2]
    r31 = R[2][0]
    r32 = R[2][1]
    r33 = R[2][2]

    # computing four sets of solutions
    qw_1 = math.sqrt(1 + r11 + r22 + r33)
    u1 = 1/2 * np.array([qw_1,
                         (r32-r23)/qw_1,
                         (r13-r31)/qw_1,
                         (r21-r12)/qw_1
                         ])

    qx_2 = math.sqrt(1 + r11 - r22 - r33)
    u2 = 1/2 * np.array([(r32-r23)/qx_2,
                         qx_2,
                         (r12+r21)/qx_2,
                         (r31+r13)/qx_2
                         ])

    qy_3 = math.sqrt(1 - r11 + r22 - r33)
    u3 = 1/2 * np.array([(r13-r31)/qy_3,
                         (r12+r21)/qy_3,
                         qy_3,
                         (r23+r32)/qy_3
                         ])

    qz_4 = math.sqrt(1 - r11 - r22 + r33)
    u4 = 1/2* np.array([(r21-r12)/qz_4,
                        (r31+r13)/qz_4,
                        (r32+r23)/qz_4,
                        qz_4
                        ])

    U = [u1,u2,u3,u4]

    idx = np.array([r11+r22+r33, r11, r22, r33]).argmax()
    q = U[idx]
    if (la.norm(q) - 1) < 1e-3:
        return q
    else:
        raise Exception('Quaternion is not normalized, please check your rotation matrix')


def fitSlope(Y, X):
    # Y is a list of numbers, could be joint values or sensor readings
    # X is time
    z = np.polyfit(X, Y, 1)
    slope = z[0,:]
    return slope

def Normalize(V):
    # Normalizes a vector
    return V / np.linalg.norm(V)

def NearZero(z):
    # Determines whether a scalar is small enough to be treated as zero
    return abs(z) < 1e-6

def VecToso3(omg):
    """Converts a 3-vector to an so(3) representation

    :param omg: A 3-vector
    :return: The skew symmetric representation of omg

    Example Input:
        omg = np.array([1, 2, 3])
    Output:
        np.array([[ 0, -3,  2],
                  [ 3,  0, -1],
                  [-2,  1,  0]])
    """
    return np.array([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])

def so3ToVec(so3mat):
    """Converts an so(3) representation to a 3-vector

    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The 3-vector corresponding to so3mat

    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([1, 2, 3])
    """
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

def AxisAng3(expc3):
    """Converts a 3-vector of exponential coordinates for rotation into
    axis-angle form

    :param expc3: A 3-vector of exponential coordinates for rotation
    :return omghat: A unit rotation axis
    :return theta: The corresponding rotation angle

    Example Input:
        expc3 = np.array([1, 2, 3])
    Output:
        (np.array([0.26726124, 0.53452248, 0.80178373]), 3.7416573867739413)
    """
    return (Normalize(expc3), np.linalg.norm(expc3))

def MatrixExp3(so3mat):
    """Computes the matrix exponential of a matrix in so(3)

    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The matrix exponential of so3mat

    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([[-0.69492056,  0.71352099,  0.08929286],
                  [-0.19200697, -0.30378504,  0.93319235],
                  [ 0.69297817,  0.6313497 ,  0.34810748]])
    """
    omgtheta = so3ToVec(so3mat)
    if NearZero(np.linalg.norm(omgtheta)):
        return np.eye(3)
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat \
               + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)

def MatrixLog3(R):
    """Computes the matrix logarithm of a rotation matrix

    :param R: A 3x3 rotation matrix
    :return: The matrix logarithm of R

    Example Input:
        R = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
    Output:
        np.array([[          0, -1.20919958,  1.20919958],
                  [ 1.20919958,           0, -1.20919958],
                  [-1.20919958,  1.20919958,           0]])
    """
    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if not NearZero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                  * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not NearZero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                  * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                  * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return VecToso3(np.pi * omg)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)

def sampleTrajectoryCartesian(P, R, P_target, R_target, resolution = 1e-3):
    dist = np.abs(np.subtract(P, P_target))
    nStep = int(np.amax(dist) / resolution)
    if nStep > 5:
        scaledTime = []
        cartesianTrajectory_P = []
        cartesianTrajectory_R = []
        for i in range(nStep + 1) :
            ST = QuinticTimeScaling(nStep, i)
            P_new = P + (P_target - P) * ST
            R_new = np.dot(R, MatrixExp3(MatrixLog3(np.dot(R.transpose(), R_target)) * ST))
            cartesianTrajectory_P.append(P_new)
            cartesianTrajectory_R.append(R_new)
        return True, cartesianTrajectory_P, cartesianTrajectory_R
    else:
        return False, []



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
        scaledTime = []
        traj = [startPoint]
        for i in range(nStep+1):
            ST = QuinticTimeScaling(nStep, i)
            scaledTime.append(ST)
            traj.append(startPoint + ST * (endPoint - startPoint))
        return True, traj
    else:
        return False, []

def velocityGeneration(traj, timeStep):
    # given traj of length nStep+1, calculate velocity for each time step
    nStep = len(traj) -1
    velocityCommand = []
    for i in range(nStep):
        velocityCommand.append((traj[i+1] - traj[i]) / timeStep)
    return velocityCommand

    
