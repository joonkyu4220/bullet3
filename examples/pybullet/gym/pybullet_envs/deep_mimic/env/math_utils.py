# reference: https://www.euclideanspace.com/maths/geometry/rotations/conversions/

import math
import numpy as np

eps = 1e-6

def DotProduct(vec_1, vec_2):
    assert len(vec_1) == len(vec_2), "MathUtils DotProduct Dimension Error: Two vectors must have a same dimension"
    out_scalar = 0
    for i in range(len(vec_1)):
        out_scalar += vec_1[i] * vec_2[i]
    return out_scalar

def Normalize(vec):
    # can be vector, quaternion, whatever  
    out_vector = []
    length2 = DotProduct(vec, vec)
    assert length2 > 0, "MathUtils Normalize Error: ZeroVector cannot be normalized"
    length = math.sqrt(length2)
    for i in range(len(vec)):
        out_vector.append(vec[i] / length)
    return np.array(out_vector, dtype=np.float32)

def CrossProduct(vec_1, vec_2):
    assert len(vec_1) == len(vec_2), "MathUtils CrossProduct Dimension Error: Two vectors must have a same dimension"
    vec_dim = len(vec_1)
    assert vec_dim > 2, "MathUtils CrossProduct Dimension Error: The dimension must be larger than 2"
    out_vector = []
    for i in range(vec_dim):
        out_vector.append(vec_1[(i + 1) % vec_dim] * vec_2[(i + 2) % vec_dim] - vec_1[(i + 2) % vec_dim] * vec_2[(i + 1) % vec_dim])
    return np.array(out_vector, dtype=np.float32)

def ScalarVectorProduct(s, vec):
    out_vector = []
    for i in range(len(vec)):
        out_vector.append(s * vec[i])
    return np.array(out_vector, dtype=np.float32)

def VectorSummation(vec_1, vec_2):
    assert len(vec_1) == len(vec_2), "MathUtils VectorSummation Dimension Error: Two vectors must have a same dimension"
    out_vector = []
    for i in range(len(vec_1)):
        out_vector.append(vec_1[i] + vec_2[i])
    return np.array(out_vector, dtype=np.float32)

def VectorSubtraction(vec_1, vec_2):
    assert len(vec_1) == len(vec_2), "MathUtils VectorSubtraction Dimension Error: Two vectors must have a same dimension"
    out_vector = []
    for i in range(len(vec_1)):
        out_vector.append(vec_1[i] - vec_2[i])
    return np.array(out_vector, dtype=np.float32)

def getRotMatFromSixDim(sixdim):
    assert len(sixdim) == 6, "MathUtils getRotMatFromSixDim Dimension Error: Dimension not 6"
    x = sixdim[:3]
    x = Normalize(x)
    y = sixdim[3:]
    y_dot_x = DotProduct(y, x)
    y = VectorSubtraction(y, ScalarVectorProduct(y_dot_x, x))
    y = Normalize(y)
    z = CrossProduct(x, y)
    return np.concatenate([x, y, z], dtype=np.float32).reshape(3, 3)

def Determinant(mat3x3):
    assert mat3x3.shape == (3, 3), "MathUtils Determinant Dimension Error: Dimension not 3x3"
    return DotProduct(mat3x3[0, :], CrossProduct(mat3x3[1, :], mat3x3[2, :]))


def getEulerFromAxisAngle(AxisAngle):
    axis = Normalize(axis[:3])
    angle = AxisAngle[3]
    
    s = math.sin(angle)
    c = math.cos(angle)
    t = 1 - c

    x, y, z = axis[0], axis[1], axis[2]

    if (x * y * t + z * s) > (1.0 - eps):
        yaw = 2 * math.atan2(x * math.sin(angle / 2.0), math.cos(angle / 2.0))
        pitch = - math.pi / 2.0
        roll = 0
    else:
        yaw = math.atan2(y * s - z * x * t, 1 - (y * y + z * z) * t)
        pitch = math.asin(x * y * t + z * s)
        roll = math.atan2(x * s - y * z * t, 1 - (z * z + x * x) * t)

    return np.array([yaw, pitch, roll], dtype=np.float32)

def getQuaternionFromFromAxisAngle(AxisAngle):
    axis = Normalize(AxisAngle[:3])
    angle = AxisAngle[3]
    w = math.cos(angle / 2.0)
    x, y, z = ScalarVectorProduct(math.sin(angle / 2.0), Normalize(axis))
    return np.array([x, y, z, w], dtype=np.float32)


def getAxisAngleFromRotMat(mat3x3):
    assert mat3x3.shape == (3, 3), "MathUtils getAxisAngleFromRotMat Dimension Error: Dimension not 3x3"
    assert abs(Determinant(mat3x3) - 1.0) < eps, "MathUtils getAxisAngleFromRotMat Determinant Error: Determinant not 1"
    angle = math.acos(min(max((mat3x3[0, 0] + mat3x3[1, 1] + mat3x3[2, 2] - 1.0) / 2.0, -1.0), 1.0))

    if abs(angle) < eps:
        return [0.0, 0.0, 1.0, 0.0]
    # if abs(abs(angle) - 180.0) < eps:
    m21 = mat3x3[2, 1] - mat3x3[1, 2]
    m02 = mat3x3[0, 2] - mat3x3[2, 0]
    m10 = mat3x3[1, 0] - mat3x3[0, 1]
    
    denom = math.sqrt(m21 * m21 + m02 * m02 + m10 * m10)
    if abs(denom) < eps:
        x = math.sqrt((mat3x3[0, 0] + 1.0) / 2.0)
        if x < eps:
            x = 0.0
            y = math.sqrt((mat3x3[1, 1] + 1.0) / 2.0)
            if y < eps:
                y = 0.0
                z = 1.0
            else:
                z = (mat3x3[1, 2] + mat3x3[2, 1]) / y / 4.0
        else:
            y = (mat3x3[0, 1] + mat3x3[1, 0]) / x / 4.0
            z = (mat3x3[0, 2] + mat3x3[2, 0]) / x / 4.0
    else:
        x = m21 / denom
        y = m02 / denom
        z = m10 / denom

    x, y, z = Normalize([x, y, z])
    return np.array([x, y, z, angle], dtype=np.float32)

def getAxisAngleFromSixDim(six_dimension):
    return getAxisAngleFromRotMat(getRotMatFromSixDim(six_dimension))

def getSixDimFromRotMat(mat3x3):
    assert mat3x3.shape == (3, 3), "MathUtils getSixDimFromRotMat Dimension Error: Dimension not 3x3"
    return mat3x3.flatten()[:6]

def getRotMatFromEuler(euler):
    assert len(euler) == 3, "MathUtils getRotMatFromEuler Dimension Error: Dimension not 3"

    yaw = euler[0]
    pitch = euler[1]
    roll = euler[2]

    s_y, c_y = math.sin(yaw), math.cos(yaw)
    s_p, c_p = math.sin(pitch), math.cos(pitch)
    s_r, c_r = math.sin(roll), math.cos(roll)

    rot_p = np.array([1,        0,          0,
                      0,        c_p,        - s_p,
                      0,        s_p,        c_p], dtype=np.float32).reshape(3, 3)
    rot_y = np.array([c_y,      - s_y,      0,
                      s_y,      c_y,        0,
                      0,        0,          1], dtype=np.float32).reshape(3, 3)
    rot_r = np.array([c_r,      0,          s_r,
                      0,        1,          0,
                      - s_r,    0,          c_r], dtype=np.float32).reshape(3, 3)
    
    return np.matmul(rot_r, np.matmul(rot_y, rot_p))

def getEulerFromQuaternion(quat):
    quat = Normalize(quat)
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    pole_check = x * y + z * w
    if (pole_check - 0.5 > - eps):
        yaw = 2 * math.atan2(x, w)
        pitch = math.pi / 2
        roll = 0
    elif (pole_check + 0.5 < eps):
        yaw = -2 * math.atan2(x, w)
        pitch = - math.pi / 2
        roll = 0
    else:
        yaw = math.atan2(2*y*w - 2*x*z, 1 - 2*y*y - 2*z*z)
        pitch = math.asin(2 * pole_check)
        roll = math.atan2(2*x*w - 2*y*z, 1 - 2*x*x - 2*z*z)
    return np.array([yaw, pitch, roll], dtype=np.float32)

def getAxisAngleFromQuaternion(quat):
    quat = Normalize(quat)
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
    angle = math.acos(qw) * 2
    ax, ay, az = Normalize([qx, qy, qz])
    return np.array([ax, ay, az, angle], dtype=np.float32)

def getRotVecFromQuaternion(quat):
    quat = Normalize(quat)
    AxisAngle = getAxisAngleFromQuaternion(quat)
    axis = AxisAngle[:3]
    angle = AxisAngle[3]
    return ScalarVectorProduct(angle, axis)
    
def getRotMatFromQuaternion(quat):
    quat = Normalize(quat)
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]

    x2, y2, z2 = 2*x*x, 2*y*y, 2*z*z
    xy, yz, zx = 2*x*y, 2*y*z, 2*z*x
    xw, yw, zw = 2*x*w, 2*y*w, 2*z*w

    rotMat = np.zeros((3, 3), dtype=np.float32)

    rotMat[0, 0] = 1 - y2 - z2
    rotMat[0, 1] = xy - zw
    rotMat[0, 2] = zx + yw
    rotMat[1, 0] = xy + zw
    rotMat[1, 1] = 1 - z2 - x2
    rotMat[1, 2] = yz - xw
    rotMat[2, 0] = zx - yw
    rotMat[2, 1] = yz + xw
    rotMat[2, 2] = 1 - x2 - y2

    return rotMat

def getSixDimFromQuaternion(quat):
    return getSixDimFromRotMat(getRotMatFromQuaternion(quat))

def getAxisAngleFromEuler(euler):
    yaw, pitch, roll = euler[0], euler[1], euler[2]
    c_y, s_y = math.cos(yaw/2), math.sin(yaw/2)
    c_p, s_p = math.cos(pitch/2), math.sin(pitch/2)
    c_r, s_r = math.cos(roll/2), math.sin(roll/2)

    angle = 2 * math.acos(c_y*c_p*c_r - s_y*s_p*s_r)

    x = s_y*s_p*c_r + c_y*c_p*s_r
    y = s_y*c_p*c_r + c_y*s_p*s_r
    z = c_y*s_p*c_r - s_y*c_p*s_r
    x, y, z = Normalize([x, y, z])

    return np.array([x, y, z, angle], dtype=np.float32)

def getAxisAngleFromRotVec(rotVec):
    angle = DotProduct(rotVec, rotVec)
    x, y, z= Normalize(rotVec)
    return np.array([x, y, z, angle], dtype=np.float32)

def getQuaternionFromEuler(euler):
    yaw, pitch, roll = euler[0], euler[1], euler[2]
    c_y, s_y = math.cos(yaw/2), math.sin(yaw/2)
    c_p, s_p = math.cos(pitch/2), math.sin(pitch/2)
    c_r, s_r = math.cos(roll/2), math.sin(roll/2)

    w = c_y*c_p*c_r - s_y*s_p*s_r
    x = s_y*s_p*c_r + c_y*c_p*s_r
    y = s_y*c_p*c_r + c_y*s_p*s_r
    z = c_y*s_p*c_r - s_y*c_p*s_r

    return Normalize([x, y, z, w])

def getQuaternionFromAxisAngle(AxisAngle):
    axis = Normalize(AxisAngle[:3])
    angle = AxisAngle[3]
    ax, ay, az = axis[0], axis[1], axis[2]
    qx = ax * math.sin(angle/2)
    qy = ay * math.sin(angle/2)
    qz = az * math.sin(angle/2)
    qw = math.cos(angle/2)

    return Normalize([qx, qy, qz, qw])

def getQuaternionFromRotVec(rotVec):
    return getQuaternionFromAxisAngle(getAxisAngleFromRotVec(rotVec))

def getQuaternionFromRotMat(rotMat):
    return getQuaternionFromAxisAngle(getAxisAngleFromRotMat(rotMat))

def getQuaternionFromSixDim(sixDim):
    return getQuaternionFromRotMat(getRotMatFromSixDim(sixDim))

def getQuaternionFromAction(action, repr):
    if repr == "Quaternion":
        return action
    elif repr == "Euler":
        return getQuaternionFromEuler(action)
    elif repr == "AxisAngle":
        return getQuaternionFromAxisAngle(action)
    elif repr == "RotVec":
        return getQuaternionFromRotVec(action)
    elif repr == "RotMat":
        return getQuaternionFromRotMat(action)
    elif repr == "6D":
        return getQuaternionFromSixDim(action)

def getStateFromQuaternion(quat, repr):
    if repr == "Quaternion":
        return quat
    elif repr == "Euler":
        return getEulerFromQuaternion(quat)
    elif repr == "AxisAngle":
        return getAxisAngleFromQuaternion(quat)
    elif repr == "RotVec":
        return getRotVecFromQuaternion(quat)
    elif repr == "RotMat":
        return getRotMatFromQuaternion(quat)
    elif repr == "6D":
        return getSixDimFromQuaternion(quat)
    
    