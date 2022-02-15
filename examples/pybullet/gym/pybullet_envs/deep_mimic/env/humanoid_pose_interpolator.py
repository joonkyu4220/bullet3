import numpy as np
from pybullet_utils import bullet_client
import math

eps = 0.000001

class HumanoidPoseInterpolator(object):

  def __init__(self):
    
    # REPRESENTATION_MODE_CHECKPOINT
    self.state_represenation_mode = "QUATERNION"
    self.action_representation_mode = "ANGLE_AXIS"
    self.state_represenation_mode = "6D"
    self.action_representation_mode = "6D"

    pass


  def Reset(self,
            
            basePos=[0, 0, 0],
            baseOrn=[0, 0, 0, 1],

            chestRot=[0, 0, 0, 1],
            neckRot=[0, 0, 0, 1],
            rightHipRot=[0, 0, 0, 1],
            rightKneeRot=[0],
            rightAnkleRot=[0, 0, 0, 1],
            rightShoulderRot=[0, 0, 0, 1],
            rightElbowRot=[0],
            leftHipRot=[0, 0, 0, 1],
            leftKneeRot=[0],
            leftAnkleRot=[0, 0, 0, 1],
            leftShoulderRot=[0, 0, 0, 1],
            leftElbowRot=[0],
            
            baseLinVel=[0, 0, 0],
            baseAngVel=[0, 0, 0],
            
            chestVel=[0, 0, 0],
            neckVel=[0, 0, 0],
            rightHipVel=[0, 0, 0],
            rightKneeVel=[0],
            rightAnkleVel=[0, 0, 0],
            rightShoulderVel=[0, 0, 0],
            rightElbowVel=[0],
            leftHipVel=[0, 0, 0],
            leftKneeVel=[0],
            leftAnkleVel=[0, 0, 0],
            leftShoulderVel=[0, 0, 0],
            leftElbowVel=[0]):

    self._basePos = basePos
    self._baseLinVel = baseLinVel
    #print("HumanoidPoseInterpolator.Reset: baseLinVel = ", baseLinVel)
    self._baseOrn = baseOrn
    self._baseAngVel = baseAngVel

    self._chestRot = chestRot
    self._chestVel = chestVel
    self._neckRot = neckRot
    self._neckVel = neckVel

    self._rightHipRot = rightHipRot
    self._rightHipVel = rightHipVel
    self._rightKneeRot = rightKneeRot
    self._rightKneeVel = rightKneeVel
    self._rightAnkleRot = rightAnkleRot
    self._rightAnkleVel = rightAnkleVel

    self._rightShoulderRot = rightShoulderRot
    self._rightShoulderVel = rightShoulderVel
    self._rightElbowRot = rightElbowRot
    self._rightElbowVel = rightElbowVel

    self._leftHipRot = leftHipRot
    self._leftHipVel = leftHipVel
    self._leftKneeRot = leftKneeRot
    self._leftKneeVel = leftKneeVel
    self._leftAnkleRot = leftAnkleRot
    self._leftAnkleVel = leftAnkleVel

    self._leftShoulderRot = leftShoulderRot
    self._leftShoulderVel = leftShoulderVel
    self._leftElbowRot = leftElbowRot
    self._leftElbowVel = leftElbowVel

  def ComputeLinVel(self, posStart, posEnd, deltaTime):
    vel = [(posEnd[0] - posStart[0]) / deltaTime, (posEnd[1] - posStart[1]) / deltaTime,
           (posEnd[2] - posStart[2]) / deltaTime]
    return vel

  def ComputeAngVel(self, ornStart, ornEnd, deltaTime, bullet_client):
    dorn = bullet_client.getDifferenceQuaternion(ornStart, ornEnd)
    axis, angle = bullet_client.getAxisAngleFromQuaternion(dorn)
    angVel = [(axis[0] * angle) / deltaTime, (axis[1] * angle) / deltaTime,
              (axis[2] * angle) / deltaTime]
    return angVel

  def ComputeAngVelRel(self, ornStart, ornEnd, deltaTime, bullet_client):
    ornStartConjugate = [-ornStart[0], -ornStart[1], -ornStart[2], ornStart[3]]
    pos_diff, q_diff = bullet_client.multiplyTransforms([0, 0, 0], ornStartConjugate, [0, 0, 0],
                                                        ornEnd)
    axis, angle = bullet_client.getAxisAngleFromQuaternion(q_diff)
    angVel = [(axis[0] * angle) / deltaTime, (axis[1] * angle) / deltaTime,
              (axis[2] * angle) / deltaTime]
    return angVel



  def NormalizeVector(self, vec):
    # length2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]
    length2 = self.DotProduct(vec, vec)
    if (length2 > 0):
      length = math.sqrt(length2)
      vec[0] /= length
      vec[1] /= length
      vec[2] /= length
      return vec

  def NormalizeQuaternion(self, orn):
    # length2 = orn[0] * orn[0] + orn[1] * orn[1] + orn[2] * orn[2] + orn[3] * orn[3]
    length2 = self.DotProduct(orn, orn)
    if (length2 > 0):
      length = math.sqrt(length2)
      orn[0] /= length
      orn[1] /= length
      orn[2] /= length
      orn[3] /= length
      return orn

    #print("Normalize? length=",length)

  def PostProcessMotionData(self, frameData):
    baseOrn1Start = [frameData[5], frameData[6], frameData[7], frameData[4]]

    chestRotStart = [frameData[9], frameData[10], frameData[11], frameData[8]]

    neckRotStart = [frameData[13], frameData[14], frameData[15], frameData[12]]
    rightHipRotStart = [frameData[17], frameData[18], frameData[19], frameData[16]]
    rightAnkleRotStart = [frameData[22], frameData[23], frameData[24], frameData[21]]
    rightShoulderRotStart = [frameData[26], frameData[27], frameData[28], frameData[25]]
    leftHipRotStart = [frameData[31], frameData[32], frameData[33], frameData[30]]
    leftAnkleRotStart = [frameData[36], frameData[37], frameData[38], frameData[35]]
    leftShoulderRotStart = [frameData[40], frameData[41], frameData[42], frameData[39]]

  def GetPose(self):
    pose = [
        # these 7 elements will be zero-ed out in pybullet_deep_mimic_env.set_action()
        self._basePos[0], self._basePos[1], self._basePos[2],
        self._baseOrn[0], self._baseOrn[1], self._baseOrn[2], self._baseOrn[3], 
        # these values will be given in ConvertFromAction()
        self._chestRot[0], self._chestRot[1], self._chestRot[2], self._chestRot[3],
        self._neckRot[0], self._neckRot[1], self._neckRot[2], self._neckRot[3],
        self._rightHipRot[0], self._rightHipRot[1], self._rightHipRot[2], self._rightHipRot[3],
        self._rightKneeRot[0],
        self._rightAnkleRot[0], self._rightAnkleRot[1], self._rightAnkleRot[2], self._rightAnkleRot[3],
        self._rightShoulderRot[0], self._rightShoulderRot[1], self._rightShoulderRot[2], self._rightShoulderRot[3],
        self._rightElbowRot[0],
        self._leftHipRot[0], self._leftHipRot[1], self._leftHipRot[2], self._leftHipRot[3],
        self._leftKneeRot[0],
        self._leftAnkleRot[0], self._leftAnkleRot[1], self._leftAnkleRot[2], self._leftAnkleRot[3],
        self._leftShoulderRot[0], self._leftShoulderRot[1], self._leftShoulderRot[2], self._leftShoulderRot[3],
        self._leftElbowRot[0]
    ]
    return pose

  def Slerp(self, frameFraction, frameData, frameDataNext, bullet_client):
    keyFrameDuration = frameData[0]
    basePos1Start = [frameData[1], frameData[2], frameData[3]]
    basePos1End = [frameDataNext[1], frameDataNext[2], frameDataNext[3]]
    self._basePos = [
        basePos1Start[0] + frameFraction * (basePos1End[0] - basePos1Start[0]),
        basePos1Start[1] + frameFraction * (basePos1End[1] - basePos1Start[1]),
        basePos1Start[2] + frameFraction * (basePos1End[2] - basePos1Start[2])
    ]
    self._baseLinVel = self.ComputeLinVel(basePos1Start, basePos1End, keyFrameDuration)
    baseOrn1Start = [frameData[5], frameData[6], frameData[7], frameData[4]]
    baseOrn1Next = [frameDataNext[5], frameDataNext[6], frameDataNext[7], frameDataNext[4]]
    self._baseOrn = bullet_client.getQuaternionSlerp(baseOrn1Start, baseOrn1Next, frameFraction)
    self._baseAngVel = self.ComputeAngVel(baseOrn1Start, baseOrn1Next, keyFrameDuration,
                                          bullet_client)

    ##pre-rotate to make z-up
    #y2zPos=[0,0,0.0]
    #y2zOrn = p.getQuaternionFromEuler([1.57,0,0])
    #basePos,baseOrn = p.multiplyTransforms(y2zPos, y2zOrn,basePos1,baseOrn1)

    chestRotStart = [frameData[9], frameData[10], frameData[11], frameData[8]]
    chestRotEnd = [frameDataNext[9], frameDataNext[10], frameDataNext[11], frameDataNext[8]]
    self._chestRot = bullet_client.getQuaternionSlerp(chestRotStart, chestRotEnd, frameFraction)
    self._chestVel = self.ComputeAngVelRel(chestRotStart, chestRotEnd, keyFrameDuration,
                                           bullet_client)

    neckRotStart = [frameData[13], frameData[14], frameData[15], frameData[12]]
    neckRotEnd = [frameDataNext[13], frameDataNext[14], frameDataNext[15], frameDataNext[12]]
    self._neckRot = bullet_client.getQuaternionSlerp(neckRotStart, neckRotEnd, frameFraction)
    self._neckVel = self.ComputeAngVelRel(neckRotStart, neckRotEnd, keyFrameDuration,
                                          bullet_client)

    rightHipRotStart = [frameData[17], frameData[18], frameData[19], frameData[16]]
    rightHipRotEnd = [frameDataNext[17], frameDataNext[18], frameDataNext[19], frameDataNext[16]]
    self._rightHipRot = bullet_client.getQuaternionSlerp(rightHipRotStart, rightHipRotEnd,
                                                         frameFraction)
    self._rightHipVel = self.ComputeAngVelRel(rightHipRotStart, rightHipRotEnd, keyFrameDuration,
                                              bullet_client)

    rightKneeRotStart = [frameData[20]]
    rightKneeRotEnd = [frameDataNext[20]]
    self._rightKneeRot = [
        rightKneeRotStart[0] + frameFraction * (rightKneeRotEnd[0] - rightKneeRotStart[0])
    ]
    self._rightKneeVel = [(rightKneeRotEnd[0] - rightKneeRotStart[0]) / keyFrameDuration]

    rightAnkleRotStart = [frameData[22], frameData[23], frameData[24], frameData[21]]
    rightAnkleRotEnd = [frameDataNext[22], frameDataNext[23], frameDataNext[24], frameDataNext[21]]
    self._rightAnkleRot = bullet_client.getQuaternionSlerp(rightAnkleRotStart, rightAnkleRotEnd,
                                                           frameFraction)
    self._rightAnkleVel = self.ComputeAngVelRel(rightAnkleRotStart, rightAnkleRotEnd,
                                                keyFrameDuration, bullet_client)

    rightShoulderRotStart = [frameData[26], frameData[27], frameData[28], frameData[25]]
    rightShoulderRotEnd = [
        frameDataNext[26], frameDataNext[27], frameDataNext[28], frameDataNext[25]
    ]
    self._rightShoulderRot = bullet_client.getQuaternionSlerp(rightShoulderRotStart,
                                                              rightShoulderRotEnd, frameFraction)
    self._rightShoulderVel = self.ComputeAngVelRel(rightShoulderRotStart, rightShoulderRotEnd,
                                                   keyFrameDuration, bullet_client)

    rightElbowRotStart = [frameData[29]]
    rightElbowRotEnd = [frameDataNext[29]]
    self._rightElbowRot = [
        rightElbowRotStart[0] + frameFraction * (rightElbowRotEnd[0] - rightElbowRotStart[0])
    ]
    self._rightElbowVel = [(rightElbowRotEnd[0] - rightElbowRotStart[0]) / keyFrameDuration]

    leftHipRotStart = [frameData[31], frameData[32], frameData[33], frameData[30]]
    leftHipRotEnd = [frameDataNext[31], frameDataNext[32], frameDataNext[33], frameDataNext[30]]
    self._leftHipRot = bullet_client.getQuaternionSlerp(leftHipRotStart, leftHipRotEnd,
                                                        frameFraction)
    self._leftHipVel = self.ComputeAngVelRel(leftHipRotStart, leftHipRotEnd, keyFrameDuration,
                                             bullet_client)

    leftKneeRotStart = [frameData[34]]
    leftKneeRotEnd = [frameDataNext[34]]
    self._leftKneeRot = [
        leftKneeRotStart[0] + frameFraction * (leftKneeRotEnd[0] - leftKneeRotStart[0])
    ]
    self._leftKneeVel = [(leftKneeRotEnd[0] - leftKneeRotStart[0]) / keyFrameDuration]

    leftAnkleRotStart = [frameData[36], frameData[37], frameData[38], frameData[35]]
    leftAnkleRotEnd = [frameDataNext[36], frameDataNext[37], frameDataNext[38], frameDataNext[35]]
    self._leftAnkleRot = bullet_client.getQuaternionSlerp(leftAnkleRotStart, leftAnkleRotEnd,
                                                          frameFraction)
    self._leftAnkleVel = self.ComputeAngVelRel(leftAnkleRotStart, leftAnkleRotEnd,
                                               keyFrameDuration, bullet_client)

    leftShoulderRotStart = [frameData[40], frameData[41], frameData[42], frameData[39]]
    leftShoulderRotEnd = [
        frameDataNext[40], frameDataNext[41], frameDataNext[42], frameDataNext[39]
    ]
    self._leftShoulderRot = bullet_client.getQuaternionSlerp(leftShoulderRotStart,
                                                             leftShoulderRotEnd, frameFraction)
    self._leftShoulderVel = self.ComputeAngVelRel(leftShoulderRotStart, leftShoulderRotEnd,
                                                  keyFrameDuration, bullet_client)

    leftElbowRotStart = [frameData[43]]
    leftElbowRotEnd = [frameDataNext[43]]
    self._leftElbowRot = [
        leftElbowRotStart[0] + frameFraction * (leftElbowRotEnd[0] - leftElbowRotStart[0])
    ]
    self._leftElbowVel = [(leftElbowRotEnd[0] - leftElbowRotStart[0]) / keyFrameDuration]

    pose = self.GetPose()
    return pose


  def ConvertFromAction(self, pybullet_client, action):
    #turn action into pose

    self.Reset()  #?? needed?
    index = 0

    if self.action_representation_mode == "6D":
      sixdim = action[index:index + 6]
      angle_axis = self.ConvertSixDimensionToAngleAxis(sixdim)
      angle = angle_axis[0] #float
      axis = angle_axis[1:4] #list
      self._chestRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)
      index += 6

      sixdim = action[index:index + 6]
      angle_axis = self.ConvertSixDimensionToAngleAxis(sixdim)
      angle = angle_axis[0]
      axis = angle_axis[1:4]
      self._neckRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)
      index += 6

      sixdim = action[index:index + 6]
      angle_axis = self.ConvertSixDimensionToAngleAxis(sixdim)
      angle = angle_axis[0]
      axis = angle_axis[1:4]
      self._rightHipRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)
      index += 6

      angle = action[index]
      self._rightKneeRot = [angle]
      index += 1

      sixdim = action[index:index + 6]
      angle_axis = self.ConvertSixDimensionToAngleAxis(sixdim)
      angle = angle_axis[0]
      axis = angle_axis[1:4]
      self._rightAnkleRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)
      index += 6

      sixdim = action[index:index + 6]
      angle_axis = self.ConvertSixDimensionToAngleAxis(sixdim)
      angle = angle_axis[0]
      axis = angle_axis[1:4]
      self._rightShoulderRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)
      index += 6

      angle = action[index]
      self._rightElbowRot = [angle]
      index += 1

      sixdim = action[index:index + 6]
      angle_axis = self.ConvertSixDimensionToAngleAxis(sixdim)
      angle = angle_axis[0]
      axis = angle_axis[1:4]
      self._leftHipRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)
      index += 6

      angle = action[index]
      self._leftKneeRot = [angle]
      index += 1

      sixdim = action[index:index + 6]
      angle_axis = self.ConvertSixDimensionToAngleAxis(sixdim)
      angle = angle_axis[0]
      axis = angle_axis[1:4]
      self._leftAnkleRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)
      index += 6

      sixdim = action[index:index + 6]
      angle_axis = self.ConvertSixDimensionToAngleAxis(sixdim)
      angle = angle_axis[0]
      axis = angle_axis[1:4]
      self._leftShoulderRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)
      index += 6

      angle = action[index]
      self._leftElbowRot = [angle]
      index += 1

    elif self.action_representation_mode == "ANGLE_AXIS":    
      angle = action[index]
      axis = [action[index + 1], action[index + 2], action[index + 3]]
      index += 4
      self._chestRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)
      #print("pose._chestRot=",pose._chestRot)

      angle = action[index]
      axis = [action[index + 1], action[index + 2], action[index + 3]]
      index += 4
      self._neckRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)

      angle = action[index]
      axis = [action[index + 1], action[index + 2], action[index + 3]]
      index += 4
      self._rightHipRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)

      angle = action[index]
      index += 1
      self._rightKneeRot = [angle]

      angle = action[index]
      axis = [action[index + 1], action[index + 2], action[index + 3]]
      index += 4
      self._rightAnkleRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)

      angle = action[index]
      axis = [action[index + 1], action[index + 2], action[index + 3]]
      index += 4
      self._rightShoulderRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)

      angle = action[index]
      index += 1
      self._rightElbowRot = [angle]

      angle = action[index]
      axis = [action[index + 1], action[index + 2], action[index + 3]]
      index += 4
      self._leftHipRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)

      angle = action[index]
      index += 1
      self._leftKneeRot = [angle]

      angle = action[index]
      axis = [action[index + 1], action[index + 2], action[index + 3]]
      index += 4
      self._leftAnkleRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)

      angle = action[index]
      axis = [action[index + 1], action[index + 2], action[index + 3]]
      index += 4
      self._leftShoulderRot = pybullet_client.getQuaternionFromAxisAngle(axis, angle)

      angle = action[index]
      index += 1
      self._leftElbowRot = [angle]
      
    pose = self.GetPose()

    return pose

  def DotProduct(self, vec1, vec2):
    assert len(vec1) == len(vec2), "PoseInterpolator DotProduct Dimension Error: Two vectors must have a same dimension"
    s = 0
    for i in range(len(vec1)):
      s += vec1[i] * vec2[i]
    return s

  def Normalize(self, vec):
    # can be vector, quaternion, whatever
    v = []
    length2 = self.DotProduct(vec, vec)
    assert length2 > 0, "PoseInterpolator Normalize Error: ZeroVector cannot be normalized"
    length = math.sqrt(length2)
    for i in range(len(vec)):
      v.append(vec[i] / length)
    return v

  def CrossProduct(self, v1, v2):
    assert len(v1) == len(v2), "PoseInterpolator CrossProduct Dimension Error: Two vectors must have a same dimension"
    vec_dim = len(v1)
    assert vec_dim > 2, "PoseInterpolator CrossProduct Dimension Error: The dimension must be larger than 2"
    v = []
    for i in range(vec_dim):
      v.append(v1[(i + 1) % vec_dim] * v2[(i + 2) % vec_dim] - v1[(i + 2) % vec_dim] * v2[(i + 1) % vec_dim])
    return v
  
  def ScalarVectorProduct(self, s, vec):
    v = []
    for i in range(len(vec)):
      v.append(s * vec[i])
    return v
  
  def VectorSummation(self, v1, v2):
    assert len(v1) == len(v2), "PoseInterpolator VectorSummation Dimension Error: Two vectors must have a same dimension"
    v = []
    for i in range(len(v1)):
      v.append(v1[i] + v2[i])
    return v

  def VectorSubtraction(self, vec1, vec2):
    assert len(vec1) == len(vec2), "PoseInterpolator VectorSubtraction Dimension Error: Two vectors must have a same dimension"
    v = []
    for i in range(len(vec1)):
      v.append(vec1[i] - vec2[i])
    return v

  def ConvertSixDimensionToMatrix(self, six_dimension):
    assert len(six_dimension) == 6, "PoseInterpolator ConvertSixDimensionToMatrix Dimension Error: Dimension not 6"
    r1 = six_dimension[:3]
    r1 = self.Normalize(r1)
    r2 = six_dimension[3:]
    r2_r1 = self.DotProduct(r1, r2)
    r2 = self.VectorSubtraction(r2, self.ScalarVectorProduct(r2_r1, r1))
    r2 = self.Normalize(r2)
    r3 = self.CrossProduct(r1, r2)
    return r1 + r2 + r3

  def Determinant(self, m):
    assert len(m) == 9, "PoseInterpolator Determinant Dimension Error: Dimension not 9"
    return self.DotProduct(m[:3], self.CrossProduct(m[3:6], m[6:9]))

  def ConvertMatrixToAngleAxis(self, m):
    assert len(m) == 9, "PoseInterpolator ConvertMatrixToAngleAxis Dimension Error: Dimension not 9"
    assert abs(self.Determinant(m) - 1.0) < eps, "PoseInterpolator ConvertMatrixToAngleAxis Determinant Error: Determinant not 1"
    angle = math.acos(min(max((m[0] + m[4] + m[8] - 1.0) / 2.0, -1.0), 1.0))
    
    if abs(angle) < eps:
      return [0.0, 0.0, 1.0, 0.0]
    # if abs(abs(angle) - 180.0) < eps:
    axis_denom = math.sqrt((m[7] - m[5]) * (m[7] - m[5]) + (m[2] - m[6]) * (m[2] - m[6]) + (m[3] - m[1]) * (m[3] - m[1]))
    if abs(axis_denom) < eps:
      x = math.sqrt((m[0] + 1.0) / 2.0)
      if x < eps:
        x = 0.0
        y = math.sqrt((m[4] + 1.0) / 2.0)
        if y < eps:
          y = 0.0
          z = 1.0
        else:
          z = (m[5] + m[7]) / y / 4.0
      else:
        y = (m[1] + m[3]) / x / 4.0
        z = (m[2] + m[6]) / x / 4.0
    else:
      x = (m[7] - m[5]) / axis_denom
      y = (m[2] - m[6]) / axis_denom
      z = (m[3] - m[1]) / axis_denom

    axis = self.Normalize([x, y, z])
    return np.array([angle] + axis, dtype=np.float32)

  def ConvertSixDimensionToAngleAxis(self, six_dimension):
    return self.ConvertMatrixToAngleAxis(self.ConvertSixDimensionToMatrix(six_dimension))


    

