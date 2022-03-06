import numpy as np
import math
from pybullet_envs.deep_mimic.env import action_space
from pybullet_envs.deep_mimic.env.env import Env
from pybullet_envs.deep_mimic.env.action_space import ActionSpace
from pybullet_utils import bullet_client
import time
from pybullet_envs.deep_mimic.env import motion_capture_data
from pybullet_envs.deep_mimic.env import humanoid_stable_pd
import pybullet_data
import pybullet as p1
import random

from enum import Enum

class InitializationStrategy(Enum):
  """Set how the environment is initialized."""
  START = 0
  RANDOM = 1  # random state initialization (RSI)


class PyBulletDeepMimicEnv(Env):

  def __init__(self, arg_parser=None, enable_draw=False, pybullet_client=None,
               time_step=1./240,
               init_strategy=InitializationStrategy.RANDOM):
    super().__init__(arg_parser, enable_draw)
    
    # REPRESENTATION_MODE_CHECKPOINT
    # self.state_representation_mode = "Quaternion"
    # self.action_representation_mode = "AxisAngle"
    # self.state_representation_mode = "6D"
    # self.action_representation_mode = "6D"
    # print("State representation mode: {:s}".format(self.state_representation_mode))
    # print("Action representation mode: {:s}".format(self.action_representation_mode))

    self._num_agents = 1
    self._pybullet_client = pybullet_client
    self._isInitialized = False
    self._useStablePD = True
    self._arg_parser = arg_parser
    self.timeStep = time_step
    self._init_strategy = init_strategy
    print("Initialization strategy: {:s}".format(init_strategy))


    # REPRESENTATION_MODE_CHECKPOINT
    self.state_representation_mode = self._arg_parser.parse_string('state_repr', default="Quaternion")
    self.action_representation_mode = self._arg_parser.parse_string('action_repr', default="AxisAngle")
    if self.action_representation_mode == "Quaternion":
      self.action_dim = 4
    elif self.action_representation_mode == "Euler":
      self.action_dim = 3
    elif self.action_representation_mode == "AxisAngle":
      self.action_dim = 4
    elif self.action_representation_mode == "RotVec":
      self.action_dim = 3
    elif self.action_representation_mode == "RotMat":
      self.action_dim = 9
    elif self.action_representation_mode == "6D":
      self.action_dim = 6

    self.reset()

  def reset(self):

    if not self._isInitialized:
      if self.enable_draw:
        self._pybullet_client = bullet_client.BulletClient(connection_mode=p1.GUI)
        #disable 'GUI' since it slows down a lot on Mac OSX and some other platforms
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI, 0)
      else:
        self._pybullet_client = bullet_client.BulletClient()

      self._pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
      z2y = self._pybullet_client.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
      self._planeId = self._pybullet_client.loadURDF("plane_implicit.urdf", [0, 0, 0],
                                                     z2y,
                                                     useMaximalCoordinates=True)
      #print("planeId=",self._planeId)
      self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_Y_AXIS_UP, 1)
      self._pybullet_client.setGravity(0, -9.8, 0)

      self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=10)
      self._pybullet_client.changeDynamics(self._planeId, linkIndex=-1, lateralFriction=0.9)

      self._mocapData = motion_capture_data.MotionCaptureData()

      motion_file = self._arg_parser.parse_strings('motion_file')
      print("motion_file=", motion_file[0])

      motionPath = pybullet_data.getDataPath() + "/" + motion_file[0]
      #motionPath = pybullet_data.getDataPath()+"/motions/humanoid3d_backflip.txt"
      self._mocapData.Load(motionPath)
      timeStep = self.timeStep
      useFixedBase = False
      self._humanoid = humanoid_stable_pd.HumanoidStablePD(self._pybullet_client, self._mocapData,
                                                           timeStep, useFixedBase, self._arg_parser)
      self._isInitialized = True

      self._pybullet_client.setTimeStep(timeStep)
      self._pybullet_client.setPhysicsEngineParameter(numSubSteps=1)

      selfCheck = False
      if (selfCheck):
        curTime = 0
        while self._pybullet_client.isConnected():
          self._humanoid.setSimTime(curTime)
          state = self._humanoid.getState()
          #print("state=",state)
          pose = self._humanoid.computePose(self._humanoid._frameFraction)
          for i in range(10):
            curTime += timeStep
            #taus = self._humanoid.computePDForces(pose)
            #self._humanoid.applyPDForces(taus)
            #self._pybullet_client.stepSimulation()
          time.sleep(timeStep)
    #print("numframes = ", self._humanoid._mocap_data.NumFrames())
    #startTime = random.randint(0,self._humanoid._mocap_data.NumFrames()-2)
    
    if self._init_strategy == InitializationStrategy.RANDOM:
      rnrange = 1000
      rn = random.randint(0, rnrange)
      startTime = float(rn) / rnrange * self._humanoid.getCycleTime()
    elif self._init_strategy == InitializationStrategy.START:
      startTime = 0
    
    self.t = startTime

    self._humanoid.setSimTime(startTime)

    self._humanoid.resetPose()
    #this clears the contact points. Todo: add API to explicitly clear all contact points?
    #self._pybullet_client.stepSimulation()
    self._humanoid.resetPose()
    self.needs_update_time = self.t - 1  #force update

  def get_num_agents(self):
    return self._num_agents

  def get_action_space(self, agent_id):
    return ActionSpace(ActionSpace.Continuous)

  def get_reward_min(self, agent_id):
    return 0

  def get_reward_max(self, agent_id):
    return 1

  def get_reward_fail(self, agent_id):
    return self.get_reward_min(agent_id)

  def get_reward_succ(self, agent_id):
    return self.get_reward_max(agent_id)

  #scene_name == "imitate" -> cDrawSceneImitate
  def get_state_size(self, agent_id):
    #cCtController::GetStateSize()
    #int state_size = cDeepMimicCharController::GetStateSize();
    #                     state_size += GetStatePoseSize();#106
    #                     state_size += GetStateVelSize(); #(3+3)*numBodyParts=90
    #state_size += GetStatePhaseSize();#1
    #197

    # REPRESENTATION_MODE_CHECKPOINT
    if self.state_representation_mode == "Quaternion":
      return 197
    elif self.state_representation_mode == "Euler":
      return 182
    elif self.state_representation_mode == "AxisAngle":
      return 197
    elif self.state_representation_mode == "RotVec":
      return 182
    elif self.state_representation_mode == "RotMat":
      return 272
    elif self.state_representation_mode == "6D":
      return 227 # 197 + numBodyPart * 2
    # return 197

  def build_state_norm_groups(self, agent_id):
    #if (mEnablePhaseInput)
    #{
    #int phase_group = gNormGroupNone;
    #int phase_offset = GetStatePhaseOffset();
    #int phase_size = GetStatePhaseSize();
    #out_groups.segment(phase_offset, phase_size) = phase_group * Eigen::VectorXi::Ones(phase_size);
    groups = [0] * self.get_state_size(agent_id)
    groups[0] = -1
    return groups

  def build_state_offset(self, agent_id):
    out_offset = [0] * self.get_state_size(agent_id)
    phase_offset = -0.5
    out_offset[0] = phase_offset
    return np.array(out_offset)

  def build_state_scale(self, agent_id):
    out_scale = [1] * self.get_state_size(agent_id)
    phase_scale = 2
    out_scale[0] = phase_scale
    return np.array(out_scale)

  def get_goal_size(self, agent_id):
    return 0

  def get_action_size(self, agent_id):
    if self.action_representation_mode == "Quaternion":
      ctrl_size = 43
    elif self.action_representation_mode == "Euler":
      ctrl_size = 35
    elif self.action_representation_mode == "AxisAngle":
      ctrl_size = 43
    elif self.action_representation_mode == "RotVec":
      ctrl_size = 35
    elif self.action_representation_mode == "RotMat":
      ctrl_size = 83
    elif self.action_representation_mode == "6D":
      ctrl_size = 59
    # if self.action_representation_mode == "6D":
    #   ctrl_size = 59 #numDof
    # elif self.action_representation_mode == "AxisAngle":
    #   ctrl_size = 43 #numDof
    root_size = 7
    return ctrl_size - root_size

  def build_goal_norm_groups(self, agent_id):
    return np.array([])

  def build_goal_offset(self, agent_id):
    return np.array([])

  def build_goal_scale(self, agent_id):
    return np.array([])

  def build_action_offset(self, agent_id):
    out_offset = [0] * self.get_action_size(agent_id)
    
    # REPRESENTATION_MODE_CHECKPOINT
    out_offset[self.action_dim * 3] = 1.57
    out_offset[self.action_dim * 5 + 1] = - 1.57
    out_offset[self.action_dim * 6 + 2] = 1.57
    out_offset[self.action_dim * 8 + 3] = - 1.57

    # REPRESENTATION_MODE_CHECKPOINT
    # intrigues the default axis to be -z axis? to bend forward?
    # this is strange. need some investigation.
    # if self.action_representation_mode == "AxisAngle":
    #   out_offset = [
    #       0.0000000000, 0.0000000000, 0.0000000000, -0.200000000, # chest 0 ~ 3
    #       0.0000000000, 0.0000000000, 0.0000000000, -0.200000000, # neck 4 ~ 7

    #       0.0000000000, 0.0000000000, 0.0000000000, -0.200000000, # rightHip 8 ~ 11
    #       1.57000000, # rightKnee 12
    #       0.0000000000, 0.0000000000, 0.0000000000, -0.200000000, # rightAnkle 13 ~ 16
    #       0.0000000000, 0.0000000000, 0.0000000000, -0.200000000, # rightShoulder 17 ~ 20
    #       -1.5700000, # rightElbow 21

    #       0.0000000000, 0.0000000000, 0.0000000000, -0.200000000, # leftHip 22 ~ 25
    #       1.5700000000, # leftKnee 26
    #       0.0000000000, 0.0000000000, 0.0000000000, -0.200000000, # leftAnkle 27 ~ 30
    #       0.0000000000, 0.0000000000, 0.0000000000, -0.200000000, # leftShoulder 31 ~ 34
    #       -1.570000000 # leftElbow 35
    #   ]
    # elif self.action_representation_mode == "6D":
    #   out_offset = [
    #       0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, # chest
    #       0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, # neck
          
    #       0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, # rightHip
    #       1.57000000, # rightKnee
    #       0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, # rightAnkle
    #       0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, # rightShoulder
    #       -1.5700000, # rightElbow
          
    #       0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, #leftHip
    #       1.5700000000, # leftKnee
    #       0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, # leftAnkle
    #       0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, # leftShoulder
    #       -1.570000000 # leftAnkle
    #   ]
    #see cCtCtrlUtil::BuildOffsetScalePDPrismatic and
    #see cCtCtrlUtil::BuildOffsetScalePDSpherical
    return np.array(out_offset)

  def build_action_scale(self, agent_id):
    out_scale = [1] * self.get_action_size(agent_id)

    # REPRESENTATION_MODE_CHECKPOINT
    out_scale[self.action_dim * 3] = 0.15923566878900
    out_scale[self.action_dim * 5 + 1] = 0.15923566878900
    out_scale[self.action_dim * 6 + 2] = 0.15923566878900
    out_scale[self.action_dim * 8 + 3] = 0.15923566878900

    
    #see cCtCtrlUtil::BuildOffsetScalePDPrismatic and
    #see cCtCtrlUtil::BuildOffsetScalePDSpherical


    # REPRESENTATION_MODE_CHECKPOINT
    # if self.action_representation_mode == "AxisAngle":
    #   out_scale = [
    #       0.20833333333333, 1.00000000000000, 1.00000000000000, 1.00000000000000, # chest
    #       0.25000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, # neck

    #       0.12077294685990, 1.00000000000000, 1.00000000000000, 1.00000000000000, # rightHip
    #       0.15923566878900, # rightKnee
    #       0.15923566878900, 1.00000000000000, 1.00000000000000, 1.00000000000000, # rightAnkle
    #       0.07961783439400, 1.00000000000000, 1.00000000000000, 1.00000000000000, # rightShoulder
    #       0.15923566878900, # rightElbow

    #       0.12077294685900, 1.00000000000000, 1.00000000000000, 1.00000000000000, # leftHip
    #       0.15923566878900, # leftKnee
    #       0.15923566878900, 1.00000000000000, 1.00000000000000, 1.00000000000000, # leftAnkle
    #       0.10775862068900, 1.00000000000000, 1.00000000000000, 1.00000000000000, # leftShoulder
    #       0.15923566878900 # leftElbow
    #   ]
    # elif self.action_representation_mode == "6D":
    #   out_scale = [
    #       1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, # chest
    #       1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, # neck

    #       1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, # rightHip
    #       0.15923566878900, # rightKnee
    #       1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, # rightAnkle
    #       1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, # rightShoulder
    #       0.15923566878900, # rightElbow
          
    #       1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, # leftHip
    #       0.15923566878900, # leftKnee
    #       1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, # leftAnkle
    #       1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, 1.00000000000000, # leftShoulder
    #       0.15923566878900 # leftElbow
    #   ]
    return np.array(out_scale)

  def build_action_bound_min(self, agent_id):
    #see cCtCtrlUtil::BuildBoundsPDSpherical
    out_min = [-1] * self.get_action_size(agent_id)

    # REPRESENTATION_MODE_CHECKPOINT
    out_min[self.action_dim * 3] = - 7.85
    out_min[self.action_dim * 5 + 1] = - 4.71
    out_min[self.action_dim * 6 + 2] = - 7.85
    out_min[self.action_dim * 8 + 3] = - 4.71

    # REPRESENTATION_MODE_CHECKPOINT
    # if self.action_representation_mode == "AxisAngle":
    #   out_min = [
    #       -4.79999999999, -1.00000000000, -1.00000000000, -1.00000000000, # chest
    #       -4.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, # neck

    #       -7.77999999999, -1.00000000000, -1.00000000000, -1.00000000000, # rightHip
    #       -7.85000000000, # rightKnee
    #       -6.28000000000, -1.00000000000, -1.00000000000, -1.00000000000, # rightAnkle
    #       -12.5600000000, -1.00000000000, -1.00000000000, -1.00000000000, # rightShoulder
    #       -4.71000000000, # rightElbow

    #       -7.77999999900, -1.00000000000, -1.00000000000, -1.00000000000, # leftHip
    #       -7.85000000000, # leftKnee
    #       -6.28000000000, -1.00000000000, -1.00000000000, -1.00000000000, # leftAnkle
    #       -8.46000000000, -1.00000000000, -1.00000000000, -1.00000000000, # leftShoulder. why is it not symmetric to rightShoulder?
    #       -4.71000000000 # leftElbow
    #   ]

    # elif self.action_representation_mode == "6D":
    #   out_min = [
    #       -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, # chest
    #       -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, # neck

    #       -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, # rightHip
    #       -7.85000000000, # rightKnee
    #       -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, # rightAnkle
    #       -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, # rightShoulder
    #       -4.71000000000, # rightElbow

    #       -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, # leftHip
    #       -7.85000000000, # leftKnee
    #       -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, # leftAnkle
    #       -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, -1.00000000000, # leftShoulder.
    #       -4.71000000000 # leftElbow
    #   ]

    return out_min

  def build_action_bound_max(self, agent_id):
    out_max = [1] * self.get_action_size(agent_id)

    # REPRESENTATION_MODE_CHECKPOINT
    out_max[self.action_dim * 3] = 4.71
    out_max[self.action_dim * 5 + 1] = 7.85
    out_max[self.action_dim * 6 + 2] = 4.71
    out_max[self.action_dim * 8 + 3] = 7.85

    # REPRESENTATION_MODE_CHECKPOINT
    # if self.action_representation_mode == "AxisAngle":
    #   out_max = [
    #       4.799999999, 1.000000000, 1.000000000, 1.000000000, # chest
    #       4.000000000, 1.000000000, 1.000000000, 1.000000000, # neck

    #       8.779999999, 1.000000000, 1.000000000, 1.000000000, # rightHip
    #       4.710000000, # rightKnee
    #       6.280000000, 1.000000000, 1.000000000, 1.000000000, # rightAnkle
    #       12.56000000, 1.000000000, 1.000000000, 1.000000000, # rightShoulder
    #       7.850000000, # rightElbow

    #       8.779999999, 1.000000000, 1.000000000, 1.000000000, # leftHip
    #       4.710000000, # leftKnee
    #       6.280000000, 1.000000000, 1.000000000, 1.000000000, # leftAnkle
    #       10.10000000, 1.000000000, 1.000000000, 1.000000000, # leftShoulder. why is it not symmetric to rightShoulder?
    #       7.850000000 # leftElbow
    #   ]
    # elif self.action_representation_mode == "6D":
    #   out_max = [
    #       1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000, # chest
    #       1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000, # neck

    #       1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000, # rightHip
    #       4.710000000, # rightKnee
    #       1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000, # rightAnkle
    #       1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000, # rightShoulder
    #       7.850000000, # rightElbow

    #       1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000, # leftHip
    #       4.710000000, # leftKnee
    #       1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000, # leftAnkle
    #       1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000, 1.000000000, # leftShoulder.
    #       7.850000000 # leftElbow
    #   ]


    return out_max

  def set_mode(self, mode):
    self._mode = mode

  def need_new_action(self, agent_id):
    if self.t >= self.needs_update_time:
      self.needs_update_time = self.t + 1. / 30.
      return True
    return False

  def record_state(self, agent_id):
    state = self._humanoid.getState()

    return np.array(state)

  def record_goal(self, agent_id):
    return np.array([])

  def calc_reward(self, agent_id):
    kinPose = self._humanoid.computePose(self._humanoid._frameFraction)
    reward = self._humanoid.getReward(kinPose)
    return reward

  def set_action(self, agent_id, action):

    #print("action=",)
    #for a in action:
    #  print(a)
    #np.savetxt("pb_action.csv", action, delimiter=",")
    self.desiredPose = self._humanoid.convertActionToPose(action)
    #we need the target root positon and orientation to be zero, to be compatible with deep mimic
    self.desiredPose[0] = 0
    self.desiredPose[1] = 0
    self.desiredPose[2] = 0
    self.desiredPose[3] = 0
    self.desiredPose[4] = 0
    self.desiredPose[5] = 0
    self.desiredPose[6] = 0
    target_pose = np.array(self.desiredPose)

    #np.savetxt("pb_target_pose.csv", target_pose, delimiter=",")

    #print("set_action: desiredPose=", self.desiredPose)

  def log_val(self, agent_id, val):
    pass

  def update(self, timeStep):
    #print("pybullet_deep_mimic_env:update timeStep=",timeStep," t=",self.t)
    self._pybullet_client.setTimeStep(timeStep)
    self._humanoid._timeStep = timeStep
    self.timeStep = timeStep

    for i in range(1):
      self.t += timeStep
      self._humanoid.setSimTime(self.t)

      if self.desiredPose:
        kinPose = self._humanoid.computePose(self._humanoid._frameFraction)
        self._humanoid.initializePose(self._humanoid._poseInterpolator,
                                      self._humanoid._kin_model,
                                      initBase=True)
        #pos,orn=self._pybullet_client.getBasePositionAndOrientation(self._humanoid._sim_model)
        #self._pybullet_client.resetBasePositionAndOrientation(self._humanoid._kin_model, [pos[0]+3,pos[1],pos[2]],orn)
        #print("desiredPositions=",self.desiredPose)
        #len(maxForces) = 43
        maxForces = [
            # every 4th elements are dummies. never used.
            0, 0, 0, # basePos
            0, 0, 0, 0, # baseOrn
            200, 200, 200, 200, # chest
            50, 50, 50, 50, # neck

            200, 200, 200, 200, # rightHip
            150, # rightKnee
            90, 90, 90, 90, # rightAnkle
            100, 100, 100, 100, # rightShoulder
            60, # rightWrist

            200, 200, 200, 200, # leftHip
            150, # leftKnee
            90, 90, 90, 90, # leftAnkle
            100, 100, 100, 100, # leftShoulder
            60 # leftWrist
        ]

        if self._useStablePD:
          usePythonStablePD = False
          if usePythonStablePD:
            taus = self._humanoid.computePDForces(self.desiredPose,
                                                desiredVelocities=None,
                                                maxForces=maxForces)
            #taus = [0]*43
            self._humanoid.applyPDForces(taus)
          else:
            self._humanoid.computeAndApplyPDForces(self.desiredPose,
                                                maxForces=maxForces)
        else:
          self._humanoid.setJointMotors(self.desiredPose, maxForces=maxForces)

        self._pybullet_client.stepSimulation()

  def set_sample_count(self, count):
    return

  def check_terminate(self, agent_id):
    return Env.Terminate(self.is_episode_end())

  def is_episode_end(self):
    isEnded = self._humanoid.terminates()
    #also check maximum time, 20 seconds (todo get from file)
    #print("self.t=",self.t)
    if (self.t > 20):
      isEnded = True
    return isEnded

  def check_valid_episode(self):
    #could check if limbs exceed velocity threshold
    return True

  def getKeyboardEvents(self):
    return self._pybullet_client.getKeyboardEvents()

  def isKeyTriggered(self, keys, key):
    o = ord(key)
    #print("ord=",o)
    if o in keys:
      return keys[ord(key)] & self._pybullet_client.KEY_WAS_TRIGGERED
    return False
