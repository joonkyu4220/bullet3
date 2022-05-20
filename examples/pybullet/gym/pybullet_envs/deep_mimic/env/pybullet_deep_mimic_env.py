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
    # REPRESENTATION_MODE_CHECKPOINT
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
    # out_offset = [0] * self.get_action_size(agent_id)
    
    # REPRESENTATION_MODE_CHECKPOINT
    # out_offset[self.action_dim * 3] = 1.57
    # out_offset[self.action_dim * 5 + 1] = - 1.57
    # out_offset[self.action_dim * 6 + 2] = 1.57
    # out_offset[self.action_dim * 8 + 3] = - 1.57

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

    # Back to the base setting, but with angle-axis to axis-angle
    # out_offset = [
    #     0.0000000000, 0.0000000000, -0.200000000, 0.0000000000, # chest 0 ~ 3
    #     0.0000000000, 0.0000000000, -0.200000000, 0.0000000000, # neck 4 ~ 7

    #     0.0000000000, 0.0000000000, -0.200000000, 0.0000000000, # rightHip 8 ~ 11
    #     1.57000000, # rightKnee 12
    #     0.0000000000, 0.0000000000, -0.200000000, 0.0000000000, # rightAnkle 13 ~ 16
    #     0.0000000000, 0.0000000000, -0.200000000, 0.0000000000, # rightShoulder 17 ~ 20
    #     -1.5700000, # rightElbow 21

    #     0.0000000000, 0.0000000000, -0.200000000, 0.0000000000, # leftHip 22 ~ 25
    #     1.5700000000, # leftKnee 26
    #     0.0000000000, 0.0000000000, -0.200000000, 0.0000000000, # leftAnkle 27 ~ 30
    #     0.0000000000, 0.0000000000, -0.200000000, 0.0000000000, # leftShoulder 31 ~ 34
    #     -1.570000000 # leftElbow 35
    # ]
    if self.action_representation_mode == "AxisAngle":
      out_offset = [0.0068755508, 0.0015763351, -0.1889394081, 0.1941892161, -0.0218390692, -0.0044824981, -0.2713390361, 0.0519516956, -0.0416964835, -0.0324836366, -0.2194207496, -0.0418716786, 0.2136441523, -0.0456953894, -0.0541528785, -0.1701527079, -0.3117513497, -0.0082508534, 0.0148627206, -0.2549700266, -0.0574835542, -0.5343895408, 0.0707179259, 0.0438515172, -0.2625762215, -0.1406294679, 0.1265910778, -0.0292067985, -0.0199558265, -0.1859595923, -0.2207922093, 0.0615333368, -0.0177153641, -0.1982041296, 0.0545032612, -0.4616308242]
    elif self.action_representation_mode == "Euler":
      out_offset = [-0.0055337809, 0.1760647961, -0.0081495298, 0.0030756048, 0.0497254311, 0.0048212991, 0.0184817684, -0.0427404359, 0.0082376780, 0.2136441523, -0.0801435483, -0.2600736033, -0.0513400693, 0.0060741188, -0.0498765028, -0.0076539819, -0.5343895408, -0.0113230695, -0.1343158948, 0.0099628547, 0.1265910778, 0.0026065106, -0.1984973286, -0.0357808968, -0.0375163823, 0.0461519098, -0.0166258762, -0.4616308242]
    elif self.action_representation_mode == "Quaternion":
      out_offset = [-0.0033733559, -0.0018363997, 0.0877804348, -0.9861415044, 0.0021053839, 0.0005597263, 0.0247960125, -0.9933219240, -0.0028963908, -0.0002116503, -0.0187910167, -0.9724083056, 0.2136441523, -0.0354918631, -0.0456177742, -0.1266701485, -0.9718618169, 0.0045213123, -0.0001421004, -0.0260037182, -0.9399837366, -0.5343895408, 0.0118919095, 0.0076987608, -0.0640084692, -0.9687923660, 0.1265910778, -0.0191103540, -0.0062755806, -0.0977620585, -0.9729132948, -0.0105958243, 0.0021083723, 0.0223357296, -0.9622992633, -0.4616308242]
    elif self.action_representation_mode == "RotMat":
      out_offset = [-0.9502896019, -0.1700744506, -0.0021317802, 0.1698302913, -0.9498163532, 0.0075045122, 0.0048660044, -0.0056212268, -0.9911931867, -0.9747009595, -0.0486631117, -0.0007045182, 0.0484574385, -0.9746148274, -0.0045272700, -0.0029192432, 0.0037454052, -0.9978380180, -0.9043115847, 0.0307061242, -0.0155150990, -0.0382870666, -0.9030529899, -0.0068091553, -0.0149052608, -0.0173035969, -0.9808094856, 0.2136441523, -0.9055277104, 0.2316480323, -0.1046465694, -0.2452063997, -0.9091631681, 0.0429026153, 0.0664734227, -0.0903971341, -0.9684995613, -0.7920828414, 0.0425153112, -0.0051461970, -0.0417037678, -0.7925169286, 0.0035726945, -0.0049175730, 0.0185259232, -0.9785829341, -0.5343895408, -0.8924999714, 0.1126582792, 0.0377994267, -0.1207527089, -0.8886141158, -0.0071508844, 0.0097042601, 0.0363953227, -0.9812800666, 0.1265910778, -0.9062825105, 0.1823434565, -0.0251299022, -0.1856967834, -0.9053936530, 0.0286957005, -0.0013234660, -0.0431265751, -0.9798168112, -0.8774339054, -0.0365282649, 0.0343789143, 0.0409069360, -0.8690125875, 0.0102394099, 0.0271548633, -0.0268755319, -0.9704271962, -0.4616308242]
    elif self.action_representation_mode == "RotVec":
      out_offset = [-0.0068089260, -0.0037324410, 0.1774723749, 0.0042358971, 0.0011234344, 0.0499412366, -0.0059838202, -0.0004640953, -0.0386632018, 0.2136441523, -0.0724906081, -0.0932101711, -0.2585146443, 0.0096432123, -0.0003586089, -0.0557854199, -0.5343895408, 0.0244967054, 0.0158767929, -0.1320256433, 0.1265910778, -0.0390211743, -0.0127726694, -0.1995010188, -0.0222684623, 0.0044153197, 0.0470070369, -0.4616308242]
    elif self.action_representation_mode == "6D":
      out_offset = [-0.9502896019, -0.1700744506, -0.0021317802, 0.1698302913, -0.9498163532, 0.0075045122, -0.9747009595, -0.0486631117, -0.0007045182, 0.0484574385, -0.9746148274, -0.0045272700, -0.9043115847, 0.0307061242, -0.0155150990, -0.0382870666, -0.9030529899, -0.0068091553, 0.2136441523, -0.9055277104, 0.2316480323, -0.1046465694, -0.2452063997, -0.9091631681, 0.0429026153, -0.7920828414, 0.0425153112, -0.0051461970, -0.0417037678, -0.7925169286, 0.0035726945, -0.5343895408, -0.8924999714, 0.1126582792, 0.0377994267, -0.1207527089, -0.8886141158, -0.0071508844, 0.1265910778, -0.9062825105, 0.1823434565, -0.0251299022, -0.1856967834, -0.9053936530, 0.0286957005, -0.8774339054, -0.0365282649, 0.0343789143, 0.0409069360, -0.8690125875, 0.0102394099, -0.4616308242]
    return np.array(out_offset)

  def build_action_scale(self, agent_id):
    # out_scale = [1] * self.get_action_size(agent_id)

    # REPRESENTATION_MODE_CHECKPOINT
    # out_scale[self.action_dim * 3] = 0.15923566878900
    # out_scale[self.action_dim * 5 + 1] = 0.15923566878900
    # out_scale[self.action_dim * 6 + 2] = 0.15923566878900
    # out_scale[self.action_dim * 8 + 3] = 0.15923566878900

    
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

    # Back to the base setting
    # out_scale = [
    #     1.00000000000000, 1.00000000000000, 1.00000000000000, 0.20833333333333, # chest
    #     1.00000000000000, 1.00000000000000, 1.00000000000000, 0.25000000000000, # neck

    #     1.00000000000000, 1.00000000000000, 1.00000000000000, 0.12077294685990, # rightHip
    #     0.15923566878900, # rightKnee
    #     1.00000000000000, 1.00000000000000, 1.00000000000000, 0.15923566878900, # rightAnkle
    #     1.00000000000000, 1.00000000000000, 1.00000000000000, 0.07961783439400, # rightShoulder
    #     0.15923566878900, # rightElbow

    #     1.00000000000000, 1.00000000000000, 1.00000000000000, 0.12077294685900, # leftHip
    #     0.15923566878900, # leftKnee
    #     1.00000000000000, 1.00000000000000, 1.00000000000000, 0.15923566878900, # leftAnkle
    #     1.00000000000000, 1.00000000000000, 1.00000000000000, 0.10775862068900, # leftShoulder
    #     0.15923566878900 # leftElbow
    # ]


    if self.action_representation_mode == "AxisAngle":
      out_scale = [1.2066897770, 1.2730279571, 1.2690108917, 0.2575108312, 1.2919408569, 1.2656822975, 1.1742534776, 0.3102073632, 1.1028469521, 1.1427577389, 1.2100446315, 0.1485470347, 0.1286925363, 1.2733765833, 1.2000895246, 1.2699688917, 0.1935719694, 1.2116641807, 1.2233423440, 1.2689393798, 0.0997391669, 0.1440270975, 1.1550555136, 1.1812798960, 1.2233601150, 0.1446811361, 0.1443466390, 1.2411161505, 1.2259570002, 1.2441556936, 0.1693927882, 1.2257717697, 1.2025925965, 1.2546243091, 0.1266573302, 0.1554505037]
    elif self.action_representation_mode == "Euler":
      out_scale = [0.6727240470, 0.2836008328, 0.6432791125, 1.4079655996, 0.3246415371, 1.3625617393, 0.3921766321, 0.1680723512, 0.3727975343, 0.1286925363, 0.3695166108, 0.2287784827, 0.4016717585, 0.1544798844, 0.1135608225, 0.1555751846, 0.1440270975, 0.3644388128, 0.1619951121, 0.3234660168, 0.1443466390, 0.3993974265, 0.1929527519, 0.3940300220, 0.2779370871, 0.1493121867, 0.2346379545, 0.1554505037]
    elif self.action_representation_mode == "Quaternion":
      out_scale = [1.4568683530, 1.5348446024, 0.5702867058, 3.7431614434, 2.9644099267, 3.0736979318, 0.6523078063, 6.9997723338, 0.9798226247, 1.0454826113, 0.3375380857, 1.8284790231, 0.1286925363, 0.9266385303, 0.8549951768, 0.4676357372, 1.9548658862, 0.9683722248, 0.9470908881, 0.2238128945, 0.8234581276, 0.1440270975, 0.9430203590, 1.1587675371, 0.3265740688, 1.5498678291, 0.1443466390, 0.9997138005, 1.0120299850, 0.3884231692, 1.9318417291, 0.7225231637, 0.9630308863, 0.2971433809, 1.2377173076, 0.1554505037]
    elif self.action_representation_mode == "RotMat":
      out_scale = [1.0526173784, 0.2970845762, 0.7787794439, 0.2973471985, 1.0458811520, 0.7358035997, 0.7710925196, 0.7422405731, 3.7657600709, 1.8594510771, 0.3331753022, 1.5534799030, 0.3333663378, 1.8568282083, 1.4774184765, 1.5300990933, 1.4989687475, 14.5885087805, 0.5441022597, 0.1824138333, 0.5333264743, 0.1824956992, 0.5393881946, 0.5076386094, 0.5391860690, 0.5058078412, 1.8170591583, 0.1286925363, 0.5916047399, 0.2565807130, 0.4261825530, 0.2482425166, 0.6163495007, 0.5072608741, 0.4723642461, 0.4494082988, 1.2381218547, 0.2591358559, 0.1330427211, 0.4833815033, 0.1326633843, 0.2603641805, 0.5017110586, 0.5018456265, 0.4870852337, 1.6064151106, 0.1440270975, 0.4645849800, 0.1797421787, 0.5724586946, 0.1792035642, 0.4538235803, 0.5030990795, 0.6028101142, 0.4843368843, 1.9429948793, 0.1443466390, 0.5737358643, 0.2097005739, 0.5152504588, 0.2092417813, 0.5718334963, 0.5209861625, 0.5312285149, 0.5113489104, 1.5674536791, 0.4112820258, 0.1661232421, 0.4822465723, 0.1658382186, 0.3906880836, 0.3866883537, 0.4881364747, 0.3847415912, 1.2482456250, 0.1554505037]
    elif self.action_representation_mode == "RotVec":
      out_scale = [0.7187231093, 0.7570354938, 0.2811034352, 1.4719907938, 1.5259426252, 0.3237818604, 0.4761269766, 0.5077147140, 0.1642371008, 0.1286925363, 0.4509565819, 0.4154344409, 0.2274690692, 0.4549374873, 0.4435391756, 0.1050427609, 0.1440270975, 0.4567181091, 0.5599240122, 0.1579295266, 0.1443466390, 0.4864009273, 0.4912708860, 0.1891086409, 0.3444831274, 0.4621097705, 0.1424557754, 0.1554505037]
    elif self.action_representation_mode == "6D":
      out_scale = [1.0526173784, 0.2970845762, 0.7787794439, 0.2973471985, 1.0458811520, 0.7358035997, 1.8594510771, 0.3331753022, 1.5534799030, 0.3333663378, 1.8568282083, 1.4774184765, 0.5441022597, 0.1824138333, 0.5333264743, 0.1824956992, 0.5393881946, 0.5076386094, 0.1286925363, 0.5916047399, 0.2565807130, 0.4261825530, 0.2482425166, 0.6163495007, 0.5072608741, 0.2591358559, 0.1330427211, 0.4833815033, 0.1326633843, 0.2603641805, 0.5017110586, 0.1440270975, 0.4645849800, 0.1797421787, 0.5724586946, 0.1792035642, 0.4538235803, 0.5030990795, 0.1443466390, 0.5737358643, 0.2097005739, 0.5152504588, 0.2092417813, 0.5718334963, 0.5209861625, 0.4112820258, 0.1661232421, 0.4822465723, 0.1658382186, 0.3906880836, 0.3866883537, 0.1554505037]
    return np.array(out_scale)

  def build_action_bound_min(self, agent_id):
    #see cCtCtrlUtil::BuildBoundsPDSpherical
    # out_min = [-1] * self.get_action_size(agent_id)

    # REPRESENTATION_MODE_CHECKPOINT
    # out_min[self.action_dim * 3] = - 7.85
    # out_min[self.action_dim * 5 + 1] = - 4.71
    # out_min[self.action_dim * 6 + 2] = - 7.85
    # out_min[self.action_dim * 8 + 3] = - 4.71

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

    # Back to the base setting
    # out_min = [
    #     -1.00000000000, -1.00000000000, -1.00000000000, -4.79999999999, # chest
    #     -1.00000000000, -1.00000000000, -1.00000000000, -4.00000000000, # neck

    #     -1.00000000000, -1.00000000000, -1.00000000000, -7.77999999999, # rightHip
    #     -7.85000000000, # rightKnee
    #     -1.00000000000, -1.00000000000, -1.00000000000, -6.28000000000, # rightAnkle
    #     -1.00000000000, -1.00000000000, -1.00000000000, -12.5600000000, # rightShoulder
    #     -4.71000000000, # rightElbow

    #     -1.00000000000, -1.00000000000, -1.00000000000, -7.77999999900, # leftHip
    #     -7.85000000000, # leftKnee
    #     -1.00000000000, -1.00000000000, -1.00000000000, -6.28000000000, # leftAnkle
    #     -1.00000000000, -1.00000000000, -1.00000000000, -8.46000000000, # leftShoulder. why is it not symmetric to rightShoulder?
    #     -4.71000000000 # leftElbow
    # ]
    if self.action_representation_mode == "AxisAngle":
      out_min = [-0.8355889609, -0.7871050381, -0.5990758930, -4.0775210181, -0.7521901711, -0.7856051897, -0.5802658508, -3.2756018040, -0.8650476464, -0.8425924762, -0.6069950486, -6.6900028561, -7.9841025522, -0.7396182508, -0.7791182897, -0.6172681546, -4.8542858770, -0.8170603310, -0.8322953918, -0.5330896837, -9.9686679636, -6.4087483649, -0.9364771801, -0.8903909389, -0.5548445754, -6.7711216208, -7.0543588922, -0.7765195629, -0.7957334675, -0.6177983338, -5.6826468362, -0.8773458924, -0.8138214365, -0.5988472210, -7.9498220570, -5.9712849686]
    elif self.action_representation_mode == "Euler":
      out_min = [-1.4809598333, -3.7021475311, -1.5463856331, -0.7133202303, -3.1300459865, -0.7387329973, -2.5683532245, -5.9070781577, -2.6906588532, -7.9841025522, -2.6260947392, -4.1109668383, -2.4382549267, -6.4794088421, -8.7559772942, -6.4201063485, -6.4087483649, -2.7326217706, -6.0387098656, -3.1014777221, -7.0543588922, -2.5063782767, -4.9841185712, -2.5020967882, -3.5604201522, -6.7435288766, -4.2452592148, -5.9712849686]
    elif self.action_representation_mode == "Quaternion":
      out_min = [-0.6830304623, -0.6496953570, -1.8412844001, 0.7189876519, -0.3394406461, -0.3259007397, -1.5578146125, 0.8504601347, -1.0176964949, -0.9562844111, -2.9438376236, 0.4255056683, -7.9841025522, -1.0436775944, -1.1239794670, -2.0117464017, 0.4603177734, -1.0371820747, -1.0557227724, -4.4420140970, -0.2744070946, -6.4087483649, -1.0723144025, -0.8706846212, -2.9980840104, 0.3235760571, -7.0543588922, -0.9811759274, -0.9818374347, -2.4767496576, 0.4552725456, -1.3734428477, -1.0404966672, -3.3877144126, 0.1543603310, -5.9712849686]
    elif self.action_representation_mode == "RotMat":
      out_min = [0.0002767857, -3.1959703729, -1.2819288197, -3.5329021648, -0.0063152284, -1.3665628266, -1.3017272430, -1.3416508521, 0.7256425458, 0.4369078375, -2.9527598433, -0.6430115659, -3.0481604276, 0.4360620439, -0.6723290276, -0.6506331995, -0.6708707216, 0.9292909159, -0.9335782277, -5.5127465039, -1.8595090151, -5.4412941187, -0.9508993397, -1.9630961702, -1.8397420633, -1.9597317880, 0.4304696712, -7.9841025522, -0.7847900518, -4.1290571096, -2.2417657672, -3.7831124133, -0.7132929204, -2.0142748442, -2.1834837772, -2.1347509162, 0.1608246170, -3.0668968305, -7.5588979583, -2.0636131436, -7.4961711741, -3.0482571676, -1.9967518021, -1.9877270714, -2.0715546969, 0.3560788295, -6.4087483649, -1.2599587673, -5.6761826959, -1.7846503513, -5.4594934467, -1.3148853132, -1.9805291589, -1.6686014427, -2.1010739224, 0.4666106711, -7.0543588922, -0.8366798214, -4.9510476202, -1.9156738040, -4.5934634489, -0.8433671428, -1.9481324764, -1.8811056054, -1.9124852972, 0.3418394257, -1.5539876917, -5.9830990153, -2.1080069242, -6.0708800546, -1.6905740546, -2.5963012616, -2.0757622750, -2.5722715912, 0.1693028182, -5.9712849686]
    elif self.action_representation_mode == "RotVec":
      out_min = [-1.3845474768, -1.3172095865, -3.7348817649, -0.6835879720, -0.6564560685, -3.1384403845, -2.0942962504, -1.9691459485, -6.0500950343, -7.9841025522, -2.1450177734, -2.3139085018, -4.1376874574, -2.2077474087, -2.2542336684, -9.4641471440, -6.4087483649, -2.2140310814, -1.8018334194, -6.1999125422, -7.0543588922, -2.0168959588, -2.0227641973, -5.0884646448, -2.8806313327, -2.1684033239, -7.0667294524, -5.9712849686]
    elif self.action_representation_mode == "6D":
      out_min = [0.0002767857, -3.1959703729, -1.2819288197, -3.5329021648, -0.0063152284, -1.3665628266, 0.4369078375, -2.9527598433, -0.6430115659, -3.0481604276, 0.4360620439, -0.6723290276, -0.9335782277, -5.5127465039, -1.8595090151, -5.4412941187, -0.9508993397, -1.9630961702, -7.9841025522, -0.7847900518, -4.1290571096, -2.2417657672, -3.7831124133, -0.7132929204, -2.0142748442, -3.0668968305, -7.5588979583, -2.0636131436, -7.4961711741, -3.0482571676, -1.9967518021, -6.4087483649, -1.2599587673, -5.6761826959, -1.7846503513, -5.4594934467, -1.3148853132, -1.9805291589, -7.0543588922, -0.8366798214, -4.9510476202, -1.9156738040, -4.5934634489, -0.8433671428, -1.9481324764, -1.5539876917, -5.9830990153, -2.1080069242, -6.0708800546, -1.6905740546, -2.5963012616, -5.9712849686]

    return out_min

  def build_action_bound_max(self, agent_id):
    # out_max = [1] * self.get_action_size(agent_id)

    # REPRESENTATION_MODE_CHECKPOINT
    # out_max[self.action_dim * 3] = 4.71
    # out_max[self.action_dim * 5 + 1] = 7.85
    # out_max[self.action_dim * 6 + 2] = 4.71
    # out_max[self.action_dim * 8 + 3] = 7.85

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

    # Back to the base setting
    # out_max = [
    #     1.000000000, 1.000000000, 1.000000000, 4.799999999, # chest
    #     1.000000000, 1.000000000, 1.000000000, 4.000000000, # neck

    #     1.000000000, 1.000000000, 1.000000000, 8.779999999, # rightHip
    #     4.710000000, # rightKnee
    #     1.000000000, 1.000000000, 1.000000000, 6.280000000, # rightAnkle
    #     1.000000000, 1.000000000, 1.000000000, 12.56000000, # rightShoulder
    #     7.850000000, # rightElbow

    #     1.000000000, 1.000000000, 1.000000000, 8.779999999, # leftHip
    #     4.710000000, # leftKnee
    #     1.000000000, 1.000000000, 1.000000000, 6.280000000, # leftAnkle
    #     1.000000000, 1.000000000, 1.000000000, 10.10000000, # leftShoulder. why is it not symmetric to rightShoulder?
    #     7.850000000 # leftElbow
    # ]
    if self.action_representation_mode == "AxisAngle":
      out_max = [0.8218378593, 0.7839523679, 0.9769547093, 3.6891425858, 0.7958683095, 0.7945701860, 1.1229439230, 3.1716984128, 0.9484406134, 0.9075597495, 1.0458365478, 6.7737462133, 7.5568142476, 0.8310090296, 0.8874240466, 0.9575735704, 5.4777885765, 0.8335620378, 0.8025699505, 1.0430297369, 10.0836350720, 7.4775274464, 0.7950413283, 0.8026879044, 1.0799970183, 7.0523805567, 6.8011767365, 0.8349331599, 0.8356451205, 0.9897175184, 6.1242312547, 0.7542792188, 0.8492521646, 0.9952554801, 7.8408155347, 6.8945466170]
    elif self.action_representation_mode == "Euler":
      out_max = [1.4920273952, 3.3500179388, 1.5626846927, 0.7071690207, 3.0305951242, 0.7290903991, 2.5313896877, 5.9925590296, 2.6741834972, 7.5568142476, 2.7863818358, 4.6311140450, 2.5409350653, 6.4672606045, 8.8557302998, 6.4354143123, 7.4775274464, 2.7552679096, 6.3073416552, 3.0815520128, 6.8011767365, 2.5011652555, 5.3811132285, 2.5736585818, 3.6354529169, 6.6512250570, 4.2785109672, 6.8945466170]
    elif self.action_representation_mode == "Quaternion":
      out_max = [0.6897771740, 0.6533681563, 1.6657235305, 1.2532953568, 0.3352298784, 0.3247812871, 1.5082225876, 1.1361837133, 1.0234892765, 0.9567077117, 2.9814196569, 1.5193109429, 7.5568142476, 1.1146613206, 1.2152150154, 2.2650866987, 1.4834058604, 1.0281394501, 1.0560069731, 4.4940215334, 2.1543745678, 7.4775274464, 1.0485305835, 0.8552870996, 3.1261009487, 1.6140086749, 6.8011767365, 1.0193966354, 0.9943885959, 2.6722737746, 1.4905540440, 1.3946344963, 1.0362799226, 3.3430429534, 1.7702381957, 6.8945466170]
    elif self.action_representation_mode == "RotMat":
      out_max = [1.9003024181, 3.5361192741, 1.2861923802, 3.1932415821, 1.9059479348, 1.3515538023, 1.2919952342, 1.3528933057, 1.2567438275, 1.5124940815, 3.0500860668, 0.6444206024, 2.9512455507, 1.5131676109, 0.6813835676, 0.6564716859, 0.6633799113, 1.0663851200, 2.7422013971, 5.4513342556, 1.8905392131, 5.5178682518, 2.7570053195, 1.9767144808, 1.8695525848, 1.9943389818, 1.5311493001, 7.5568142476, 2.5958454727, 3.6657610451, 2.4510589061, 4.2735252128, 2.5316192565, 1.9284696137, 2.0505369317, 2.3155451845, 1.7761745056, 4.6510625133, 7.4738673359, 2.0739055377, 7.5795787097, 4.6332910248, 1.9896064131, 1.9975622174, 2.0345028506, 1.6010870387, 7.4775274464, 3.0449587101, 5.4508661374, 1.7090514978, 5.7009988645, 3.0921135448, 1.9948309276, 1.6491929225, 2.0282832770, 1.4959494621, 6.8011767365, 2.6492448424, 4.5863607071, 1.9659336083, 4.9648570156, 2.6541544489, 1.8907410753, 1.8837525374, 1.9987384475, 1.6177941966, 3.3088555025, 6.0561555451, 2.0392490957, 5.9890661826, 3.4285992295, 2.5758224418, 2.0214525484, 2.6260226550, 1.7715515743, 6.8945466170]
    elif self.action_representation_mode == "RotVec":
      out_max = [1.3981653289, 1.3246744684, 3.3799370150, 0.6751161778, 0.6542091996, 3.0385579113, 2.1062638908, 1.9700741390, 6.1274214379, 7.5568142476, 2.2899989896, 2.5003288439, 4.6547167461, 2.1884609840, 2.2549508862, 9.5757179839, 7.4775274464, 2.1650376705, 1.7700798336, 6.4639638287, 6.8011767365, 2.0949383075, 2.0483095361, 5.4874666823, 2.9251682574, 2.1595726844, 6.9727153786, 6.8945466170]
    elif self.action_representation_mode == "6D":
      out_max = [1.9003024181, 3.5361192741, 1.2861923802, 3.1932415821, 1.9059479348, 1.3515538023, 1.5124940815, 3.0500860668, 0.6444206024, 2.9512455507, 1.5131676109, 0.6813835676, 2.7422013971, 5.4513342556, 1.8905392131, 5.5178682518, 2.7570053195, 1.9767144808, 7.5568142476, 2.5958454727, 3.6657610451, 2.4510589061, 4.2735252128, 2.5316192565, 1.9284696137, 4.6510625133, 7.4738673359, 2.0739055377, 7.5795787097, 4.6332910248, 1.9896064131, 7.4775274464, 3.0449587101, 5.4508661374, 1.7090514978, 5.7009988645, 3.0921135448, 1.9948309276, 6.8011767365, 2.6492448424, 4.5863607071, 1.9659336083, 4.9648570156, 2.6541544489, 1.8907410753, 3.3088555025, 6.0561555451, 2.0392490957, 5.9890661826, 3.4285992295, 2.5758224418, 6.8945466170]

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
