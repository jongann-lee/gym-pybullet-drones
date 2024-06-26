import numpy as np

from gym_pybullet_drones.envs.BaseGeoRLAviary import BaseGeoRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class GeoHoverAviary(BaseGeoRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=np.array([[0,0,1] for i in range(1)]),
                 initial_rpys=np.array([[np.pi/3, np.pi/3, 0] for i in range(1)]),
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 120,
                 update_freq: int = 10,
                 gui=False,
                 record=False,
                 train = False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        update_freq : int, optional
            The frequency at which the environment steps (and updates the controller parameters).
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.TARGET_POS = np.array([0,0,1])
        self.EPISODE_LEN_SEC = 2
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         update_freq=update_freq,
                         gui=gui,
                         record=record,
                         train = train,
                         obs=obs,
                         act=act
                         )

    ################################################################################
    
    def _computeReward(self, action):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        obs = self._getDroneStateVector(0)
        time_elapsed = int(self.step_counter/self.PYB_FREQ)  # elapsed time in seconds 
        error = self.error_buffer[-1][:]
        pos_e = error[0:3]
        rot_e = error[6]
        omega_e = error[7:10]
        
        reward_angle = (1/24)*(2 - rot_e)# + (0.5 * time_elapsed) * (1 - np.linalg.norm(omega_e))
        reward_position = (1/12) * (1 - np.linalg.norm(pos_e)) - (time_elapsed/24) * (np.abs(action[0,0]) + np.abs(action[0,1]))
        return reward_position

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0000001:
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 2 or abs(state[1]) > 2 or state[2] > 3  # Truncate when the drone is too far away or when it hits the ground
             # Truncate when the drone is too tilted or state[2] < 0.001
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
