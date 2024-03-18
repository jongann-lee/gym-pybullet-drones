import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary

class GeometricControl(BaseControl):
    """
    Implementation of the geometric controller for a quadrotor by Taeyoung Lee

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()
        self.k_x = 2
        self.k_v = 18
        self.k_R = 5
        self.k_Omega = 1
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([ [.5, -.5,  -1], [.5, .5, 1], [-.5,  .5,  -1], [-.5, -.5, 1] ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([ [0, -1,  -1], [+1, 0, 1], [0,  1,  -1], [-1, 0, 1] ])
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    ################################################################################
    
    def computeControl(self,
                       drone_m,
                       drone_J,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_angular_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_acc=np.zeros(3),
                       target_angular_vel=np.zeros(3),
                       target_angular_acc=np.zeros(3)
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`.
        Parameter `cur_ang_vel` is unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_angular_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity in the world frame.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_angular_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired angular velocity in the world frame.
        target_angular_acc : ndarray, optional
            (3,1)-shaped array of floats containing the desired angular accelaration in the world frame.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(drone_m,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel,
                                                                         target_acc
                                                                         )
        rpm = self._dslPIDAttitudeControl(drone_J,
                                          thrust,
                                          cur_quat,
                                          cur_angular_vel,
                                          computed_target_rpy,
                                          target_angular_vel,
                                          target_angular_acc
                                          )
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]
    
    ################################################################################

    def _dslPIDPositionControl(self,
                               drone_m,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel,
                               target_acc
                               ):
        """Geometric Controller: Position Control

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.
        target_acc : ndarray
            (3,1)-shaped array of floats containing the desired accelaration.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = cur_pos - target_pos
        vel_e = cur_vel - target_vel
        #### Geometric target thrust #####################################
        target_thrust = -self.k_x*pos_e - self.k_v*vel_e + drone_m*np.array([0, 0, 9.8]) + drone_m*target_acc
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        #### Target rotation #######################################
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter, "in Control._GeometricControl(), values outside range [-pi,pi]")
        return thrust, target_euler, pos_e
    
    ################################################################################

    def _dslPIDAttitudeControl(self,
                               drone_J,
                               thrust,
                               cur_quat,
                               cur_angular_vel,
                               target_euler,
                               target_angular_vel,
                               target_angular_acc
                               ):
        """Geometric Controller: Attitude Control

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_angular_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity in the world frame.
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles.
        target_angular_vel : ndarray
            (3,1)-shaped array of floats containing the desired angular velocity in the world frame.
        target_angular_acc : ndarray
            (3,1)-shaped array of floats containing the desired angular accelration in the world frame

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        
        # Calculate the attitude error
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        #cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = 0.5 * np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 

        # Convert all angular velocity and angular accelaration to the body frame
        cur_angular_vel = cur_rotation.transpose() @ cur_angular_vel
        target_angular_vel = target_rotation.transpose() @ target_angular_vel
        target_angular_acc = target_rotation.transpose() @ target_angular_acc

        # Calculate the angular velocity error
        omega_e = cur_angular_vel - cur_rotation.transpose()@target_rotation@target_angular_vel
        cur_angular_vel_hat = np.array([[0, -cur_angular_vel[2], cur_angular_vel[1]], \
                                        [cur_angular_vel[2], 0, -cur_angular_vel[0]], \
                                        [-cur_angular_vel[1], cur_angular_vel[0], 0]])
        #### PID target torques ####################################
        target_torques = - self.k_R*rot_e - self.k_Omega*omega_e + np.cross(cur_angular_vel, drone_J@cur_angular_vel) \
                        - drone_J @ (cur_angular_vel_hat@cur_rotation.transpose()@target_rotation@target_angular_vel - cur_rotation.transpose()@target_rotation@target_angular_acc)
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
    
    
    ################################################################################

    def _one23DInterface(self,
                         thrust
                         ):
        """Utility function interfacing 1, 2, or 3D thrust input use cases.

        Parameters
        ----------
        thrust : ndarray
            Array of floats of length 1, 2, or 4 containing a desired thrust input.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.

        """
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust)/(self.KF*(4/DIM)))-self.PWM2RPM_CONST)/self.PWM2RPM_SCALE, self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4/DIM)
        elif DIM==2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()
