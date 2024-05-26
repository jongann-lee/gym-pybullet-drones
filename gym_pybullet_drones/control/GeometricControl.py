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
            print("[ERROR] in GeometricControl.__init__(), GeometricControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()
        self.k_x = 5 * 0.01         # optimal value 5 * 0.01
        self.k_v = 4 * 0.01         # optimal value 4 * 0.01
        self.k_R = 50 * 0.01       # optimal value 5 * 0.01
        self.k_omega = 0.5 * 0.01   # optimal value 0.1 * 0.01
        #print("GeoCon Init")
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 0
        self.MAX_PWM = 65535
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.L = self.L / np.sqrt(2) # to get the proper arm length for moment calculation
            k_yaw = self.KM / self.KF
            self.MIXER_MATRIX = np.array([ 
                                    [1, 1, 1, 1],
                                    [-self.L, -self.L, self.L, self.L],
                                    [-self.L, self.L, self.L, -self.L],
                                    [-k_yaw, k_yaw, -k_yaw, k_yaw]
                                    ]) # To be multiplied to the individual force components
            self.inv_mixer = np.linalg.inv(self.MIXER_MATRIX)

        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                                    [0, -1,  -1],
                                    [+1, 0, 1],
                                    [0,  1,  -1],
                                    [-1, 0, 1]
                                    ])
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
        """Computes the Geometric control action (as RPMs) for a single drone.

        This methods sequentially calls `_GeoPositionControl()` and `_GeoAttitudeControl()`.

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
        thrust, computed_target_rotation, pos_e = self._GeoPositionControl(drone_m,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel,
                                                                         target_acc
                                                                         )
        rpm = self._GeoAttitudeControl(drone_m,
                                          drone_J,
                                          thrust,
                                          cur_quat,
                                          cur_angular_vel,
                                          computed_target_rotation,
                                          target_angular_vel,
                                          target_angular_acc
                                          )
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e
    
    ################################################################################

    def _GeoPositionControl(self,
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
            The current position error.arrauy

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = cur_pos - target_pos # e_x
        vel_e = cur_vel - target_vel # e_v
        #### Geometric target thrust #####################################
        target_thrust = - self.k_x*pos_e - self.k_v*vel_e + drone_m*np.array([0, 0, 9.8]) + drone_m*target_acc
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        #thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE # thrust in PWM
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose() # check to see if correct and not transposed incorrectly
        target_rpy = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        return scalar_thrust, target_rotation, pos_e
    
    ################################################################################

    def _GeoAttitudeControl(self,
                               drone_m,
                               drone_J,
                               thrust,
                               cur_quat,
                               cur_angular_vel,
                               target_rotation,
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
        rot_matrix_e = (target_rotation.transpose())@cur_rotation - (cur_rotation.transpose())@target_rotation
        rot_e = 0.5 * np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])

        # Calculate the desired angular velocity and accelaration
        x_b = cur_rotation[:,0]
        y_b = cur_rotation[:,1]
        z_b = cur_rotation[:,2]
        x_3d = np.zeros(3) # modify if we ever use a non stationary trajectory
        x_4d = np.zeros(3) # modify if we ever use a non stationary trajectory
        dyaw_d = 0
        ddyaw_d = 0
        f_dot = drone_m * np.dot(z_b, x_3d)
        if thrust == 0 : 
            thrust = thrust + 1e-8
        h_w = (drone_m / thrust) * (x_3d - (f_dot / drone_m) * z_b)
        p_omega = -np.dot(h_w, y_b)
        q_omega = np.dot(h_w, x_b)
        r_omega = dyaw_d * np.dot(z_b, np.array([0,0,1]))
        f_ddot = drone_m * np.dot(z_b, x_4d) - thrust * np.dot(z_b, np.cross(cur_angular_vel, np.cross(cur_angular_vel, z_b)))
        h_alpha = (drone_m / thrust) * (x_4d - (f_ddot/drone_m)*z_b - 2*(f_dot/drone_m)*np.cross(cur_angular_vel,z_b))\
            - np.cross(cur_angular_vel, np.cross(cur_angular_vel, z_b))
        p_alpha = -np.dot(h_alpha, x_b)
        q_alpha = np.dot(h_alpha, y_b)
        r_alpha = ddyaw_d * np.dot(z_b, np.array([0,0,1]))

        # Convert all angular velocity and angular accelaration to the body frame
        cur_angular_vel = cur_rotation.transpose() @ cur_angular_vel
        target_angular_vel = np.array([p_omega, q_omega, r_omega])
        target_angular_acc = np.array([p_alpha, q_alpha, r_alpha])

        # Calculate the angular velocity error
        omega_e = cur_angular_vel - cur_rotation.transpose()@target_rotation@target_angular_vel
        cur_angular_vel_hat = np.array([[0, -cur_angular_vel[2], cur_angular_vel[1]], \
                                        [cur_angular_vel[2], 0, -cur_angular_vel[0]], \
                                        [-cur_angular_vel[1], cur_angular_vel[0], 0]])
        
        #### PID target torques ####################################
        target_torques =  - self.k_R*rot_e - self.k_omega*omega_e + np.cross(cur_angular_vel, drone_J@cur_angular_vel) \
                       - drone_J @ (cur_angular_vel_hat@cur_rotation.transpose()@target_rotation@target_angular_vel - cur_rotation.transpose()@target_rotation@target_angular_acc)
        target_torques = np.clip(target_torques, -3200, 3200)
        force_wrench = np.hstack([thrust, target_torques])
        force_component = self.inv_mixer @ force_wrench
        force_component = np.maximum(np.zeros(4), force_component)
        rpm = (np.sqrt(force_component / self.KF)) 
        return rpm
    
    
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
