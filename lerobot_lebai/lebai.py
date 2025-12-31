import logging
import time
import os
import copy
from typing import Any

import numpy as np
import transforms3d as t3d
import lebai_sdk

from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots.robot import Robot

from .config_lebai import LebaiConfig
JOINT_NUM = 6

logger = logging.getLogger(__name__)


class Lebai(Robot):
    config_class = LebaiConfig
    name = "lebai"

    def __init__(self, config: LebaiConfig):
        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras)

        this_dir = os.path.dirname(os.path.abspath(__file__))

        self.config = config
        self._arm = None
        self._initial_pose = None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect to the robot
        self._arm = lebai_sdk.connect(self.config.ip, self.config.port)
        self._arm.start_sys()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def _motors_ft(self) -> dict[str, type]:
        motors = {f"joint{i}.pos": float for i in range(1, JOINT_NUM + 1)}
        motors["gripper.pos"] = float
        return motors

    @property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        action = copy.deepcopy(action)

        if "delta_x" in action and "delta_y" in action and "delta_z" in action:
            pose = self._arm.get_kin_data().actual_tcp_pose
            target = copy.deepcopy(pose)
            if (
                "delta_roll" in action
                and "delta_pitch" in action
                and "delta_yaw" in action
            ):
                roll = action["delta_roll"]
                pitch = action["delta_pitch"]
                yaw = action["delta_yaw"]
                target = lebai:pose_trans(pose, {'x' : 0, 'y' : 0, 'z' : 0, 'rz' : roll, 'ry' : pitch, 'rx' : yaw} )

            target['x']=target['x']+action["delta_x"]
            target['y']=target['y']+action["delta_y"]
            target['z']=target['z']+action["delta_z"]
            self._arm.movel(target, self.config.a, self.config.v, 0, self.config.r)
            return action

        if "pose" in action:
            pose = action["pose"]
            xyz = pose[:3, 3]
            rpy = t3d.euler.mat2euler(pose[:3, :3])
            target = {'x' : xyz[0], 'y' : xyz[1], 'z' : xyz[2], 'rz' : rpy[0], 'ry' : rpy[1], 'rx' : rpy[2]}
            self._arm.movel(pose, self.config.a, self.config.v, 0, self.config.r)
            return action

        # Execute joint positions
        if "joint1.pos" in action:
            joint_positions = []
            for i in range(1, JOINT_NUM + 1):
                joint_pos = action[f"joint{i}.pos"]
                joint_positions.append(joint_pos)

            self._arm.movej(joint_positions, self.config.a, self.config.v, 0, self.config.r)
            return action

        # Send gripper command
        if "gripper.pos" in action or "gripper" in action:
            gripper_pos = action.get("gripper.pos", action.get("gripper", 0.0))
            self._arm.set_claw(self.config.gripper_force, gripper_pos)
            return action

        return action

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()

        # Read joint positions
        kin_data = self._arm.get_kin_data()

        obs_dict = {}
        for i, angle in enumerate(kin_data.actual_joint_pose):
            obs_dict[f"joint{i+1}.pos"] = angle
            obs_dict[f"joint{i+1}.effort"] = kin_data.actual_joint_torque[i]
            obs_dict[f"joint{i+1}.vel"] = kin_data.actual_joint_speed[i]
            obs_dict[f"joint{i+1}.acc"] = kin_data.actual_joint_acc[i]
        obs_dict["gripper.pos"] = self._arm.get_claw().amplitude

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def reset(self):
        self._arm.init_claw(True)

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        if self._arm is not None:
            self._arm.stop_sys()
            self._arm = None

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._initial_pose = self._arm.get_kin_data().actual_tcp_pose
        print(f"Initial pose: {self._initial_pose}")

    def is_calibrated(self) -> bool:
        return self.is_connected

    @property
    def is_connected(self) -> bool:
        return self._arm is not None and self._arm.is_connected()

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict[str, Any]:
        features = {**self._motors_ft, **self._cameras_ft}
        if self.config.use_effort:
            for i in range(1, JOINT_NUM + 1):
                features[f"joint{i}.effort"] = float
        if self.config.use_velocity:
            for i in range(1, JOINT_NUM + 1):
                features[f"joint{i}.vel"] = float
        if self.config.use_acceleration:
            for i in range(1, JOINT_NUM + 1):
                features[f"joint{i}.acc"] = float
        return features

    @property
    def cameras(self):
        return self._cameras

    @cameras.setter
    def cameras(self, value):
        self._cameras = value

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value
