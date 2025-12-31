from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig

@RobotConfig.register_subclass("lerobot_lebai")
@dataclass
class LebaiConfig(RobotConfig):
    ip: str = "192.168.1.184"
    port: int = 3031
    gripper_force: int = 100
    a: int = 1
    v: int = 1
    r: int = 0
    use_effort: bool = False
    use_velocity: bool = True
    use_acceleration: bool = True
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
