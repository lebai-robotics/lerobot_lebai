# LeRobot + Lebai Integration

Brings a simple integration with [LeRobot](https://github.com/huggingface/lerobot) and [Lebai](https://lebai.ltd).

## Getting Started

```bash
pip install lerobot_lebai lerobot-teleoperator-teleop

lerobot-teleoperate \
    --robot.type=lerobot_lebai \
    --robot.id=black \
    --teleop.type=lerobot_teleoperator_teleop \
    --fps=60
```

## Development

Install the package in editable mode:

```bash
git clone https://github.com/lebai-robotics/lerobot_lebai.git
cd lerobot_lebai
pip install -e .
```
