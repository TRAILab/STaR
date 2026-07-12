import subprocess
from dataclasses import dataclass

DEFAULT_GOAL_TOPIC = "/goal_pose"

@dataclass
class GoalPose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0 
    qw: float = 1.0

    frame_id: str = "map"

    @classmethod
    def from_dict(cls, data: dict) -> "GoalPose":
        return cls(
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            z=data.get("z", 0.0),
            qx=data.get("qx", 0.0),
            qy=data.get("qy", 0.0),
            qz=data.get("qz", 0.0),
            qw=data.get("qw", 1.0),
            frame_id=data.get("frame_id", "map"),
        )
    
    def to_pose_stamped(self) -> str:
        return f'''
        header:
            frame_id: {self.frame_id}
        pose:
            position:
                x: {self.x}
                y: {self.y}
                z: {self.z}
            orientation:
                x: {self.qx}
                y: {self.qy}
                z: {self.qz}
                w: {self.qw}
        '''


def publish_goal(pred_pose: dict, topic: str = DEFAULT_GOAL_TOPIC) -> None:
    goal: GoalPose = GoalPose.from_dict(pred_pose)
    pose_stamped: str = goal.to_pose_stamped()
    
    subprocess.run([
        "ros2", "topic", "pub", "--once",
        topic, "geometry_msgs/msg/PoseStamped", pose_stamped
    ], check=True)