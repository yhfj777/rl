"""
轨迹智能体：每条活跃轨迹对应一个独立的PPO Agent
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import torch
from collections import defaultdict


@dataclass
class Detection:
    """检测结果类"""
    frame: int
    track_id: int  # 真实轨迹ID（用于评估）
    x: float
    z: float
    v: float
    pha: float
    w: float
    a: float
    label: int  # 1=真实点，0=噪声点
    is_occluded: bool = False  # 是否为消失点

    @classmethod
    def from_array(cls, data) -> 'Detection':
        """从numpy数组创建Detection对象"""
        def is_value_occluded(val):
            return abs(float(val) - (-1.0)) < 1e-6

        return cls(
            frame=int(float(data[0])),
            track_id=int(float(data[1])),
            x=float(data[2]),
            z=float(data[3]),
            v=float(data[4]),
            pha=float(data[5]),
            w=float(data[6]),
            a=float(data[7]),
            label=int(float(data[8])),
            is_occluded=any(is_value_occluded(data[i]) for i in range(2, 9))
        )

    @classmethod
    def from_dict(cls, data: dict) -> 'Detection':
        """从字典创建Detection对象"""
        return cls(
            frame=int(data['frame']),
            track_id=int(data['track_id']) if 'track_id' in data else -1,
            x=float(data['x']),
            z=float(data['z']),
            v=float(data.get('v', -1)),
            pha=float(data.get('pha', -1)),
            w=float(data.get('w', -1)),
            a=float(data.get('a', -1)),
            label=int(data.get('label', 1)),
            is_occluded=bool(data.get('is_occluded', False))
        )


@dataclass
class Track:
    """轨迹类"""
    track_id: int
    gt_track_id: int = -1  # 真实轨迹ID（用于评估）
    detections: List[Detection] = field(default_factory=list)
    is_active: bool = True
    last_update_frame: int = -1
    confirmed: bool = False
    missed_frames: int = 0
    hidden_state: Optional[torch.Tensor] = None  # 策略网络的隐藏状态
    policy_hidden: Optional[torch.Tensor] = None  # 简化的策略隐藏状态
    _tracker_confirm_frames: int = 2

    def add_detection(self, detection: Detection):
        """添加检测结果到轨迹"""
        self.detections.append(detection)
        self.last_update_frame = detection.frame
        self.is_active = not detection.is_occluded
        self.missed_frames = 0
        if self.get_track_length() >= self._tracker_confirm_frames:
            self.confirmed = True

    def add_occlusion(self, frame: int):
        """添加消失点"""
        occlusion = Detection(
            frame=frame,
            track_id=self.track_id,
            x=-1, z=-1, v=-1, pha=-1, w=-1, a=-1,
            label=-1,
            is_occluded=True
        )
        self.detections.append(occlusion)
        self.missed_frames += 1
        track_length = self.get_track_length()
        if self.confirmed:
            self.is_active = self.missed_frames <= 10
        elif track_length >= 2:
            # Let tentative tracks survive a bit longer to increase chance of
            # obtaining the second/third valid hit and becoming stable.
            self.is_active = self.missed_frames <= 6
        else:
            self.is_active = self.missed_frames <= 3

    def get_last_position(self) -> Optional[Tuple[float, float]]:
        """获取轨迹最后位置"""
        if not self.detections:
            return None
        last_det = self.detections[-1]
        if last_det.is_occluded:
            return None
        return (last_det.x, last_det.z)

    def get_last_velocity(self) -> Optional[Tuple[float, float]]:
        """获取轨迹最后速度（基于位置变化）"""
        if len(self.detections) < 2:
            return None

        valid_dets = [d for d in self.detections if not d.is_occluded]
        if len(valid_dets) < 2:
            return None

        d1, d2 = valid_dets[-2], valid_dets[-1]
        if d2.frame <= d1.frame:
            return None

        vx = (d2.x - d1.x) / (d2.frame - d1.frame)
        vz = (d2.z - d1.z) / (d2.frame - d1.frame)
        return (vx, vz)

    def get_history_positions(self, max_frames: int = 20) -> np.ndarray:
        """获取历史位置序列"""
        valid_dets = [d for d in self.detections if not d.is_occluded]
        positions = [(d.x, d.z) for d in valid_dets[-max_frames:]]
        # 填充
        while len(positions) < max_frames:
            if positions:
                positions.insert(0, positions[0])
            else:
                positions.insert(0, (0, 0))
        return np.array(positions, dtype=np.float32)

    def get_history_velocities(self, max_frames: int = 20) -> np.ndarray:
        """获取历史速度序列（与位置序列维度一致）"""
        valid_dets = [d for d in self.detections if not d.is_occluded]

        # 获取与位置序列相同范围的数据
        position_dets = valid_dets[-max_frames:]
        velocities = []

        # 第一个点没有前一帧速度，设为0
        if len(position_dets) > 0:
            velocities.append((0.0, 0.0))

        # 计算速度（与位置点一一对应）
        for i in range(1, len(position_dets)):
            d1 = position_dets[i - 1]
            d2 = position_dets[i]
            dt = d2.frame - d1.frame
            if dt > 0:
                vx = (d2.x - d1.x) / dt
                vz = (d2.z - d1.z) / dt
                velocities.append((vx, vz))
            else:
                velocities.append((0.0, 0.0))

        # 填充到 max_frames 长度
        while len(velocities) < max_frames:
            velocities.insert(0, (0.0, 0.0))

        return np.array(velocities, dtype=np.float32)

    def get_track_length(self) -> int:
        """获取轨迹长度（非消失点数）"""
        return len([d for d in self.detections if not d.is_occluded])

    def get_smoothness(self) -> float:
        """计算轨迹平滑度"""
        valid_dets = [d for d in self.detections if not d.is_occluded]
        if len(valid_dets) < 3:
            return 1.0

        smoothness = 0
        count = 0
        for i in range(2, len(valid_dets)):
            d1, d2, d3 = valid_dets[i - 2], valid_dets[i - 1], valid_dets[i]
            # 计算角度变化
            v1 = np.array([d2.x - d1.x, d2.z - d1.z])
            v2 = np.array([d3.x - d2.x, d3.z - d2.z])

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 0.01 and norm2 > 0.01:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)
                smoothness += (1 - cos_angle) / 2
                count += 1

        return 1.0 - (smoothness / max(1, count)) if count > 0 else 1.0


class TrackingAgent:
    """
    轨迹追踪智能体

    每条活跃轨迹对应一个智能体，智能体负责：
    1. 维护轨迹状态
    2. 编码状态
    3. 选择动作（关联哪个候选点）
    4. 接收奖励并更新策略
    """

    def __init__(
            self,
            track_id: int,
            gt_track_id: int = -1,
            max_history_frames: int = 20,
            device: torch.device = None
    ):
        self.track_id = track_id
        self.gt_track_id = gt_track_id
        self.max_history_frames = max_history_frames
        self.device = device

        # 轨迹状态
        self.track = Track(track_id=track_id, gt_track_id=gt_track_id)

        # 策略相关
        self.hidden_state = None  # 策略网络隐藏状态
        self.actions: List[int] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.states: List[Dict[str, torch.Tensor]] = []  # 存储状态用于PPO更新

        # 统计数据
        self.episode_length = 0
        self.episode_reward = 0
        self.last_collected_idx = 0  # 上次收集的转换索引

    def reset(self, track_id: int = None, gt_track_id: int = None):
        """重置智能体状态"""
        if track_id is not None:
            self.track_id = track_id
        if gt_track_id is not None:
            self.gt_track_id = gt_track_id

        self.track = Track(track_id=self.track_id, gt_track_id=self.gt_track_id)
        self.hidden_state = None
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.states = []
        self.episode_length = 0
        self.episode_reward = 0
        self.last_collected_idx = 0

    def add_detection(self, detection: Detection):
        """添加检测结果"""
        self.track.add_detection(detection)
        self.episode_length += 1

    def add_occlusion(self, frame: int):
        """添加消失点"""
        self.track.add_occlusion(frame)
        self.episode_length += 1

    def record_transition(
            self,
            action: int,
            reward: float,
            value: torch.Tensor,
            log_prob: torch.Tensor,
            done: bool,
            state_dict: Dict[str, torch.Tensor] = None
    ):
        """记录转换"""
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.episode_reward += reward

        # 存储状态用于PPO更新
        if state_dict is not None:
            self.states.append(state_dict)

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """获取状态字典（用于策略网络输入）"""
        # 历史位置
        history_positions = self.track.get_history_positions(self.max_history_frames)
        history_velocities = self.track.get_history_velocities(self.max_history_frames)

        # 转换为张量（不添加批次维度）
        state_dict = {
            'history_positions': torch.tensor(history_positions, dtype=torch.float32, device=self.device),
            'history_velocities': torch.tensor(history_velocities, dtype=torch.float32, device=self.device),
        }

        return state_dict

    def set_hidden_state(self, hidden_state: torch.Tensor):
        """设置策略网络隐藏状态"""
        self.hidden_state = hidden_state

    def get_track_info(self) -> Dict:
        """获取轨迹信息"""
        return {
            'track_id': self.track_id,
            'gt_track_id': self.gt_track_id,
            'length': self.track.get_track_length(),
            'smoothness': self.track.get_smoothness(),
            'is_active': self.track.is_active,
            'last_frame': self.track.last_update_frame,
            'episode_reward': self.episode_reward
        }

    def get_transition_data(self) -> Dict:
        """获取转换数据（用于PPO更新）"""
        return {
            'log_probs': self.log_probs,
            'values': self.values,
            'rewards': self.rewards,
            'dones': self.dones,
            'hidden_state': self.hidden_state
        }

    def clear_transition_data(self):
        """清空转换数据"""
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []


class AgentManager:
    """
    智能体管理器

    负责：
    1. 管理所有活跃轨迹的智能体
    2. 协调智能体的动作选择
    3. 收集转换数据
    """

    def __init__(
            self,
            max_history_frames: int = 20,
            max_agents: int = 100,
            device: torch.device = None
    ):
        self.max_history_frames = max_history_frames
        self.max_agents = max_agents
        self.device = device

        # 活跃智能体字典
        self.active_agents: Dict[int, TrackingAgent] = {}
        
        # 保存所有创建过的智能体（用于评估）
        self.all_agents: Dict[int, TrackingAgent] = {}

        # 智能体ID生成器
        self.next_agent_id = 1

        # 候选点缓存
        self.candidate_cache: Dict[int, List[Detection]] = defaultdict(list)

    def create_agent(self, gt_track_id: int = -1) -> TrackingAgent:
        """创建新智能体"""
        agent_id = self.next_agent_id
        self.next_agent_id += 1

        agent = TrackingAgent(
            track_id=agent_id,
            gt_track_id=gt_track_id,
            max_history_frames=self.max_history_frames,
            device=self.device
        )

        self.active_agents[agent_id] = agent
        
        # 保存所有创建过的agent
        self.all_agents[agent_id] = agent
        
        return agent

    def get_agent(self, agent_id: int) -> Optional[TrackingAgent]:
        """获取智能体"""
        return self.active_agents.get(agent_id)

    def remove_agent(self, agent_id: int):
        """移除智能体"""
        if agent_id in self.active_agents:
            del self.active_agents[agent_id]

    def update_candidates(self, frame: int, detections: List[Detection]):
        """更新当前帧的候选检测点"""
        self.candidate_cache[frame] = detections

    def get_candidates(self, frame: int) -> List[Detection]:
        """获取指定帧的候选检测点"""
        return self.candidate_cache.get(frame, [])

    def get_all_agents_data(self) -> List[TrackingAgent]:
        """获取所有活跃智能体"""
        return list(self.active_agents.values())
    
    def get_all_created_agents(self) -> List[TrackingAgent]:
        """获取所有创建过的智能体（包括已移除的）"""
        return list(self.all_agents.values())
    
    def clear_frame_candidates(self, frame: int):
        """清理指定帧的候选缓存"""
        # 保留最近几帧
        frames_to_keep = [frame - 1, frame - 2]
        for f in list(self.candidate_cache.keys()):
            if f not in frames_to_keep:
                del self.candidate_cache[f]

    def get_agent_count(self) -> int:
        """获取活跃智能体数量"""
        return len(self.active_agents)

    def reset(self):
        """重置管理器"""
        self.active_agents = {}
        self.all_agents = {}
        self.next_agent_id = 1
        self.candidate_cache.clear()

