"""
PPO追踪器配置文件
"""

from dataclasses import dataclass


@dataclass
class PPOConfig:
    """PPO算法配置"""
    # 神经网络配置（增大以充分利用GPU）
    hidden_dim: int = 512
    gru_hidden_dim: int = 512  # 与 hidden_dim 保持一致
    num_layers: int = 2
    dropout: float = 0.1

    # PPO超参数
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.1  # 增大以防止策略崩溃，维持探索能力 (原0.01导致Entropy降到0.6)
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    batch_size: int = 256  # 增大批量大小

    # 训练配置
    num_episodes: int = 1000
    max_frames_per_episode: int = 5000
    update_interval: int = 500
    eval_interval: int = 10

    # 轨迹追踪配置
    max_distance: float = 15.0  # 最大关联距离 (增大以适应新奖励)
    max_velocity_diff: float = 5.0  # 最大速度差异
    max_skip_frames: int = 5  # 最大允许跳过的帧数
    max_predict_horizon: int = 10  # 最大预测范围
    num_candidates: int = 10  # 候选点数量

    # 奖励配置 (修改后的权重)
    reward_smooth: float = 0.5   # 平滑奖励权重 (已不再使用，主要靠稀疏奖励)
    reward_jump_penalty: float = -2.0  # 大跳变惩罚
    reward_continuation: float = 0.1  # 轨迹延续奖励
    reward_idswap_penalty: float = -8.5  # ID切换惩罚
    reward_fragment_penalty: float = -1.0  # 轨迹碎片化惩罚

    # 奖励归一化系数 (增大以匹配新奖励设计)
    reward_scale: float = 0.1  # 奖励缩放系数 (原奖励 × 0.1)

    # 状态空间维度（用于网络输入）
    position_dim: int = 2  # [x, z]
    velocity_dim: int = 2  # [vx, vz]
    candidate_features_dim: int = 5  # [dx, dz, dv, d, rho] (修复后正确)
    status_dim: int = 4  # [track_length, frames_since_update, missed_frames, is_active]
    max_history_frames: int = 20


@dataclass
class TrackingConfig:
    """追踪环境配置"""
    data_path: str = "../simulate/simulate/dataset/train/train_tracks_complex.csv"
    frame_rate: float = 50.0  # Hz
    v_max: float = 25.0  # 最大速度 mm/s

    # 状态特征维度
    position_dim: int = 2  # [x, z]
    velocity_dim: int = 2  # [vx, vz]
    candidate_features_dim: int = 5  # [dx, dz, dv, d, rho]
    status_dim: int = 4  # [track_length, frames_since_update, missed_frames, is_active]

    # Whether to keep label=0 detections in the tracking candidates.
    # Set to False to reproduce the old "association-only on valid detections" setup.
    include_noise_detections: bool = True

    # 最大历史帧数
    max_history_frames: int = 20

    # 智能体管理
    max_agents_per_frame: int = 3000  # 每帧最大活跃智能体数 (原500不足以覆盖1999条GT轨迹)


@dataclass
class LoggingConfig:
    """日志配置"""
    log_dir: str = "logs"
    save_dir: str = "checkpoints"
    tensorboard: bool = True
    log_interval: int = 10
    save_interval: int = 100


# 单例配置实例
ppo_config = PPOConfig()
tracking_config = TrackingConfig()
logging_config = LoggingConfig()
