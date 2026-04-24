"""
神经网络架构：GRU编码器 + 策略头 + 价值头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Dict, List, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class NetworkOutput:
    """网络输出"""
    logits: torch.Tensor  # 动作logits
    value: torch.Tensor  # 状态价值
    hidden_state: torch.Tensor  # GRU隐藏状态


class FeatureEncoder(nn.Module):
    """特征编码器：将原始特征编码到隐藏空间"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrajectoryEncoder(nn.Module):
    """轨迹历史编码器：使用GRU编码历史轨迹信息"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim) 历史轨迹特征
            hidden: (num_layers, batch, hidden_dim) 初始隐藏状态
        Returns:
            output: (batch, seq_len, hidden_dim) GRU输出
            h_n: (num_layers, batch, hidden_dim) 最终隐藏状态
        """
        output, h_n = self.gru(x, hidden)
        output = self.layer_norm(output)
        return output, h_n

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """初始化隐藏状态"""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)


class AttentionModule(nn.Module):
    """注意力机制模块：处理候选点特征"""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (batch, hidden_dim) 查询向量
            key_value: (batch, num_candidates, hidden_dim) 候选点特征
        Returns:
            output: (batch, hidden_dim) 注意力输出
        """
        # 添加序列维度
        if query.dim() == 2:
            query = query.unsqueeze(1)

        attn_output, _ = self.attention(query, key_value, key_value)
        attn_output = attn_output.squeeze(1)

        # 残差连接和前馈网络
        output = self.layer_norm(attn_output + query.squeeze(1))
        output = self.layer_norm(self.feed_forward(output) + output)

        return output


class PPONetwork(nn.Module):
    """PPO策略网络：编码状态，输出动作logits和价值"""

    def __init__(
            self,
            position_dim: int = 2,
            velocity_dim: int = 2,
            candidate_features_dim: int = 5,
            status_dim: int = 4,
            max_history_frames: int = 20,
            hidden_dim: int = 256,
            gru_hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.1,
            num_candidates: int = 10
    ):
        super().__init__()

        self.position_dim = position_dim
        self.velocity_dim = velocity_dim
        self.candidate_features_dim = candidate_features_dim
        self.status_dim = status_dim
        self.max_history_frames = max_history_frames
        self.hidden_dim = hidden_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.num_candidates = num_candidates

        # 历史轨迹特征维度
        history_dim = position_dim + velocity_dim  # 位置 + 速度

        # 候选点特征维度（修改：从5维增加到8维）
        # 8维: [dx, dz, dv, dist, conf, angle_diff, cand_v, cand_x]
        self.candidate_total_dim = 8

        # 编码器
        self.history_encoder = FeatureEncoder(history_dim, gru_hidden_dim)
        self.trajectory_gru = TrajectoryEncoder(gru_hidden_dim, gru_hidden_dim, num_layers, dropout)
        self.candidate_encoder = FeatureEncoder(self.candidate_total_dim, hidden_dim)
        self.status_encoder = FeatureEncoder(status_dim, hidden_dim // 2)

        # 注意力机制
        self.attention = AttentionModule(hidden_dim, num_heads=4)

        # 融合层
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 策略头（输出动作logits）
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_candidates + 1)  # +1 表示"不关联"动作
        )

        # 价值头（输出状态价值）
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.GRU):
            torch.nn.init.orthogonal_(module.weight_ih_l0, gain=np.sqrt(2))
            torch.nn.init.orthogonal_(module.weight_hh_l0, gain=np.sqrt(2))
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(
            self,
            history_positions: torch.Tensor,      # (batch, max_history, 2)
            history_velocities: torch.Tensor,      # (batch, max_history, 2)
            candidate_features: torch.Tensor,      # (batch, num_candidates, candidate_total_dim)
            status_features: torch.Tensor,         # (batch, status_dim)
            hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[NetworkOutput, torch.Tensor]:
        """
        前向传播

        Args:
            history_positions: 历史位置 (batch, max_history, 2)
            history_velocities: 历史速度 (batch, max_history, 2)
            candidate_features: 候选点特征 (batch, num_candidates, candidate_total_dim)
            status_features: 状态特征 (batch, status_dim)
            hidden_state: 初始隐藏状态 (num_layers, batch, hidden_dim)

        Returns:
            network_output: 包含logits, value, hidden_state
            fused_features: 融合后的特征用于价值计算
        """
        batch_size = history_positions.shape[0]
        device = history_positions.device

        # 1. 编码历史轨迹
        history_features = torch.cat([history_positions, history_velocities], dim=-1)
        history_encoded = self.history_encoder(history_features)  # (batch, max_history, gru_hidden)

        # GRU编码
        trajectory_output, hidden_state = self.trajectory_gru(history_encoded, hidden_state)
        # 使用最后时刻的输出作为轨迹表示
        trajectory_repr = trajectory_output[:, -1, :]  # (batch, hidden_dim)

        # 2. 编码候选点
        candidate_encoded = self.candidate_encoder(candidate_features)  # (batch, num_candidates, hidden)

        # 3. 注意力机制融合
        fused_features = self.attention(trajectory_repr, candidate_encoded)  # (batch, hidden)

        # 4. 编码状态
        status_encoded = self.status_encoder(status_features)  # (batch, hidden//2)

        # 5. 融合所有特征
        fused_features = torch.cat([fused_features, status_encoded], dim=-1)
        fused_features = self.fusion_fc(fused_features)  # (batch, hidden)

        # 6. 计算动作logits和价值
        logits = self.policy_head(fused_features)  # (batch, num_candidates + 1)
        value = self.value_head(fused_features).squeeze(-1)  # (batch,)

        output = NetworkOutput(
            logits=logits,
            value=value,
            hidden_state=hidden_state
        )

        return output, fused_features

    def get_action(self, state_dict: Dict, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        根据状态选择动作

        Args:
            state_dict: 状态字典
            deterministic: 是否使用确定性动作（用于评估）

        Returns:
            action: 选择的动作
            log_prob: 动作的log概率
            value: 状态价值
            info: 额外信息
        """
        with torch.no_grad():
            # 添加批次维度
            history_positions = state_dict['history_positions'].unsqueeze(0)
            history_velocities = state_dict['history_velocities'].unsqueeze(0)
            candidate_features = state_dict['candidate_features'].unsqueeze(0)
            status_features = state_dict['status_features'].unsqueeze(0)
            hidden_state = state_dict.get('hidden_state')

            output, features = self(
                history_positions,
                history_velocities,
                candidate_features,
                status_features,
                hidden_state
            )

            probs = F.softmax(output.logits, dim=-1)
            dist = Categorical(probs)

            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            value = output.value

            info = {
                'logits': output.logits,
                'probs': probs,
                'hidden_state': output.hidden_state,
                'entropy': dist.entropy()
            }

        return action, log_prob, value, info

    def evaluate_actions(
            self,
            state_dict: Dict,
            action: torch.Tensor,
            hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定动作的log概率和价值

        Args:
            state_dict: 状态字典
            action: 动作 tensor
            hidden_state: 隐藏状态

        Returns:
            log_prob: 动作log概率
            value: 状态价值
            entropy: 策略熵
        """
        # 如果已经有批次维度则不添加
        if state_dict['history_positions'].dim() == 2:
            # 单步输入，添加批次维度
            history_positions = state_dict['history_positions'].unsqueeze(0)
            history_velocities = state_dict['history_velocities'].unsqueeze(0)
            candidate_features = state_dict['candidate_features'].unsqueeze(0)
            status_features = state_dict['status_features'].unsqueeze(0)
        else:
            # 批次输入，直接使用
            history_positions = state_dict['history_positions']
            history_velocities = state_dict['history_velocities']
            candidate_features = state_dict['candidate_features']
            status_features = state_dict['status_features']

        output, features = self(
            history_positions,
            history_velocities,
            candidate_features,
            status_features,
            hidden_state
        )

        probs = F.softmax(output.logits, dim=-1)
        dist = Categorical(probs)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, output.value, entropy


class SharedPPOAgent:
    """共享策略网络的PPO智能体"""

    def __init__(self, config, device: torch.device = None):
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建网络
        self.network = PPONetwork(
            position_dim=config.position_dim,
            velocity_dim=config.velocity_dim,
            candidate_features_dim=config.candidate_features_dim,
            status_dim=config.status_dim,
            max_history_frames=config.max_history_frames,
            hidden_dim=config.hidden_dim,
            gru_hidden_dim=config.gru_hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_candidates=config.num_candidates
        ).to(self.device)

        # 优化器
        self.optimizer = torch.optim.AdamW([
            {'params': self.network.parameters(), 'lr': config.lr_actor}
        ])

        # 打印网络结构
    def get_init_hidden(self, batch_size: int) -> torch.Tensor:
        """获取初始隐藏状态"""
        return self.network.trajectory_gru.init_hidden(batch_size, self.device)

    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def to(self, device: torch.device):
        """移动到设备"""
        self.device = device
        self.network.to(device)

    def get_action(self, state_dict: Dict, deterministic: bool = False):
        """获取动作"""
        return self.network.get_action(state_dict, deterministic)

    def evaluate_actions(self, state_dict: Dict, action: torch.Tensor):
        """评估动作"""
        return self.network.evaluate_actions(state_dict, action)

    def train_mode(self):
        """训练模式"""
        self.network.train()

    def eval_mode(self):
        """评估模式"""
        self.network.eval()

