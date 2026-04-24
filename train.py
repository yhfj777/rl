"""
PPO训练脚本
"""

import os
import sys
import torch
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

from config.config import PPOConfig, TrackingConfig, LoggingConfig
from core.network import SharedPPOAgent
from core.ppo import PPOLearner
from core.environment import TrackingEnvironment

# 创建日志目录
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# 配置日志
log_filename = f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger(__name__)


class PPOTrainer:
    """PPO训练器"""

    def __init__(
            self,
            data_path: str,
            ppo_config: PPOConfig,
            tracking_config: TrackingConfig,
            logging_config: LoggingConfig,
            device: torch.device = None,
            eval_data_path: str = None
    ):
        self.data_path = data_path
        self.eval_data_path = eval_data_path if eval_data_path else data_path
        self.ppo_config = ppo_config
        self.tracking_config = tracking_config
        self.logging_config = logging_config

        # 设备
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # 创建目录
        os.makedirs(logging_config.log_dir, exist_ok=True)
        os.makedirs(logging_config.save_dir, exist_ok=True)

        # 初始化组件
        self.agent = SharedPPOAgent(ppo_config, self.device)
        self.learner = PPOLearner(self.agent, ppo_config, self.device)
        self.env = None  # 将在train中初始化

        # 训练统计
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.associations_per_episode = []

    def train(self, num_episodes: int = None, max_frames: int = None):
        """
        训练PPO智能体

        Args:
            num_episodes: 训练回合数
            max_frames: 每回合最大帧数
        """
        if num_episodes is None:
            num_episodes = self.ppo_config.num_episodes
        if max_frames is None:
            max_frames = self.ppo_config.max_frames_per_episode

        logger.info("=" * 60)
        logger.info("Starting PPO Training")
        logger.info("=" * 60)
        logger.info(f"Episodes: {num_episodes}")
        logger.info(f"Max frames per episode: {max_frames}")
        logger.info(f"Update interval: {self.ppo_config.update_interval}")
        logger.info("=" * 60)

        # 初始化环境
        self.env = TrackingEnvironment(
            data_path=self.data_path,
            ppo_config=self.ppo_config,
            tracking_config=self.tracking_config,
            device=self.device,
            eval_mode=False
        )

        frame_count = 0
        update_count = 0

        for episode in range(num_episodes):
            logger.info(f"\n{'='*40}")
            logger.info(f"Episode {episode + 1}/{num_episodes}")
            logger.info(f"{'='*40}")

            # 重置环境
            state = self.env.reset(frame_idx=0)
            total_reward = 0
            frame = 0
            done = False

            # 收集转换数据
            all_transitions = []

            while not done and frame < max_frames:
                # 环境交互
                next_state, reward, done, info = self.env.step(self.agent)
                total_reward += reward
                frame += 1
                frame_count += 1

                # 收集活跃智能体的转换数据（只收集新数据）
                for agent_id, tracking_agent in self.env.agent_manager.active_agents.items():
                    num_rewards = len(tracking_agent.rewards)
                    if num_rewards > 0:
                        # 计算新数据的起始索引（上次收集后新增的数据）
                        start_idx = getattr(tracking_agent, 'last_collected_idx', 0)
                        for i in range(start_idx, num_rewards):
                            transition = {
                                'agent_id': agent_id,
                                'reward': tracking_agent.rewards[i],
                                'log_prob': tracking_agent.log_probs[i],
                                'value': tracking_agent.values[i],
                                'done': tracking_agent.dones[i],
                                'action': tracking_agent.actions[i] if i < len(tracking_agent.actions) else 0,
                            }
                            # 添加状态数据（如果有）
                            if i < len(tracking_agent.states):
                                transition['state'] = tracking_agent.states[i]
                            all_transitions.append(transition)

                        # 记录已收集的索引
                        tracking_agent.last_collected_idx = num_rewards

                # 定期更新
                if frame_count % self.ppo_config.update_interval == 0 and all_transitions:
                    logger.info(f"  Update at frame {frame_count}")
                    update_info = self._update_policy(all_transitions)
                    update_count += 1

                    # 记录更新信息
                    self.policy_losses.append(update_info.get('policy_loss', 0))
                    self.value_losses.append(update_info.get('value_loss', 0))
                    self.entropies.append(update_info.get('entropy', 0))

                    # 清空已收集的转换
                    all_transitions = []

                    # 保存检查点
                    if update_count % 10 == 0:
                        self.save_checkpoint(f"{self.logging_config.save_dir}/checkpoint_{update_count}.pt")

                # 打印进度
                if frame % 100 == 0:
                    avg_reward_per_frame = total_reward / max(1, frame)
                    avg_reward_per_assoc = total_reward / max(1, self.env.metrics['total_associations'])
                    logger.info(
                        f"  Frame {frame}, Reward: {total_reward:.2f}, "
                        f"Avg/frame: {avg_reward_per_frame:.4f}, "
                        f"Avg/assoc: {avg_reward_per_assoc:.4f}, "
                        f"Agents: {info['agent_count']}"
                    )

            # 回合结束 - 收集所有活跃轨迹的信息
            for agent_id, tracking_agent in self.env.agent_manager.active_agents.items():
                track_info = tracking_agent.get_track_info()
                # 只收集有效轨迹（至少有一个检测点）
                if track_info['length'] > 0:
                    self.env.metrics['track_lengths'].append(track_info['length'])
                    self.env.metrics['smoothness_scores'].append(track_info['smoothness'])

            self.episode_rewards.append(total_reward)
            self.associations_per_episode.append(self.env.metrics['total_associations'])

            logger.info(f"  Episode {episode + 1} finished:")
            logger.info(f"    Total reward: {total_reward:.2f}")
            logger.info(f"    Average reward per frame: {total_reward / max(1, frame):.4f}")
            logger.info(f"    Average reward per association: {total_reward / max(1, self.env.metrics['total_associations']):.4f}")
            logger.info(f"    Total frames: {frame}")
            logger.info(f"    Total associations: {self.env.metrics['total_associations']}")
            logger.info(f"    Total tracks created: {self.env.metrics['total_tracks_created']}")
            logger.info(f"    Total tracks completed: {len(self.env.metrics['track_lengths'])}")
            step_count = max(1, self.env.metrics.get('step_count', 0))
            avg_active_agents = self.env.metrics.get('step_agent_count_sum', 0) / step_count
            avg_unassociated_gt = self.env.metrics.get('step_unassociated_gt_sum', 0) / step_count
            logger.info(f"    Avg active agents: {avg_active_agents:.2f}")
            logger.info(f"    Avg unassociated GT/frame: {avg_unassociated_gt:.2f}")

            # 定期评估
            if (episode + 1) % self.ppo_config.eval_interval == 0:
                self._evaluate(episode + 1)

        # 保存最终模型
        self.save_checkpoint(f"{self.logging_config.save_dir}/final_model.pt")

        # 打印训练统计
        self._print_training_summary()

    def _update_policy(self, transitions: List[Dict]) -> Dict:
        """
        更新策略

        Args:
            transitions: 转换数据列表

        Returns:
            更新信息
        """
        if not transitions:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}

        # 准备数据
        rewards = [t['reward'] for t in transitions]
        values = [t['value'] for t in transitions]
        dones = [t['done'] for t in transitions]

        # 计算优势
        advantages, returns = self.learner.compute_advantages(rewards, values, dones)

        # 准备PPO更新数据 - 从转换中收集状态和动作
        states = [t.get('state', {}) for t in transitions]
        actions = torch.tensor([t.get('action', 0) for t in transitions], dtype=torch.long, device=self.device)
        old_log_probs = torch.stack([t['log_prob'] for t in transitions])
        old_values = torch.stack([t['value'] for t in transitions]).view(-1).to(self.device)

        # 收集隐藏状态
        hidden_states_list = [t.get('hidden_state') for t in transitions if t.get('hidden_state') is not None]
        hidden_states = torch.stack(hidden_states_list) if hidden_states_list else None

        # PPO更新
        update_info = self.learner.ppo_update(
            states, actions, old_log_probs, advantages, returns, old_values=old_values, hidden_states=hidden_states
        )

        logger.info(f"    Policy loss: {update_info['policy_loss']:.4f}")
        logger.info(f"    Value loss: {update_info['value_loss']:.4f}")
        logger.info(f"    Entropy: {update_info['entropy']:.4f}")  # 原始熵（正值）
        logger.info(f"    Entropy loss: {update_info.get('entropy_loss', 0):.4f}")  # 熵损失（负值）

        return update_info

    def _evaluate(self, episode: int):
        """评估当前策略"""
        logger.info(f"\n{'='*40}")
        logger.info(f"Evaluation at Episode {episode}")
        logger.info(f"{'='*40}")

        # 切换到评估模式
        self.agent.eval_mode()

        eval_env = TrackingEnvironment(
            data_path=self.eval_data_path,
            ppo_config=self.ppo_config,
            tracking_config=self.tracking_config,
            device=self.device,
            eval_mode=True
        )

        state = eval_env.reset(frame_idx=0)
        total_reward = 0
        frame = 0
        done = False

        while not done:  # 运行完整episode以正确评估
            next_state, reward, done, info = eval_env.step(self.agent)
            total_reward += reward
            frame += 1

        metrics = eval_env.get_evaluation_metrics()

        logger.info(f"  Evaluation results:")
        logger.info(f"    Total reward: {total_reward:.2f}")
        logger.info(f"    Total associations: {metrics['total_associations']}")
        logger.info(f"    Total tracks created: {metrics['total_tracks_created']}")
        logger.info(f"    Total tracks completed: {metrics['total_tracks_completed']}")
        conversion_rate = (
            metrics['total_tracks_completed'] / max(1, metrics['total_tracks_created'])
        )
        logger.info(f"    Track completion rate: {conversion_rate:.4f}")
        logger.info(f"    Avg track length: {metrics['avg_track_length']:.2f}")
        logger.info(f"    Avg smoothness: {metrics['avg_smoothness']:.4f}")

        # 切回训练模式
        self.agent.train_mode()

    def _print_training_summary(self):
        """打印训练统计摘要"""
        logger.info("\n" + "=" * 60)
        logger.info("Training Summary")
        logger.info("=" * 60)
        logger.info(f"Total episodes: {len(self.episode_rewards)}")
        logger.info(f"Average episode reward: {np.mean(self.episode_rewards):.2f} ± {np.std(self.episode_rewards):.2f}")
        logger.info(f"Max episode reward: {np.max(self.episode_rewards):.2f}")
        logger.info(f"Min episode reward: {np.min(self.episode_rewards):.2f}")
        logger.info(f"Average associations: {np.mean(self.associations_per_episode):.2f}")
        logger.info(f"Average policy loss: {np.mean(self.policy_losses) if self.policy_losses else 0:.4f}")
        logger.info(f"Average value loss: {np.mean(self.value_losses) if self.value_losses else 0:.4f}")
        logger.info("=" * 60)

    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            'agent_state_dict': self.agent.network.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropies': self.entropies,
            'config': self.ppo_config
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.network.load_state_dict(checkpoint['agent_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        logger.info(f"Checkpoint loaded from {path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='PPO Microbubble Tracking Training')
    parser.add_argument('--data', type=str, default='../simulate/simulate/dataset/train/train_tracks_complex.csv',
                        help='Path to training data file')
    parser.add_argument('--eval-data', type=str, default='../simulate/simulate/dataset/test/test_tracks_complex.csv',
                        help='Path to evaluation data file')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of training episodes')
    parser.add_argument('--max-frames', type=int, default=2000,
                        help='Maximum frames per episode')
    parser.add_argument('--update-interval', type=int, default=200,
                        help='Update policy every N frames (reduce if OOM)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for logs')
    parser.add_argument('--exclude-noise', action='store_true',
                        help='Ignore label=0 detections and reproduce the old association-only setup')

    args = parser.parse_args()

    # 配置
    ppo_config = PPOConfig()
    tracking_config = TrackingConfig()
    logging_config = LoggingConfig()
    logging_config.save_dir = args.output_dir
    logging_config.log_dir = args.log_dir
    tracking_config.include_noise_detections = not args.exclude_noise

    # 更新配置
    ppo_config.update_interval = args.update_interval

    # 设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建训练器并训练
    trainer = PPOTrainer(
        data_path=args.data,
        ppo_config=ppo_config,
        tracking_config=tracking_config,
        logging_config=logging_config,
        device=device,
        eval_data_path=args.eval_data
    )

    trainer.train(num_episodes=args.episodes, max_frames=args.max_frames)


if __name__ == "__main__":
    main()

