"""
PPO learner and trajectory-level tracking helpers.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import logging
import numpy as np
import torch

from config.config import PPOConfig
from core.network import SharedPPOAgent

logger = logging.getLogger(__name__)


@dataclass
class PPOTransition:
    state: Dict[str, torch.Tensor]
    action: torch.Tensor
    reward: float
    value: torch.Tensor
    log_prob: torch.Tensor
    done: bool
    hidden_state: Optional[torch.Tensor] = None


@dataclass
class TrajectoryInfo:
    track_id: int
    gt_track_id: int
    length: int
    smoothness: float
    id_swaps: int
    fragments: int
    completeness: float
    start_time: int
    end_time: int


class PPOLearner:
    """PPO update logic."""

    def __init__(self, agent: SharedPPOAgent, config: PPOConfig, device: torch.device):
        self.agent = agent
        self.config = config
        self.device = device
        self.gamma = config.gamma
        self.lam = config.lam
        self.value_loss_coef = config.value_loss_coef
        self.entropy_coef = config.entropy_coef
        self.clip_epsilon = config.clip_epsilon
        self.max_grad_norm = config.max_grad_norm

    def compute_advantages(
        self, rewards: List[float], values: List[torch.Tensor], dones: List[bool]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = []
        returns = []

        gae = 0.0
        next_value = 0.0
        values_cpu = [v.detach().cpu() for v in values]

        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            value_t = values_cpu[t].item()
            delta = rewards[t] + self.gamma * next_value * mask - value_t
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + value_t)
            next_value = value_t

        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        if advantages_t.numel() > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        return advantages_t, returns_t

    def ppo_update(
        self,
        states: List[Dict[str, torch.Tensor]],
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        del hidden_states

        self.agent.train_mode()

        if not states or states[0] is None or len(states[0]) == 0:
            logger.warning("No valid states for PPO update, skipping")
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "entropy_loss": 0.0,
                "num_updates": 0,
            }

        state_keys = [k for k in states[0].keys() if k != "hidden_state"]
        batch_size = len(states)
        state_batch = {}

        for key in state_keys:
            tensors = [s[key].detach().cpu() for s in states]
            state_batch[key] = torch.stack(tensors)

        batch_size_per_update = max(16, min(self.config.batch_size, max(16, batch_size // 2)))

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_entropy_loss = 0.0
        num_updates = 0

        for _ in range(self.config.update_epochs):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, batch_size_per_update):
                end = start + batch_size_per_update
                batch_indices = indices[start:end]

                batch_state = {k: v[batch_indices].to(self.device) for k, v in state_batch.items()}
                batch_actions = actions[batch_indices].to(self.device)
                batch_old_log_probs = old_log_probs[batch_indices].to(self.device)
                batch_advantages = advantages[batch_indices].to(self.device)
                batch_returns = returns[batch_indices].to(self.device)
                batch_old_values = None
                if old_values is not None:
                    batch_old_values = old_values[batch_indices].to(self.device)

                new_log_prob, value, entropy = self.agent.network.evaluate_actions(
                    batch_state, batch_actions, None
                )

                ratio = torch.exp(new_log_prob - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss_unclipped = (value - batch_returns) ** 2
                if batch_old_values is not None:
                    value_pred_clipped = batch_old_values + (value - batch_old_values).clamp(
                        -self.clip_epsilon, self.clip_epsilon
                    )
                    value_loss_clipped = (value_pred_clipped - batch_returns) ** 2
                else:
                    value_loss_clipped = value_loss_unclipped
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss = -entropy.mean()
                total_loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.agent.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.network.parameters(), self.max_grad_norm)
                self.agent.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_entropy_loss += entropy_loss.item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(1, num_updates),
            "value_loss": total_value_loss / max(1, num_updates),
            "entropy": total_entropy / max(1, num_updates),
            "entropy_loss": total_entropy_loss / max(1, num_updates),
            "num_updates": num_updates,
        }


class TrajectoryTracker:
    """Per-track candidate feature extraction and reward shaping."""

    def __init__(self, config: PPOConfig, device: torch.device):
        self.config = config
        self.device = device
        self.max_distance = config.max_distance
        self.max_velocity_diff = config.max_velocity_diff
        self.max_predict_horizon = config.max_predict_horizon
        self.num_candidates = config.num_candidates

    def get_valid_detections(self, frame_detections: List, include_noise: bool = True) -> List:
        detections = [d for d in frame_detections if not d.is_occluded]
        if include_noise:
            return detections
        return [d for d in detections if d.label == 1]

    def compute_features(self, track, candidates: List, frame_rate: float = 50.0) -> Dict:
        valid_dets = [d for d in track.detections if not d.is_occluded]

        if len(valid_dets) < 2:
            last_pos = (valid_dets[-1].x, valid_dets[-1].z) if valid_dets else (0.0, 0.0)
            last_vel = (0.0, 0.0)
        else:
            d1, d2 = valid_dets[-2], valid_dets[-1]
            dt = d2.frame - d1.frame
            last_pos = (d2.x, d2.z)
            last_vel = ((d2.x - d1.x) / dt, (d2.z - d1.z) / dt) if dt > 0 else (0.0, 0.0)

        pos_scale = 100.0
        vel_scale = 10.0
        dist_scale = 50.0

        candidate_features = []
        for i in range(self.num_candidates):
            if i < len(candidates):
                cand = candidates[i]
                dx = cand.x - last_pos[0]
                dz = cand.z - last_pos[1]
                dist = np.sqrt(dx ** 2 + dz ** 2)

                dvx = (dx / 1.0) - last_vel[0] if frame_rate > 0 else 0.0
                dvz = (dz / 1.0) - last_vel[1] if frame_rate > 0 else 0.0
                dv = np.sqrt(dvx ** 2 + dvz ** 2)

                hist_dir = (
                    np.arctan2(last_vel[1], last_vel[0])
                    if (last_vel[0] != 0 or last_vel[1] != 0)
                    else 0.0
                )
                cand_dir = np.arctan2(dz, dx) if (dx != 0 or dz != 0) else 0.0
                angle_diff = np.sin(cand_dir - hist_dir)
                conf = 1.0 / (1.0 + dist)

                feat = [
                    dx / pos_scale,
                    dz / pos_scale,
                    dv / vel_scale,
                    dist / dist_scale,
                    conf,
                    angle_diff,
                    cand.v / 10.0,
                    cand.x / 500.0,
                ]
            else:
                feat = [0.0] * 8
            candidate_features.append(feat)

        missed = getattr(track, "missed_frames", 0)
        status = [
            min(len(valid_dets) / 100.0, 1.0),
            min(missed / 10.0, 1.0),
            min(missed / 10.0, 1.0),
            1.0 if track.is_active else 0.0,
        ]

        return {
            "candidate_features": np.array(candidate_features, dtype=np.float32),
            "status": np.array(status, dtype=np.float32),
            "last_position": last_pos,
            "last_velocity": last_vel,
        }

    def select_action(
        self, agent: SharedPPOAgent, state_dict: Dict, deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor, Dict]:
        action, log_prob, value, info = agent.get_action(state_dict, deterministic=deterministic)
        return action.item(), log_prob, value, info

    def compute_reward(
        self,
        action: int,
        prev_track,
        selected_candidate,
        candidates: List,
        all_candidates: List,
    ) -> Tuple[float, Dict]:
        del all_candidates

        reward_scale = self.config.reward_scale
        info = {}
        gt_track_id = getattr(prev_track, "gt_track_id", None)
        num_valid_candidates = sum(1 for cand in candidates if getattr(cand, "label", 1) == 1)

        is_correct_association = False
        is_track_switch = False
        is_new_track = False

        if selected_candidate is not None:
            if getattr(selected_candidate, "label", 1) == 0:
                noise_penalty = -5.2
                if confirmed_track := bool(getattr(prev_track, "confirmed", False)):
                    noise_penalty *= 1.45
                return noise_penalty * reward_scale, {
                    "action": "noise_association",
                    "is_correct": False,
                }
            if gt_track_id is not None and gt_track_id > 0:
                if selected_candidate.track_id == gt_track_id:
                    is_correct_association = True
                else:
                    is_track_switch = True
            elif gt_track_id is not None and gt_track_id <= 0:
                is_new_track = True

        reward_gap_penalty = getattr(self.config, "reward_gap_penalty", -0.15)
        reward_unconfirmed_scale = getattr(self.config, "reward_unconfirmed_scale", 0.90)
        reward_confirmed_bonus = getattr(self.config, "reward_confirmed_bonus", 1.2)
        missed_frames = max(0, int(getattr(prev_track, "missed_frames", 0)))
        track_length = len([d for d in prev_track.detections if not d.is_occluded])
        confirmed_track = bool(getattr(prev_track, "confirmed", False))

        if action == len(candidates) or selected_candidate is None:
            none_penalty = -0.22 if confirmed_track else -0.12
            if num_valid_candidates > 0:
                none_penalty *= 1.8 if confirmed_track else 1.5
            if missed_frames > 0:
                none_penalty *= 0.9
            return none_penalty * reward_scale, {"action": "none", "is_correct": False}

        if is_new_track:
            return 0.4 * reward_scale, {"action": "new_track_association", "is_correct": True}

        if is_track_switch:
            switch_penalty = self.config.reward_idswap_penalty
            if confirmed_track:
                switch_penalty *= 1.9
            else:
                switch_penalty *= 1.4
            if track_length >= 5:
                switch_penalty *= 1.5
            return (
                switch_penalty * reward_scale,
                {"action": "track_switch", "is_correct": False},
            )

        if not is_correct_association:
            return -3.2 * reward_scale, {"action": "wrong_association", "is_correct": False}

        base_reward = 3.74 * reward_scale
        valid_dets = [d for d in prev_track.detections if not d.is_occluded]

        if len(valid_dets) >= 2:
            d1, d2 = valid_dets[-2], valid_dets[-1]
            dt_hist = d2.frame - d1.frame
            if dt_hist > 0:
                vx_hist = (d2.x - d1.x) / dt_hist
                vz_hist = (d2.z - d1.z) / dt_hist
            else:
                vx_hist, vz_hist = 0.0, 0.0
        else:
            d2 = valid_dets[-1] if valid_dets else None
            if d2:
                vx_hist, vz_hist = d2.v, 0.0
            else:
                vx_hist, vz_hist = 0.0, 0.0

        d2 = valid_dets[-1] if valid_dets else None
        if d2:
            dt = selected_candidate.frame - d2.frame
            vx_cand = (selected_candidate.x - d2.x) / dt if dt > 0 else 0.0
            vz_cand = (selected_candidate.z - d2.z) / dt if dt > 0 else 0.0
        else:
            vx_cand, vz_cand = selected_candidate.v, 0.0

        dv = np.sqrt((vx_cand - vx_hist) ** 2 + (vz_cand - vz_hist) ** 2)
        dist = (
            np.sqrt((selected_candidate.x - d2.x) ** 2 + (selected_candidate.z - d2.z) ** 2)
            if d2
            else 0.0
        )

        if dv < self.max_velocity_diff:
            smoothness_reward = 0.8 * (1.0 - dv / self.max_velocity_diff) * reward_scale
        else:
            smoothness_reward = 0.0

        if dist < self.max_distance:
            distance_reward = 0.8 * (1.0 - dist / self.max_distance) * reward_scale
        else:
            distance_reward = 0.0

        confirmation_scale = 1.0 if confirmed_track else reward_unconfirmed_scale
        reward = (base_reward + smoothness_reward + distance_reward) * confirmation_scale

        if confirmed_track:
            reward += reward_confirmed_bonus * reward_scale

        if missed_frames > 0:
            gap_penalty = abs(reward_gap_penalty) * min(missed_frames, 2) * reward_scale
            reward -= gap_penalty
        else:
            gap_penalty = 0.0

        info = {
            "action": "associate",
            "is_correct": True,
            "velocity_change": dv,
            "distance": dist,
            "base_reward": base_reward,
            "smoothness_reward": smoothness_reward,
            "distance_reward": distance_reward,
            "confirmation_scale": confirmation_scale,
            "gap_penalty": gap_penalty,
        }
        return reward, info
