"""
评估脚本：评估PPO追踪模型
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy.spatial import cKDTree

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from config.config import PPOConfig, TrackingConfig, LoggingConfig
from core.network import SharedPPOAgent
from core.environment import TrackingEnvironment
from core.agent import Detection

logger = logging.getLogger(__name__)


class PPOMultiObjectTracker:
    """基于PPO的多目标追踪器"""

    def __init__(
            self,
            checkpoint_path: str,
            data_path: str,
            ppo_config: PPOConfig,
            tracking_config: TrackingConfig,
            device: torch.device = None
    ):
        self.data_path = data_path
        self.ppo_config = ppo_config
        self.tracking_config = tracking_config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.agent = SharedPPOAgent(ppo_config, self.device)
        self._load_checkpoint(checkpoint_path)
        self.agent.eval_mode()

        # 初始化环境
        self.env = TrackingEnvironment(
            data_path=data_path,
            ppo_config=ppo_config,
            tracking_config=tracking_config,
            device=self.device,
            eval_mode=True
        )
        self.last_predictions = {
            'all': {},
            'confirmed': {}
        }

    def _estimate_velocity(self, detections: List[Detection]) -> Tuple[float, float]:
        valid = [d for d in detections if not d.is_occluded]
        if len(valid) < 2:
            return 0.0, 0.0
        d1, d2 = valid[-2], valid[-1]
        dt = max(1, d2.frame - d1.frame)
        return (d2.x - d1.x) / dt, (d2.z - d1.z) / dt

    def _merge_fragmented_tracks(
            self,
            predictions: Dict[int, List[Detection]],
            max_frame_gap: int = 3,
            distance_scale: float = 1.1,
            velocity_scale: float = 1.5
    ) -> Dict[int, List[Detection]]:
        if not predictions:
            return {}

        merge_distance = self.ppo_config.max_distance * distance_scale
        merge_velocity_diff = self.ppo_config.max_velocity_diff * velocity_scale

        track_items = []
        for track_id, detections in predictions.items():
            valid = sorted([d for d in detections if not d.is_occluded], key=lambda d: d.frame)
            if not valid:
                continue
            track_items.append({
                'track_id': track_id,
                'detections': valid,
                'start_frame': valid[0].frame,
                'end_frame': valid[-1].frame,
            })

        track_items.sort(key=lambda item: (item['start_frame'], item['track_id']))
        used = set()
        merged_predictions = {}

        for idx, item in enumerate(track_items):
            if idx in used:
                continue

            used.add(idx)
            merged_detections = list(item['detections'])
            merged_track_id = item['track_id']

            while True:
                tail_det = merged_detections[-1]
                vx_tail, vz_tail = self._estimate_velocity(merged_detections)
                best_j = None
                best_score = None

                for j in range(idx + 1, len(track_items)):
                    if j in used:
                        continue
                    candidate = track_items[j]
                    gap = candidate['start_frame'] - tail_det.frame
                    if gap <= 0 or gap > max_frame_gap:
                        continue

                    head_det = candidate['detections'][0]
                    dist = float(np.hypot(head_det.x - tail_det.x, head_det.z - tail_det.z))
                    if dist > merge_distance:
                        continue

                    link_vx = (head_det.x - tail_det.x) / gap
                    link_vz = (head_det.z - tail_det.z) / gap
                    velocity_diff = float(np.hypot(link_vx - vx_tail, link_vz - vz_tail))
                    if velocity_diff > merge_velocity_diff:
                        continue

                    score = dist + 0.5 * velocity_diff + 0.2 * gap
                    if best_score is None or score < best_score:
                        best_score = score
                        best_j = j

                if best_j is None:
                    break

                used.add(best_j)
                merged_detections.extend(track_items[best_j]['detections'])

            merged_predictions[merged_track_id] = merged_detections

        return merged_predictions

    def _merge_all_tracks(self, predictions: Dict[int, List[Detection]], aggressive: bool = False) -> Dict[int, List[Detection]]:
        """Merge for all tracks, optionally with stronger post-processing."""
        first_pass = self._merge_fragmented_tracks(
            predictions,
            max_frame_gap=3,
            distance_scale=1.1,
            velocity_scale=1.5
        )
        if not aggressive:
            return first_pass
        second_pass = self._merge_fragmented_tracks(
            first_pass,
            max_frame_gap=4,
            distance_scale=1.25,
            velocity_scale=1.8
        )
        return second_pass

    def _merge_confirmed_tracks(self, predictions: Dict[int, List[Detection]], aggressive: bool = False) -> Dict[int, List[Detection]]:
        """Apply a slightly stronger merge only to confirmed output tracks.

        This is intentionally evaluation-only post-processing to reduce ID
        fragmentation without perturbing training behaviour.
        """
        first_pass = self._merge_fragmented_tracks(
            predictions,
            max_frame_gap=3,
            distance_scale=1.1,
            velocity_scale=1.5
        )
        second_pass = self._merge_fragmented_tracks(
            first_pass,
            max_frame_gap=4,
            distance_scale=1.2,
            velocity_scale=1.7
        )
        if not aggressive:
            return second_pass
        third_pass = self._merge_fragmented_tracks(
            second_pass,
            max_frame_gap=5,
            distance_scale=1.3,
            velocity_scale=2.0
        )
        return third_pass

    def _load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.network.load_state_dict(checkpoint['agent_state_dict'])
        logger.info(f"Loaded checkpoint from {path}")

    def track(self, save_path: str = None, output_mode: str = 'all', merge_profile: str = 'standard') -> Dict[int, List[Detection]]:
        """
        执行追踪

        Args:
            save_path: 保存预测轨迹的路径

        Returns:
            预测轨迹字典 {track_id: [Detection列表]}
        """
        logger.info("Starting tracking...")
        state = self.env.reset(frame_idx=0)
        logger.info(f"Environment reset, frame={state.frame}, detections={len(state.frame_detections)}")
        
        frame = 0
        done = False

        while not done:
            next_state, reward, done, info = self.env.step(self.agent)
            frame += 1
            if frame % 100 == 0:
                logger.info(f"Processing frame {frame}")

        # 收集预测轨迹 - 获取所有创建过的agent，而不只是活跃的
        predictions = {}
        all_predictions = {}
        all_agents = self.env.agent_manager.get_all_created_agents()
        
        # DEBUG: 打印创建的所有 agent 统计
        for tracking_agent in all_agents:
            detections = [d for d in tracking_agent.track.detections if not d.is_occluded]
            if detections:
                all_predictions[tracking_agent.track_id] = detections

            if not tracking_agent.track.confirmed:
                continue
            if len(detections) >= 2:
                predictions[tracking_agent.track_id] = detections
        
        aggressive_merge = (merge_profile == 'aggressive')
        merged_all_predictions = self._merge_all_tracks(all_predictions, aggressive=aggressive_merge)
        merged_predictions = self._merge_confirmed_tracks(predictions, aggressive=aggressive_merge)

        selected_predictions = merged_predictions if output_mode == 'confirmed' else merged_all_predictions
        logger.info(f"Predicted tracks (all merged): {len(merged_all_predictions)}")
        logger.info(f"Predicted tracks (confirmed merged): {len(merged_predictions)}")
        logger.info(f"Predicted tracks (selected output): {len(selected_predictions)}")
        env_metrics = self.env.get_evaluation_metrics()
        conflict_total = int(env_metrics.get('conflict_total', 0))
        conflict_multi = int(env_metrics.get('conflict_multi', 0))
        guard_applicable = int(env_metrics.get('guard_applicable', 0))
        guard_overrides = int(env_metrics.get('guard_overrides', 0))
        if conflict_total > 0:
            logger.info(
                "[ConflictDiag] total=%d, multi=%d, guard_applicable=%d, guard_overrides=%d, override_rate=%.4f",
                conflict_total,
                conflict_multi,
                guard_applicable,
                guard_overrides,
                guard_overrides / max(1, guard_applicable),
            )
        self.last_predictions = {
            'all': merged_all_predictions,
            'confirmed': merged_predictions
        }

        # 保存结果
        if save_path:
            self._save_predictions(selected_predictions, save_path)

        return selected_predictions

    def _save_predictions(self, predictions: Dict[int, List[Detection]], save_path: str):
        """保存预测轨迹"""
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write('pred_id,frame,x,z,v,pha,w,a,label\n')
            for pred_id, detections in predictions.items():
                for det in detections:
                    f.write(f"{pred_id},{det.frame},{det.x},{det.z},{det.v},{det.pha},{det.w},{det.a},{det.label}\n")

        logger.info(f"Predictions saved to {save_path}")


def evaluate_tracking(
        ground_truth_path: str,
        predictions: Dict[int, List[Detection]],
        threshold: float = 2.0
) -> Dict:
    """
    评估追踪性能

    Args:
        ground_truth_path: 真实标签路径
        predictions: 预测轨迹
        threshold: 匹配阈值

    Returns:
        评估指标
    """
    # 加载真实标签 (列索引: 0=track_id, 1=frame, 2=x, 3=z, 8=label)
    gt_data = pd.read_csv(ground_truth_path, header=None)
    gt_by_frame = defaultdict(list)

    for _, row in gt_data.iterrows():
        if row[8] == 1:  # label = 1 表示有效检测
            frame_id = int(row[0])  # frame
            track_id = int(row[1])  # track_id
            x, z = float(row[2]), float(row[3])
            gt_by_frame[frame_id].append((track_id, x, z))

    # 构建预测索引。保留所有非遮挡预测点，这样由噪声点生成的轨迹会被计为 FP。
    pred_by_frame = defaultdict(list)
    for pred_id, detections in predictions.items():
        for det in detections:
            if not det.is_occluded:
                pred_by_frame[det.frame].append((pred_id, det.x, det.z))

    # 评估
    total_TP, total_FP, total_FN = 0, 0, 0
    id_switches = 0
    last_match_for_gt = {}
    total_distance = 0.0
    matched_count = 0

    frames_all = sorted(set(list(gt_by_frame.keys()) + list(pred_by_frame.keys())))

    for frame in frames_all:
        gt_list = gt_by_frame.get(frame, [])
        pred_list = pred_by_frame.get(frame, [])

        if not gt_list and not pred_list:
            continue

        if gt_list and pred_list:
            gt_positions = np.array([[gt[1], gt[2]] for gt in gt_list])
            pred_positions = np.array([[pd[1], pd[2]] for pd in pred_list])

            tree = cKDTree(pred_positions)
            used_cols = set()

            for i, gt_pos in enumerate(gt_positions):
                dist, j = tree.query(gt_pos, k=1)
                if dist <= threshold and j not in used_cols:
                    used_cols.add(j)
                    total_TP += 1
                    total_distance += dist
                    matched_count += 1

                    gt_id = gt_list[i][0]
                    pred_id = pred_list[j][0]

                    prev = last_match_for_gt.get(gt_id)
                    if prev is not None and prev != pred_id:
                        id_switches += 1
                    last_match_for_gt[gt_id] = pred_id
                else:
                    total_FN += 1

            total_FP += len(pred_list) - len(used_cols)
        else:
            total_FN += len(gt_list)
            total_FP += len(pred_list)

    # 计算指标
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    MOTP = total_distance / matched_count if matched_count > 0 else 0
    MOTA = 1.0 - (total_FN + total_FP + id_switches) / max(1, total_TP + total_FN)
    return {
        'MOTA': MOTA,
        'MOTP': MOTP,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'TP': total_TP,
        'FP': total_FP,
        'FN': total_FN,
        'id_switches': id_switches
    }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='PPO Tracking Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='../simulate/simulate/dataset/test/test_tracks_complex.csv',
                        help='Path to test data')
    parser.add_argument('--ground-truth', type=str, default='../simulate/simulate/dataset/test/test_tracks_complex.csv',
                        help='Path to ground truth labels (for evaluation)')
    parser.add_argument('--output', type=str, default='ppo_predictions.csv',
                        help='Output file for predictions')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    parser.add_argument('--exclude-noise', action='store_true',
                        help='Ignore label=0 detections and reproduce the old association-only setup')
    parser.add_argument('--output-mode', type=str, choices=['all', 'confirmed'], default='all',
                        help='Which merged track set to save and evaluate')
    parser.add_argument('--merge-profile', type=str, choices=['standard', 'aggressive'], default='standard',
                        help='Post-merge strength for reducing fragmented IDs during evaluation')

    args = parser.parse_args()

    # 配置
    ppo_config = PPOConfig()
    tracking_config = TrackingConfig()
    tracking_config.include_noise_detections = not args.exclude_noise

    # 设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建追踪器
    tracker = PPOMultiObjectTracker(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        ppo_config=ppo_config,
        tracking_config=tracking_config,
        device=device
    )

    # 执行追踪
    predictions = tracker.track(
        save_path=args.output,
        output_mode=args.output_mode,
        merge_profile=args.merge_profile
    )

    # 评估
    if args.ground_truth:
        metrics = evaluate_tracking(args.ground_truth, predictions)

        logger.info("=" * 50)
        logger.info("Evaluation Results:")
        logger.info("=" * 50)
        logger.info(f"MOTA: {metrics['MOTA']:.4f}")
        logger.info(f"MOTP: {metrics['MOTP']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1: {metrics['f1']:.4f}")
        logger.info(f"TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}")
        logger.info(f"ID Switches: {metrics['id_switches']}")
        logger.info("=" * 50)


if __name__ == "__main__":
    main()

