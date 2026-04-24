"""
最近邻算法微泡追踪系统 (Nearest Neighbor Microbubble Tracking)

基于轨迹数据的多目标追踪系统，使用最近邻算法进行数据关联。
支持处理消失点和噪声点，用于微泡追踪实验的baseline比较。

数据格式：[frame, trackID, x, z, v, pha, w, a, label]
- label = 1: 真实轨迹点
- label = 0: 噪声点
- 其他字段为-1: 消失点

作者: Assistant
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt
import csv
import math
from scipy.spatial import cKDTree
import logging

# Logger配置
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# 重定向print到logger
def _print_to_logger(*args, **kwargs):
    try:
        msg = " ".join(str(a) for a in args)
        logger.debug(msg)
    except Exception:
        pass

print = _print_to_logger


@dataclass
class Detection:
    """检测结果类"""
    frame: int
    track_id: int  # 原始轨迹ID（用于评估）
    x: float
    z: float
    v: float
    pha: float
    w: float
    a: float
    label: int  # 1=真实点，0=噪声点
    is_occluded: bool = False  # 是否为消失点

    @classmethod
    def from_array(cls, data: np.ndarray) -> 'Detection':
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


class KalmanFilterCA:
    """2D 常加速度卡尔曼滤波器，状态 [x, z, vx, vz, ax, az]"""
    def __init__(self, x: float, z: float, dt: float = 1.0, var_process: float = 0.5, var_measure: float = 1.0):
        self.dt = dt
        self.x = np.array([x, z, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.P = np.diag([1.0, 1.0, 5.0, 5.0, 2.0, 2.0])
        self.Q = np.eye(6) * var_process
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]], dtype=float)
        self.R = np.eye(2) * var_measure

    def predict_state(self, dt: float):
        """返回预测状态位置（不改变内部状态）"""
        px = self.x[0] + self.x[2] * dt + 0.5 * self.x[4] * dt * dt
        pz = self.x[1] + self.x[3] * dt + 0.5 * self.x[5] * dt * dt
        return px, pz

    def predict(self, dt: float = None):
        if dt is None:
            dt = self.dt
        F = np.array([
            [1, 0, dt, 0, 0.5 * dt * dt, 0],
            [0, 1, 0, dt, 0, 0.5 * dt * dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=float)
        self.x = F.dot(self.x)
        self.P = F.dot(self.P).dot(F.T) + self.Q
        return (self.x[0], self.x[1])

    def update(self, meas: Tuple[float, float]):
        z = np.array([meas[0], meas[1]], dtype=float)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        y = z - self.H.dot(self.x)
        self.x = self.x + K.dot(y)
        I = np.eye(6)
        self.P = (I - K.dot(self.H)).dot(self.P)


KalmanFilterCV = KalmanFilterCA


@dataclass
class Track:
    """轨迹类"""
    track_id: int
    detections: List[Detection] = field(default_factory=list)
    is_active: bool = True
    last_update_frame: int = -1
    kf: Optional[KalmanFilterCV] = None
    confirmed: bool = True
    missed_frames: int = 0
    _tracker_confirm_frames: int = 3

    def add_detection(self, detection: Detection):
        """添加检测结果到轨迹"""
        pred_x_before = pred_z_before = None
        if self.kf is None:
            try:
                self.kf = KalmanFilterCA(detection.x, detection.z, dt=1.0)
            except Exception:
                try:
                    self.kf = KalmanFilterCV(detection.x, detection.z, dt=1.0)
                except Exception:
                    self.kf = None
        else:
            if self.last_update_frame >= 0:
                dt = detection.frame - self.last_update_frame
                if dt <= 0:
                    dt = 1.0
            else:
                dt = 1.0
            try:
                self.kf.predict(dt)
                pred_x_before = float(self.kf.x[0])
                pred_z_before = float(self.kf.x[1])
                self.kf.update((detection.x, detection.z))
            except Exception:
                pred_x_before = pred_z_before = None

        self.detections.append(detection)
        self.last_update_frame = detection.frame
        self.is_active = not detection.is_occluded
        self.missed_frames = 0

        if not self.confirmed and hasattr(self, '_tracker_confirm_frames'):
            valid_detections = sum(1 for d in self.detections if not d.is_occluded)
            if valid_detections >= self._tracker_confirm_frames:
                self.confirmed = True

        if pred_x_before is not None and pred_z_before is not None:
            predicted_x, predicted_z = float(pred_x_before), float(pred_z_before)
        elif self.kf is not None:
            predicted_x, predicted_z = float(self.kf.x[0]), float(self.kf.x[1])
        else:
            predicted_x, predicted_z = detection.x, detection.z

        if self.kf is not None:
            corrected_x, corrected_z = float(self.kf.x[0]), float(self.kf.x[1])
        else:
            corrected_x, corrected_z = detection.x, detection.z
        setattr(detection, 'predicted_x', float(predicted_x))
        setattr(detection, 'predicted_z', float(predicted_z))
        setattr(detection, 'corrected_x', float(corrected_x))
        setattr(detection, 'corrected_z', float(corrected_z))

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

    def get_predicted_position(self, target_frame: int) -> Optional[Tuple[float, float]]:
        """
        预测在 target_frame 的位置，使用线性外推（更鲁棒）。
        """
        valid_dets = [d for d in self.detections if not d.is_occluded]
        if len(valid_dets) >= 2:
            d1, d2 = valid_dets[-2], valid_dets[-1]
            if d2.frame > d1.frame:
                dt_hist = d2.frame - d1.frame
                vx = (d2.x - d1.x) / dt_hist
                vz = (d2.z - d1.z) / dt_hist
                dt_future = target_frame - d2.frame
                pred_x = d2.x + vx * dt_future
                pred_z = d2.z + vz * dt_future
                return (pred_x, pred_z)

        if self.kf is not None and self.last_update_frame >= 0:
            dt = target_frame - self.last_update_frame
            try:
                px, pz = self.kf.predict_state(dt)
                return (px, pz)
            except Exception:
                pass

        return self.get_last_position()


class NearestNeighborTracker:
    """基于最近邻算法的多目标追踪器"""

    def __init__(self,
                 max_distance: float = 2.0,  # ????????
                 max_velocity_diff: float = 5.0,  # ???????
                 max_frames_skip: int = 4,  # ?????????
                 v_max: float = 25.0,  # ???? (mm/s)
                 frame_rate: float = 50.0,  # ??
                 confirm_frames: int = 2):  # 帧率

        self.max_distance = max_distance
        self.max_velocity_diff = max_velocity_diff
        self.max_frames_skip = max_frames_skip
        self.v_max = v_max
        self.frame_rate = frame_rate
        # 硬门控半径
        self.hard_gating_radius = max(
            float(self.v_max) / float(self.frame_rate) if self.frame_rate > 0 else self.max_distance,
            self.max_distance * 2.5
        )
        # 轨迹确认和死亡延迟
        self.confirm_frames = confirm_frames
        self.death_delay = 6
        # 预测最大外推步数（帧）
        self.max_predict_horizon = 15
        # 轨迹复活参数
        self.revival_maha_threshold = 4.0
        self.revival_max_frames = min(self.max_predict_horizon, 4)

        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_detections: Dict[int, List[Detection]] = {}
        # 诊断计数
        self.last_nn_matches = 0
        self.last_nn_fallback = 0

    def load_data(self, csv_path: str):
        """加载轨迹数据"""
        logger.info("Loading data from %s...", csv_path)

        frame_dict = defaultdict(list)
        total_count = 0

        with open(csv_path, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    values = line.split(',')
                    if len(values) != 9:
                        logger.warning("Line %d has %d values, expected 9", line_num+1, len(values))
                        continue

                    data = np.array([float(v) for v in values])
                    detection = Detection.from_array(data)
                    frame_dict[detection.frame].append(detection)
                    total_count += 1

                except (ValueError, IndexError) as e:
                    logger.warning("Failed to parse line %d: %s", line_num+1, e)
                    continue

        self.frame_detections = dict(frame_dict)
        logger.info("Loaded %d detections across %d frames", total_count, len(self.frame_detections))

        return self.frame_detections

    def calculate_distance(self, track: Track, detection: Detection) -> float:
        """计算轨迹和检测之间的欧氏距离"""
        if detection.is_occluded or detection.label != 1:
            return float('inf')

        valid_dets = [d for d in track.detections if not d.is_occluded]
        if not valid_dets:
            return float('inf')
        last_det = valid_dets[-1]

        pos_distance = math.hypot(last_det.x - detection.x, last_det.z - detection.z)

        track_vel = track.get_last_velocity()
        if track_vel is not None:
            dt = detection.frame - last_det.frame
            if dt > 0:
                vel_det = ((detection.x - last_det.x) / dt, (detection.z - last_det.z) / dt)
                vel_diff = math.hypot(track_vel[0] - vel_det[0], track_vel[1] - vel_det[1])
                if vel_diff > self.max_velocity_diff:
                    large_penalty = max(self.max_distance * 10.0, vel_diff)
                    return pos_distance + large_penalty

        return pos_distance

    def get_active_tracks(self, current_frame: int) -> List[Track]:
        """获取当前帧活跃的轨迹"""
        active_tracks = []
        for track in self.tracks.values():
            frames_since_update = current_frame - track.last_update_frame
            if frames_since_update <= self.max_predict_horizon:
                if track.get_last_position() is not None:
                    active_tracks.append(track)
        return active_tracks

    def associate_detections_nearest_neighbor(self, tracks: List[Track], detections: List[Detection]) -> Dict[int, int]:
        """
        使用最近邻算法进行数据关联
        返回: {detection_idx: track_id}
        """
        # 过滤有效检测（排除消失点和噪声点）
        valid_detections = [d for d in detections if not d.is_occluded and d.label == 1]

        if not tracks:
            return {}

        if not valid_detections:
            return {}

        # 构建检测位置数组
        det_positions = np.array([[d.x, d.z] for d in valid_detections])
        tree = cKDTree(det_positions)

        current_frame = valid_detections[0].frame

        associations = {}  # detection_idx -> track_id
        used_detections = set()
        used_tracks = set()

        # 第一阶段：预锁定 - 为上一帧更新的轨迹找最近邻
        for i, track in enumerate(tracks):
            if not track.detections:
                continue
            last_det = track.detections[-1]
            if last_det.frame != current_frame - 1:
                continue

            # 使用KD-tree找最近邻
            try:
                dist, nn_idx = tree.query([last_det.x, last_det.z], k=1)
            except Exception:
                continue

            # 保守锁定条件
            if dist <= (0.8 * self.max_distance):
                if nn_idx not in used_detections:
                    associations[nn_idx] = track.track_id
                    used_detections.add(nn_idx)
                    used_tracks.add(i)

        self.last_nn_matches = len(associations)

        # 第二阶段：处理未预锁定的轨迹和检测
        remaining_tracks = [(i, t) for i, t in enumerate(tracks) if i not in used_tracks]
        remaining_dets = [j for j in range(len(valid_detections)) if j not in used_detections]

        if remaining_tracks and remaining_dets:
            # 构建剩余检测的位置数组
            remaining_positions = np.array([[valid_detections[j].x, valid_detections[j].z] for j in remaining_dets])
            remaining_tree = cKDTree(remaining_positions)

            # 为每个剩余轨迹找最近邻
            fallback_matches = []
            for track_idx, track in remaining_tracks:
                pred = track.get_predicted_position(current_frame)
                if pred is None:
                    continue

                # 硬门控过滤
                try:
                    candidate_idxs = remaining_tree.query_ball_point([pred[0], pred[1]], r=self.hard_gating_radius)
                except Exception:
                    candidate_idxs = []

                if not candidate_idxs:
                    continue

                # 找最近邻
                for cand_j in candidate_idxs:
                    det_j = remaining_dets[cand_j]
                    detection = valid_detections[det_j]
                    dist = self.calculate_distance(track, detection)
                    if dist < float('inf') and dist <= (self.max_distance * 2.0):
                        fallback_matches.append((track_idx, det_j, dist))

            # 按距离排序并分配
            fallback_matches.sort(key=lambda x: x[2])
            for track_idx, det_idx, dist in fallback_matches:
                if det_idx not in used_detections and track_idx not in used_tracks:
                    if dist <= (self.max_distance * self.hard_gating_radius / self.max_distance):
                        associations[det_idx] = tracks[track_idx].track_id
                        used_detections.add(det_idx)
                        used_tracks.add(track_idx)

        self.last_nn_fallback = len(associations) - self.last_nn_matches
        return associations

    def track(self, verbose: bool = True, progress_interval: int = 100, include_noise: bool = False) -> Dict[int, Track]:
        """执行追踪"""
        logger.info("Starting tracking with Nearest Neighbor algorithm...")

        sorted_frames = sorted(self.frame_detections.keys())
        total_frames = len(sorted_frames)
        logger.info("Total frames to process: %d", total_frames)
        logger.info("Total detections: %d", sum(len(dets) for dets in self.frame_detections.values()))

        cumulative_stats = {
            'tracks_created': 0,
            'tracks_revived': 0,
            'total_associations': 0,
            'total_detections': 0
        }

        for frame_idx, frame in enumerate(sorted_frames):
            if frame_idx % 10 == 0 and verbose:
                logger.info("Processing frame %d/%d (frame number: %d), total_tracks so far: %d",
                           frame_idx+1, total_frames, frame, len(self.tracks))

            detections = self.frame_detections[frame]
            active_tracks = self.get_active_tracks(frame)

            if verbose:
                logger.debug("Frame %d: %d detections, %d active tracks", frame, len(detections), len(active_tracks))

            valid_count = sum(1 for d in detections if d.label == 1 and not d.is_occluded)
            cumulative_stats['total_detections'] += valid_count

            # 使用最近邻算法进行数据关联
            associations = self.associate_detections_nearest_neighbor(active_tracks, detections)

            # 处理关联结果
            used_detections = set()

            # 更新已关联的轨迹
            for det_idx, track_id in associations.items():
                detection = detections[det_idx]
                track = self.tracks[track_id]
                track.add_detection(detection)
                used_detections.add(det_idx)
                cumulative_stats['total_associations'] += 1

            # 进度报告
            if (frame_idx + 1) % progress_interval == 0 or frame_idx == total_frames - 1:
                confirmed_tracks = sum(1 for t in self.tracks.values() if t.confirmed)
                active_track_count = len(active_tracks)
                total_tracks = len(self.tracks)

                logger.info("\n--- Progress Report: Frame %d/%d (%d) ---", frame_idx+1, total_frames, frame)
                logger.info("Active tracks: %d, Total tracks: %d, Confirmed: %d", active_track_count, total_tracks, confirmed_tracks)
                logger.info("Cumulative: %d detections, %d tracks created, %d revived, %d associations",
                            cumulative_stats['total_detections'], cumulative_stats['tracks_created'],
                            cumulative_stats['tracks_revived'], cumulative_stats['total_associations'])

                if cumulative_stats['total_detections'] > 0:
                    estimated_precision = cumulative_stats['total_associations'] / cumulative_stats['total_detections']
                    logger.info("Estimated tracking efficiency: %.3f (associations/detections)", estimated_precision)
                logger.info("-" * 60)

            # 尝试复活未死亡轨迹
            for det_idx, detection in enumerate(detections):
                if det_idx in used_detections or detection.label != 1:
                    continue
                revived = False
                sleeping_tracks = [t for t in self.tracks.values() if (not t.is_active) and (frame - t.last_update_frame) <= self.death_delay]
                best_t = None
                best_mahal = float('inf')

                for t in sleeping_tracks:
                    frames_since_update = frame - t.last_update_frame
                    if t.last_update_frame < 0 or frames_since_update > self.revival_max_frames:
                        continue
                    if t.kf is None:
                        continue
                    try:
                        px, pz = t.kf.predict_state(frames_since_update)
                        innovation = np.array([detection.x - px, detection.z - pz])
                        S = t.kf.H @ t.kf.P @ t.kf.H.T + t.kf.R
                        if hasattr(t.kf, 'Q') and t.kf.Q is not None:
                            S = S + np.eye(2) * (frames_since_update * float(np.trace(t.kf.Q)))
                        maha = float(innovation.T @ np.linalg.inv(S) @ innovation)
                    except Exception:
                        continue
                    if maha < best_mahal and maha <= self.revival_maha_threshold:
                        best_mahal = maha
                        best_t = t

                if best_t is not None:
                    best_t.add_detection(detection)
                    best_t.is_active = True
                    best_t.missed_frames = 0
                    used_detections.add(det_idx)
                    cumulative_stats['tracks_revived'] += 1
                    revived = True
                    logger.debug("Revived track %d for detection %d (maha=%.3f)", best_t.track_id, det_idx, best_mahal)

            # 为未关联的检测创建新轨迹
            for det_idx, detection in enumerate(detections):
                if det_idx in used_detections:
                    continue
                # 允许噪声点(label=0)直接创建轨迹，30%概率
                if detection.label == 0:
                    if np.random.random() > 0.3:
                        continue

                current_track_id = self.next_track_id
                self.next_track_id += 1

                new_track = Track(current_track_id)
                new_track.confirmed = False
                new_track._tracker_confirm_frames = self.confirm_frames
                new_track.add_detection(detection)
                self.tracks[current_track_id] = new_track
                cumulative_stats['tracks_created'] += 1

            # 处理未关联的活跃轨迹（可能消失）
            for track in active_tracks:
                track_in_associations = any(track.track_id == tid for tid in associations.values())
                if not track_in_associations:
                    frames_since_update = frame - track.last_update_frame
                    if frames_since_update <= self.max_frames_skip:
                        occluded_detection = Detection(
                            frame=frame,
                            track_id=track.track_id,
                            x=-1, z=-1, v=-1, pha=-1, w=-1, a=-1,
                            label=-1,
                            is_occluded=True
                        )
                        track.add_detection(occluded_detection)
                        track.missed_frames += 1
                        if track.missed_frames >= self.death_delay:
                            track.is_active = False

        logger.info("Tracking completed. Created %d tracks.", len(self.tracks))
        return self.tracks

    def evaluate_tracking(self, ground_truth_tracks: Dict[int, List[Detection]] = None, progress_interval: int = 200, include_noise_gt: bool = False) -> Dict[str, float]:
        """评估追踪性能"""
        logger.info("Starting evaluation...")

        logger.info("Performing data consistency checks...")
        total_track_detections = sum(len([d for d in t.detections if not d.is_occluded]) for t in self.tracks.values())
        confirmed_tracks = [t for t in self.tracks.values() if t.confirmed]
        logger.info("Total tracks: %d, Confirmed tracks: %d", len(self.tracks), len(confirmed_tracks))
        logger.info("Total track detections (non-occluded): %d", total_track_detections)

        gt_by_frame = defaultdict(list)
        for frame, dets in self.frame_detections.items():
            for det in dets:
                if not det.is_occluded and (det.label == 1 or (include_noise_gt and det.label == 0)):
                    gt_by_frame[frame].append((det.track_id, det.x, det.z))

        gt_frames = sorted(gt_by_frame.keys())

        logger.info("Building track-frame index for %d confirmed tracks...", len(confirmed_tracks))
        pred_by_frame = defaultdict(list)

        for i, track in enumerate(confirmed_tracks):
            frame_to_det = {}
            for d in track.detections:
                if not d.is_occluded:
                    frame_to_det[d.frame] = d

            for frame, det in frame_to_det.items():
                if frame in gt_by_frame:
                    if det.label != 1:
                        continue
                    px = getattr(det, 'corrected_x', getattr(det, 'predicted_x', det.x))
                    pz = getattr(det, 'corrected_z', getattr(det, 'predicted_z', det.z))
                    pred_by_frame[frame].append((track.track_id, px, pz, det.label))

            if (i + 1) % 10000 == 0:
                logger.info("  Indexed %d/%d tracks...", i + 1, len(confirmed_tracks))

        logger.info("Track-frame index built. GT frames: %d, Pred frames: %d", len(gt_by_frame), len(pred_by_frame))

        total_pred_detections = sum(len(preds) for preds in pred_by_frame.values())
        total_gt_detections = sum(len(gts) for gts in gt_by_frame.values())

        logger.info("Data Validation:")
        logger.info("  GT detections: %d", total_gt_detections)
        logger.info("  Predicted detections: %d", total_pred_detections)

        if total_pred_detections > total_gt_detections * 3:
            logger.warning("Predictions greatly exceed GT! Possible over-generation.")
        elif total_pred_detections < total_gt_detections * 0.1:
            logger.warning("Very few predictions! Possible under-detection.")

        logger.info("  Frames with GT: %d", len(gt_by_frame))
        logger.info("  Frames with predictions: %d", len(pred_by_frame))
        logger.info("  Overlapping frames: %d", len(set(gt_by_frame.keys()) & set(pred_by_frame.keys())))

        def_eval_threshold = 0.8
        match_threshold = def_eval_threshold
        total_TP = 0
        total_FP = 0
        total_FN = 0
        total_distance = 0.0
        id_switches = 0
        last_match_for_gt = {}

        frames_all = sorted(set(list(gt_by_frame.keys()) + list(pred_by_frame.keys())))
        total_eval_frames = len(frames_all)
        logger.info("Evaluating %d frames...", total_eval_frames)

        eval_stats = {'processed_frames': 0, 'frames_with_matches': 0}

        for frame_idx, frame in enumerate(frames_all):
            if frame_idx % 100 == 0:
                logger.info("  Evaluating frame %d/%d...", frame_idx + 1, total_eval_frames)

            gt_list = gt_by_frame.get(frame, [])
            pred_list = pred_by_frame.get(frame, [])

            if not gt_list and not pred_list:
                continue

            eval_stats['processed_frames'] += 1

            noise_preds = [pd for pd in pred_list if len(pd) >= 4 and pd[3] == 0]
            signal_preds = [pd for pd in pred_list if len(pd) < 4 or pd[3] != 0]

            total_FP += len(noise_preds)
            pred_list_for_match = signal_preds

            TP = 0
            matched_pairs = []

            if gt_list and pred_list_for_match:
                eval_stats['frames_with_matches'] += 1

                # 使用最近邻匹配（简化版）
                if len(gt_list) <= 50 and len(pred_list_for_match) <= 50:
                    # 小矩阵：直接最近邻
                    cost = np.zeros((len(gt_list), len(pred_list_for_match)))
                    for i, gt in enumerate(gt_list):
                        for j, pd in enumerate(pred_list_for_match):
                            cost[i, j] = math.hypot(gt[1] - pd[1], gt[2] - pd[2])

                    cost_masked = cost.copy()
                    cost_masked[cost_masked > match_threshold] = 1e9

                    # 最近邻分配
                    used_cols = set()
                    for i in range(cost.shape[0]):
                        j = int(np.argmin(cost[i]))
                        if cost_masked[i, j] < 1e9 and j not in used_cols:
                            used_cols.add(j)
                            matched_pairs.append((i, j))
                            TP += 1
                            total_distance += cost[i, j]
                else:
                    # 大矩阵：KD-tree + 最近邻
                    gt_positions = np.array([[gt[1], gt[2]] for gt in gt_list])
                    pd_positions = np.array([[pd[1], pd[2]] for pd in pred_list_for_match])

                    tree = cKDTree(pd_positions)
                    used_cols = set()

                    for i, gt_pos in enumerate(gt_positions):
                        dist, j = tree.query(gt_pos, k=1)
                        if dist <= match_threshold and j not in used_cols:
                            used_cols.add(j)
                            matched_pairs.append((i, j))
                            TP += 1
                            total_distance += dist

            total_TP += TP
            total_FP += len(pred_list_for_match) - len(matched_pairs)
            total_FN += len(gt_list) - TP

            for r, c in matched_pairs:
                gt_id = gt_list[r][0]
                pred_id = pred_list_for_match[c][0]
                prev = last_match_for_gt.get(gt_id)
                if prev is not None and prev != pred_id:
                    id_switches += 1
                last_match_for_gt[gt_id] = pred_id

            if (frame_idx + 1) % progress_interval == 0:
                current_tp = total_TP
                current_fp = total_FP
                current_fn = total_FN
                current_ids = id_switches

                if current_tp + current_fp > 0:
                    current_precision = current_tp / (current_tp + current_fp)
                    current_recall = current_tp / (current_tp + current_fn) if (current_tp + current_fn) > 0 else 0
                    current_mota = 1.0 - (current_fn + current_fp + current_ids) / float(total_gt_detections) if total_gt_detections > 0 else 0

                    logger.info("Evaluation Progress: Frame %d/%d", frame_idx+1, total_eval_frames)
                    logger.info("  Current: TP=%d, FP=%d, FN=%d, IDSW=%d", current_tp, current_fp, current_fn, current_ids)
                    logger.info("  Current Precision: %.4f, Recall: %.4f, MOTA: %.4f", current_precision, current_recall, current_mota)
                    logger.info("  Frames with matches: %d/%d", eval_stats['frames_with_matches'], eval_stats['processed_frames'])
                    logger.info("-" * 50)

        total_gt_detections = sum(len(v) for v in gt_by_frame.values())
        total_pred_detections = sum(len(v) for v in pred_by_frame.values())

        MOTA = 1.0 - (total_FN + total_FP + id_switches) / float(total_gt_detections) if total_gt_detections > 0 else 0.0
        MOTP = (total_distance / total_TP) if total_TP > 0 else 0.0
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        logger.info("\n" + "="*60)
        logger.info("FINAL TRACKING EVALUATION RESULTS (Nearest Neighbor):")
        logger.info("="*60)
        logger.info("Total GT detections: %d", total_gt_detections)
        logger.info("Total predicted detections: %d", total_pred_detections)
        logger.info("TP: %d, FP: %d, FN: %d, IDSW: %d", total_TP, total_FP, total_FN, id_switches)
        logger.info("MOTA: %.4f, MOTP: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f", MOTA, MOTP, precision, recall, f1)
        logger.info("Evaluation coverage: %d/%d frames processed", eval_stats['processed_frames'], total_eval_frames)
        logger.info("Frames with GT-Pred matches: %d", eval_stats['frames_with_matches'])
        logger.info("="*60)

        # 保存预测轨迹
        try:
            out_path = os.path.join(os.getcwd(), 'nn_predicted_tracks.csv')
            with open(out_path, 'w') as f:
                f.write('pred_id,frame,pred_x,pred_z,corrected_x,corrected_z,label\n')
                for pred_id, track in self.tracks.items():
                    for d in track.detections:
                        px = getattr(d, 'corrected_x', getattr(d, 'predicted_x', d.x))
                        pz = getattr(d, 'corrected_z', getattr(d, 'predicted_z', d.z))
                        cx = getattr(d, 'corrected_x', d.x)
                        cz = getattr(d, 'corrected_z', d.z)
                        f.write(f"{pred_id},{d.frame},{px},{pz},{cx},{cz},{d.label}\n")
            logger.info("Predicted tracks saved to %s", out_path)
        except Exception as e:
            logger.warning("Failed to save predicted tracks: %s", e)

        return {
            'MOTA': MOTA,
            'MOTP': MOTP,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': total_TP,
            'fp': total_FP,
            'fn': total_FN,
            'ids': id_switches,
            'gt_detections': total_gt_detections,
            'pred_detections': total_pred_detections,
        }

    def visualize_tracks(self, save_path: str = None):
        """可视化追踪结果 - 优化版：分层展示、颜色编码、透明度减少重叠"""
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        import matplotlib.patches as mpatches

        # 收集所有轨迹数据
        all_tracks_data = []
        for track_id, track in self.tracks.items():
            valid_dets = [d for d in track.detections if not d.is_occluded]
            if len(valid_dets) >= 2:
                x_coords = [getattr(d, 'corrected_x', getattr(d, 'predicted_x', d.x)) for d in valid_dets]
                z_coords = [getattr(d, 'corrected_z', getattr(d, 'predicted_z', d.z)) for d in valid_dets]
                frames = [d.frame for d in valid_dets]
                track_len = len(valid_dets)
                all_tracks_data.append({
                    'track_id': track_id,
                    'x': x_coords,
                    'z': z_coords,
                    'frames': frames,
                    'length': track_len,
                    'start_x': x_coords[0],
                    'start_z': z_coords[0],
                    'end_x': x_coords[-1],
                    'end_z': z_coords[-1]
                })

        if not all_tracks_data:
            logger.warning("No valid tracks to visualize!")
            return

        # 创建2x2子图布局
        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.2)

        # 统计信息
        track_lengths = [t['length'] for t in all_tracks_data]
        avg_length = np.mean(track_lengths)
        max_length = max(track_lengths)
        min_length = min(track_lengths)

        # ========== 子图1: 轨迹主体视图（按帧号着色，使用透明度）==========
        ax1 = fig.add_subplot(gs[0, 0])
        all_frames = [f for t in all_tracks_data for f in t['frames']]
        frame_min, frame_max = min(all_frames), max(all_frames)
        norm_frame = Normalize(vmin=frame_min, vmax=frame_max)

        for track_data in all_tracks_data:
            frames = track_data['frames']
            x_coords = track_data['x']
            z_coords = track_data['z']

            # 使用帧号作为颜色映射
            colors = plt.cm.viridis(norm_frame(frames))
            # 使用透明度减少重叠
            alpha = min(0.8, 0.3 + 0.5 * (track_data['length'] / max_length))

            for i in range(len(x_coords) - 1):
                ax1.plot([x_coords[i], x_coords[i+1]], [z_coords[i], z_coords[i+1]],
                        color=colors[i], alpha=alpha, linewidth=1.5)

            # 起点和终点标记
            ax1.scatter(x_coords[0], z_coords[0], c='green', s=15, marker='o', alpha=0.8, zorder=5)
            ax1.scatter(x_coords[-1], z_coords[-1], c='red', s=15, marker='x', alpha=0.8, zorder=5)

        sm_frame = ScalarMappable(cmap=plt.cm.viridis, norm=norm_frame)
        sm_frame.set_array([])
        cbar_frame = plt.colorbar(sm_frame, ax=ax1, shrink=0.8)
        cbar_frame.set_label('Frame Number', fontsize=10)

        ax1.set_xlabel('X Position (mm)', fontsize=11)
        ax1.set_ylabel('Z Position (mm)', fontsize=11)
        ax1.set_title('Tracked Trajectories (Color = Frame, G=Start, R=End)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_aspect('equal')

        # ========== 子图2: 轨迹长度热力图 ==========
        ax2 = fig.add_subplot(gs[0, 1])

        # 计算每条轨迹的中心点
        centers_x = [(t['start_x'] + t['end_x']) / 2 for t in all_tracks_data]
        centers_z = [(t['start_z'] + t['end_z']) / 2 for t in all_tracks_data]
        lengths = [t['length'] for t in all_tracks_data]

        scatter = ax2.scatter(centers_x, centers_z, c=lengths, cmap='hot', s=80, alpha=0.7, edgecolors='white', linewidths=0.5)
        cbar_len = plt.colorbar(scatter, ax=ax2, shrink=0.8)
        cbar_len.set_label('Track Length (frames)', fontsize=10)

        ax2.set_xlabel('X Position (mm)', fontsize=11)
        ax2.set_ylabel('Z Position (mm)', fontsize=11)
        ax2.set_title('Track Length Distribution (Hot Colormap)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_aspect('equal')

        # ========== 子图3: 起点-终点流向图 ==========
        ax3 = fig.add_subplot(gs[1, 0])

        # 使用统一颜色，更清晰地展示流向
        for track_data in all_tracks_data:
            ax3.plot([track_data['start_x'], track_data['end_x']],
                    [track_data['start_z'], track_data['end_z']],
                    color='steelblue', alpha=0.4, linewidth=1)

            # 起点终点标记
            ax3.scatter(track_data['start_x'], track_data['start_z'], c='green', s=20, marker='o', alpha=0.6)
            ax3.scatter(track_data['end_x'], track_data['end_z'], c='red', s=20, marker='s', alpha=0.6)

        ax3.set_xlabel('X Position (mm)', fontsize=11)
        ax3.set_ylabel('Z Position (mm)', fontsize=11)
        ax3.set_title('Start-to-End Flow (G=Start, R=End)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_aspect('equal')

        # ========== 子图4: 轨迹长度分布直方图 + 统计信息 ==========
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(track_lengths, bins=30, color='steelblue', alpha=0.7, edgecolor='white')

        # 添加统计线
        ax4.axvline(avg_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_length:.1f}')
        ax4.axvline(np.median(track_lengths), color='orange', linestyle=':', linewidth=2, label=f'Median: {np.median(track_lengths):.1f}')

        ax4.set_xlabel('Track Length (frames)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title(f'Track Length Distribution\nTotal: {len(all_tracks_data)} tracks, Min: {min_length}, Max: {max_length}', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3, axis='y')

        # 总标题
        fig.suptitle('Nearest Neighbor Tracking Results - Comprehensive Visualization',
                    fontsize=14, fontweight='bold', y=0.98)

        # 添加详细统计信息文本框
        stats_text = (
            f"Statistics Summary:\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Total Tracks: {len(all_tracks_data)}\n"
            f"Avg Length: {avg_length:.1f} frames\n"
            f"Median Length: {np.median(track_lengths):.1f} frames\n"
            f"Length Range: [{min_length}, {max_length}]\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Frame Range: [{frame_min}, {frame_max}]"
        )
        fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout(rect=[0, 0.12, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info("Visualization saved to %s", save_path)
        else:
            plt.show()


def main(quick_test: bool = False, include_noise: bool = True, confirm_frames: int = 2):
    """主函数"""
    data_path = "../simulate/simulate/dataset/test/test_tracks_complex.csv"
    max_distance = 2.0
    max_velocity_diff = 3.0
    max_frames_skip = 4

    if quick_test:
        logger.info("="*60)
        logger.info("快速测试模式：只处理前1000帧数据")
        logger.info("="*60)
        max_frames = 1000
        progress_interval_track = 100
        progress_interval_eval = 200
    else:
        logger.info("="*60)
        logger.info("完整测试模式：处理全部数据")
        logger.info("="*60)
        max_frames = None
        progress_interval_track = 100
        progress_interval_eval = 200

    # 创建最近邻追踪器
    tracker = NearestNeighborTracker(
        max_distance=max_distance,
        max_velocity_diff=max_velocity_diff,
        max_frames_skip=max_frames_skip,
        confirm_frames=confirm_frames
    )

    # 加载数据
    tracker.load_data(data_path)

    # 快速测试：限制处理的帧数
    if max_frames is not None:
        original_frames = tracker.frame_detections
        sorted_frames = sorted(original_frames.keys())[:max_frames]
        tracker.frame_detections = {f: original_frames[f] for f in sorted_frames}
        logger.info("限制处理帧数为前 %d 帧，实际处理 %d 帧", max_frames, len(tracker.frame_detections))

    # 执行追踪
    tracks = tracker.track(progress_interval=progress_interval_track, include_noise=True)

    # 评估结果
    logger.info("Starting evaluation...")
    metrics = tracker.evaluate_tracking(progress_interval=progress_interval_eval, include_noise_gt=include_noise)

    # 验证检查
    logger.info("\n" + "="*40)
    logger.info("额外验证检查:")
    logger.info("="*40)

    total_pred_detections = metrics['pred_detections']
    total_gt_detections = metrics['gt_detections']
    ratio = total_pred_detections / total_gt_detections if total_gt_detections > 0 else float('inf')

    logger.info("预测检测总数: %d", total_pred_detections)
    logger.info("GT检测总数: %d", total_gt_detections)

    if ratio > 2.0:
        logger.warning("预测检测数量远超GT检测，可能存在过度生成预测的问题！")
    elif ratio < 0.5:
        logger.warning("预测检测数量远少于GT检测，可能存在漏检问题！")
    else:
        logger.info("预测检测数量与GT检测数量比例合理")

    confirmed_tracks = sum(1 for t in tracks.values() if t.confirmed)
    total_tracks = len(tracks)
    logger.info("确认轨迹数: %d/%d", confirmed_tracks, total_tracks)

    if not quick_test:
        tracker.visualize_tracks("nn_tracking_results.png")

    if quick_test:
        logger.info("\n" + "="*60)
        logger.info("快速测试完成！")
        logger.info("="*60)
    else:
        logger.info("\nNearest Neighbor Tracking Completed!")


if __name__ == "__main__":
    import sys

    quick_test = "--quick" in sys.argv or "-q" in sys.argv
    include_noise = not ("--no-noise" in sys.argv or "--exclude-noise" in sys.argv)
    confirm_frames = 2

    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--confirm-frames' and i + 2 <= len(sys.argv[1:]):
            try:
                confirm_frames = int(sys.argv[i + 2])
            except Exception:
                pass

    if quick_test:
        logger.info("运行快速测试模式...")
    else:
        logger.info("运行完整测试模式...")
        logger.info("提示：使用 --quick 或 -q 参数运行快速测试")

    if include_noise:
        logger.info("包含噪声点(label=0)在追踪和评估中...")
    else:
        logger.info("噪声点(label=0)不参与追踪...")

    main(quick_test=quick_test, include_noise=include_noise, confirm_frames=confirm_frames)
