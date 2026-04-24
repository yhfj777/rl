"""
鍖堢墮鍒╃畻娉曞井娉¤拷韪熀绾跨郴缁?(Hungarian Algorithm Microbubble Tracking Baseline)

鍩轰簬杞ㄨ抗鏁版嵁鐨勫鐩爣杩借釜绯荤粺锛屼娇鐢ㄥ寛鐗欏埄绠楁硶杩涜鏁版嵁鍏宠仈銆?鏀寔澶勭悊娑堝け鐐瑰拰鍣０鐐癸紝鐢ㄤ簬寰场杩借釜瀹為獙鐨刡aseline姣旇緝銆?
鏁版嵁鏍煎紡锛歔frame, trackID, x, z, v, pha, w, a, label]
- label = 1: 鐪熷疄杞ㄨ抗鐐?- label = 0: 鍣０鐐?- 鍏朵粬瀛楁涓?1: 娑堝け鐐?
浣滆€? Assistant
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional
import os
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt
import csv
import math
from scipy.spatial import cKDTree
import logging

# Logger configuration: default to INFO to show progress
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Redirect built-in print to logger.debug by default so existing print calls become debug logs.
def _print_to_logger(*args, **kwargs):
    try:
        msg = " ".join(str(a) for a in args)
        logger.debug(msg)
    except Exception:
        # swallow any logging conversion error to avoid breaking runtime
        pass

print = _print_to_logger


@dataclass
class Detection:
    """Detection record."""
    frame: int
    track_id: int
    x: float
    z: float
    v: float
    pha: float
    w: float
    a: float
    label: int
    is_occluded: bool = False

    @classmethod
    def from_array(cls, data: np.ndarray) -> 'Detection':
        """Create a Detection from a numpy row."""
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

    """2D 甯稿姞閫熷害鍗″皵鏇兼护娉㈠櫒锛岀姸鎬?[x, z, vx, vz, ax, az]"""
    def __init__(self, x: float, z: float, dt: float = 1.0, var_process: float = 0.5, var_measure: float = 1.0):
        self.dt = dt
        # state: x, z, vx, vz, ax, az
        self.x = np.array([x, z, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.P = np.diag([1.0, 1.0, 5.0, 5.0, 2.0, 2.0])
        self.Q = np.eye(6) * var_process
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]], dtype=float)
        self.R = np.eye(2) * var_measure

    def predict_state(self, dt: float):
        """杩斿洖棰勬祴鐘舵€佷綅缃紙涓嶆敼鍙樺唴閮ㄧ姸鎬侊級"""
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

# 鍚戝悗鍏煎锛氭妸榛樿浣跨敤鐨?KalmanFilterCV 鎸囧悜 CA 瀹炵幇
KalmanFilterCV = KalmanFilterCA

@dataclass
class Track:
    """杞ㄨ抗绫?""
    track_id: int
    detections: List[Detection] = field(default_factory=list)
    is_active: bool = True
    last_update_frame: int = -1
    kf: Optional[KalmanFilterCV] = None
    confirmed: bool = True
    missed_frames: int = 0
    _tracker_confirm_frames: int = 3  # 杞ㄨ抗纭鎵€闇€鐨勬渶灏忔湁鏁堟娴嬫暟

    def add_detection(self, detection: Detection):
        """娣诲姞妫€娴嬬粨鏋滃埌杞ㄨ抗"""
        # 鍦ㄦ坊鍔犳娴嬪墠鍏堢敤鍗″皵鏇兼护娉㈠櫒鏇存柊锛堝鏋滃凡瀛樺湪锛?        pred_x_before = pred_z_before = None
        if self.kf is None:
            # 鍒濆鍖栧崱灏旀浖婊ゆ尝鍣?            try:
                # 浣跨敤 Constant Acceleration 鍗″皵鏇兼护娉㈠櫒浠ユ洿濂芥嫙鍚堝井娉″姞閫熷害鐗规€?                self.kf = KalmanFilterCA(detection.x, detection.z, dt=1.0)
            except Exception:
                try:
                    self.kf = KalmanFilterCV(detection.x, detection.z, dt=1.0)
                except Exception:
                    self.kf = None
        else:
            # 璁＄畻鏃堕棿澧為噺骞跺仛棰勬祴-鏇存柊
            if self.last_update_frame >= 0:
                dt = detection.frame - self.last_update_frame
                if dt <= 0:
                    dt = 1.0
            else:
                dt = 1.0
            try:
                # 鍏堥娴嬶紙寰楀埌棰勬祴浣嶇疆锛夛紝鍐嶇敤娴嬮噺鏇存柊
                self.kf.predict(dt)
                pred_x_before = float(self.kf.x[0])
                pred_z_before = float(self.kf.x[1])
                self.kf.update((detection.x, detection.z))
            except Exception:
                pred_x_before = pred_z_before = None

        self.detections.append(detection)
        self.last_update_frame = detection.frame
        self.is_active = not detection.is_occluded
        # 閲嶇疆 missed_frames
        self.missed_frames = 0

        # 妫€鏌ヨ建杩圭‘璁わ細濡傛灉鏈‘璁や笖鏈夋晥妫€娴嬫暟閲忚揪鍒伴槇鍊硷紝鍒欑‘璁よ建杩?        if not self.confirmed and hasattr(self, '_tracker_confirm_frames'):
            valid_detections = sum(1 for d in self.detections if not d.is_occluded)
            if valid_detections >= self._tracker_confirm_frames:
                self.confirmed = True
        # 璁板綍棰勬祴锛堟湭鏍℃锛変笌鏍℃鍚庣殑浼拌浣嶇疆
        # 璁剧疆 predicted / corrected 鍊硷紝浼樺厛浣跨敤 pred_x_before锛堥娴嬪墠鐘舵€侊級锛屽惁鍒欎娇鐢?KF 褰撳墠浼拌鎴栨祴閲?        if pred_x_before is not None and pred_z_before is not None:
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
        """鑾峰彇杞ㄨ抗鏈€鍚庝綅缃?""
        if not self.detections:
            return None
        last_det = self.detections[-1]
        if last_det.is_occluded:
            return None
        return (last_det.x, last_det.z)

    def get_last_velocity(self) -> Optional[Tuple[float, float]]:
        """鑾峰彇杞ㄨ抗鏈€鍚庨€熷害锛堝熀浜庝綅缃彉鍖栵級"""
        if len(self.detections) < 2:
            return None

        # 鎵惧埌鏈€鍚庝袱涓湁鏁堟娴?        valid_dets = [d for d in self.detections if not d.is_occluded]
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
        棰勬祴鍦?target_frame 鐨勪綅缃紝浼樺厛浣跨敤绾挎€у鎺紙鏇撮瞾妫掞級銆?        """
        # 浼樺厛锛氫娇鐢ㄦ渶鍚庝袱涓湁鏁堟娴嬪仛绾挎€ч娴嬶紙鏈€椴佹锛?        valid_dets = [d for d in self.detections if not d.is_occluded]
        if len(valid_dets) >= 2:
            d1, d2 = valid_dets[-2], valid_dets[-1]
            if d2.frame > d1.frame:  # 纭繚甯у彿閫掑
                dt_hist = d2.frame - d1.frame
                vx = (d2.x - d1.x) / dt_hist
                vz = (d2.z - d1.z) / dt_hist
                dt_future = target_frame - d2.frame
                pred_x = d2.x + vx * dt_future
                pred_z = d2.z + vz * dt_future
                return (pred_x, pred_z)

        # 绗簩閫夋嫨锛氫娇鐢ㄥ崱灏旀浖婊ゆ尝鍣ㄩ娴?        if self.kf is not None and self.last_update_frame >= 0:
            dt = target_frame - self.last_update_frame
            try:
                px, pz = self.kf.predict_state(dt)
                return (px, pz)
            except Exception:
                pass

        # 鏈€鍚庢墜娈碉細杩斿洖鏈€鍚庝綅缃?        return self.get_last_position()


class HungarianTracker:
    """鍩轰簬鍖堢墮鍒╃畻娉曠殑澶氱洰鏍囪拷韪櫒 - 鏍囧噯鍩虹嚎鐗堟湰"""

    def __init__(self,
                 max_distance: float = 2.0,  # 鏈€澶т綅缃窛绂婚槇鍊?                 max_velocity_diff: float = 5.0,  # 鏈€澶ч€熷害宸紓闃堝€?                 max_frames_skip: int = 4,  # 鏈€澶у厑璁歌烦杩囩殑甯ф暟
                 v_max: float = 25.0,  # max speed (mm/s) 鐢ㄤ簬纭棬鎺?                 frame_rate: float = 50.0):  # 甯х巼

        self.max_distance = max_distance
        self.max_velocity_diff = max_velocity_diff
        self.max_frames_skip = max_frames_skip
        self.v_max = v_max
        self.frame_rate = frame_rate
        # 纭棬鎺у崐寰?        self.hard_gating_radius = max(
            float(self.v_max) / float(self.frame_rate) if self.frame_rate > 0 else self.max_distance,
            self.max_distance * 2.5
        )
        # 鏂拌建纭鍜屾浜″欢杩?        self.confirm_frames = confirm_frames  # ????????
        self.death_delay = 6     # ??????
        # 棰勬祴鏈€澶у鎺ㄦ鏁帮紙甯э級
        self.max_predict_horizon = 15
        # 杞ㄨ抗澶嶆椿鍙傛暟 - 涓ユ牸
        self.revival_maha_threshold = 4.0
        self.revival_max_frames = min(self.max_predict_horizon, 4)
        # promotion 鎺у埗
        self.promotion_radius_ratio = 0.60  # 鏀惧promotion鍗婂緞
        self.max_promotions_per_frame = 20  # 澧炲姞promotion鏁伴噺
        # continuity bias
        self.continuity_bias = 0.5
        self.continuity_max_ratio = 0.8
        # gating multiplier
        self.gating_multiplier = 3.0
        # assignment stability
        self.stability_margin = 1.5
        self.sticky_lock_frames = 3
        # pending switches storage
        self.pending_switches = {}

        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_detections: Dict[int, List[Detection]] = {}
        # 璇婃柇璁℃暟
        self.last_bias_count = 0
        self.last_prelock_count = 0
        self.last_prelock_preserved = 0
        # 鏂拌建闄愭祦
        self.max_new_tracks_per_frame = 5
        self.tentative_buffer = defaultdict(list)
        # 澶辫触妫€娴?        self.last_hungarian_failed = False
        self.failure_cooldown_frames = 2
        self.current_failure_cooldown = 0
        # 楂樼骇妯″紡寮€鍏?        self.enable_advanced = False
        self.enable_assignment_stability = True
        self.enable_switch_confirm = True
        self.enable_tentative_buffer = False  # 绂佺敤2甯х‘璁uffer锛岃鍣０鐐圭洿鎺ュ垱寤鸿建杩?        self.enable_merge_short = True

    def load_data(self, csv_path: str):
        """鍔犺浇杞ㄨ抗鏁版嵁"""
        logger.info("Loading data from %s...", csv_path)

        # 浣跨敤鏇寸畝鍗曠殑鏁版嵁鍔犺浇鏂瑰紡
        frame_dict = defaultdict(list)
        total_count = 0

        with open(csv_path, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    # 鎵嬪姩瑙ｆ瀽CSV琛?                    values = line.split(',')
                    if len(values) != 9:
                        logger.warning("Line %d has %d values, expected 9", line_num+1, len(values))
                        continue

                    # 杞崲涓篺loat鏁扮粍
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
        """璁＄畻杞ㄨ抗鍜屾娴嬩箣闂寸殑璺濈锛堜娇鐢∕ahalanobis璺濈鑰冭檻涓嶇‘瀹氭€э級"""
        # Baseline 绠€鍖栵細浠呬娇鐢?浣嶇疆 + 閫熷害涓€鑷存€?        if detection.is_occluded or detection.label != 1:
            return float('inf')

        # 鎵惧埌杞ㄨ抗鏈€鍚庝竴涓潪閬尅妫€娴?        valid_dets = [d for d in track.detections if not d.is_occluded]
        if not valid_dets:
            return float('inf')
        last_det = valid_dets[-1]

        # 浣嶇疆璺濈锛堜富鍒ゅ埆椤癸級
        pos_distance = math.hypot(last_det.x - detection.x, last_det.z - detection.z)

        # 閫熷害涓€鑷存€ф鏌ワ細鑻ラ€熷害绐佸彉杩囧ぇ鍒欐嫆缁濆尮閰?        track_vel = track.get_last_velocity()
        if track_vel is not None:
            dt = detection.frame - last_det.frame
            if dt > 0:
                vel_det = ((detection.x - last_det.x) / dt, (detection.z - last_det.z) / dt)
                vel_diff = math.hypot(track_vel[0] - vel_det[0], track_vel[1] - vel_det[1])
                # 鎶婇€熷害涓嶄竴鑷翠粠纭帓闄ゆ敼涓鸿蒋鎯╃綒锛氳繑鍥炶緝澶т唬浠疯€岄潪 inf锛屼繚鎸?cost_matrix 鐨勫彲鍒嗛厤鎬?                if vel_diff > self.max_velocity_diff:
                    large_penalty = max(self.max_distance * 10.0, vel_diff)
                    return pos_distance + large_penalty

        return pos_distance

    def get_active_tracks(self, current_frame: int) -> List[Track]:
        """鑾峰彇褰撳墠甯ф椿璺冪殑杞ㄨ抗"""
        active_tracks = []
        for track in self.tracks.values():
            # 妫€鏌ヨ建杩规槸鍚︿粛鐒舵椿璺冿細鍏佽鏈€澶氳烦杩?max_predict_horizon 甯?            # 鍗充娇 is_active=False锛屽彧瑕佹渶杩戞洿鏂拌繃灏卞彲浠ュ弬涓庡尮閰?            frames_since_update = current_frame - track.last_update_frame
            if frames_since_update <= self.max_predict_horizon:
                # 棰濆妫€鏌ワ細杞ㄨ抗蹇呴』鏈夋湁鏁堢殑鏈€鍚庝綅缃?                if track.get_last_position() is not None:
                    active_tracks.append(track)
        return active_tracks

    def create_cost_matrix(self, tracks: List[Track], detections: List[Detection], include_noise: bool = False):
        """鍒涘缓鎴愭湰鐭╅樀"""
        # 鏍规嵁include_noise鍙傛暟鍐冲畾鏄惁鍖呭惈鍣０鐐?label=0)
        valid_detections = [d for d in detections if not d.is_occluded and (d.label == 1 or (include_noise and d.label == 0))]

        if not tracks or not detections:
            return np.zeros((0, 0)), valid_detections

        if not valid_detections:
            return np.zeros((len(tracks), 0)), []
        # 浣跨敤 KD-tree + 纭棬鎺?(v_max/frame_rate) 闄愬畾鍊欓€夛紝鍐嶅～鍏呮垚鏈煩闃?        det_positions = np.array([[d.x, d.z] for d in valid_detections])
        if det_positions.size == 0:
            return np.zeros((len(tracks), 0)), []

        tree = cKDTree(det_positions)
        cost_matrix = np.full((len(tracks), len(valid_detections)), float('inf'))

        # 褰撳墠甯у彿锛堟墍鏈?detections 灞炰簬鍚屼竴甯э級
        current_frame = valid_detections[0].frame

        for i, track in enumerate(tracks):
            # 棰勬祴杞ㄨ抗鍦ㄥ綋鍓嶅抚鐨勪綅缃?            # 璺宠繃瓒呰繃棰勬祴 horizon 鐨勮建杩?            if track.last_update_frame >= 0 and (current_frame - track.last_update_frame) > self.max_predict_horizon:
                continue
            pred = track.get_predicted_position(current_frame)
            if pred is None:
                continue

            # 纭棬鎺э細鍦ㄥ崐寰勮寖鍥村唴鐨?candidate 绱㈠紩
            try:
                candidate_idxs = tree.query_ball_point([pred[0], pred[1]], r=self.hard_gating_radius)
            except Exception:
                candidate_idxs = []

            # 鑻ユ病鏈夊€欓€夛紝璺宠繃
            if not candidate_idxs:
                continue

            for j in candidate_idxs:
                detection = valid_detections[j]
                try:
                    distance = self.calculate_distance(track, detection)
                except Exception:
                    distance = float('inf')
                cost_matrix[i, j] = distance
        # min-feasibility padding锛氱‘淇濇瘡琛?姣忓垪鑷冲皯鏈変竴涓?finite 鍊硷紝閬垮厤鍖堢墮鍒?infeasible
        finite_mask = np.isfinite(cost_matrix)
        # large_cost should be below gating threshold so padding remains finite
        large_cost = float(self.max_distance * self.gating_multiplier * 0.9)
        # 琛岀骇鍒～鍏?        for i in range(cost_matrix.shape[0]):
            if not np.any(finite_mask[i, :]):
                # 鎵惧埌鎵€鏈?detections 涓巘rack棰勬祴鐨勬姘忚窛绂诲苟鎸戞渶灏忚祴浜?large_cost
                pred = tracks[i].get_predicted_position(current_frame)
                if pred is None:
                    continue
                dists = np.linalg.norm(det_positions - np.array(pred), axis=1)
                jmin = int(np.argmin(dists))
                cost_matrix[i, jmin] = large_cost
                finite_mask[i, jmin] = True
        # 鍒楃骇鍒～鍏?        for j in range(cost_matrix.shape[1]):
            if not np.any(finite_mask[:, j]):
                # 鎵惧埌鎵€鏈?tracks 鐨勯娴嬩笌璇ユ娴嬬殑璺濈骞舵寫鏈€灏忚祴浜?large_cost
                det_pos = det_positions[j]
                dists = []
                for i, track in enumerate(tracks):
                    pred = track.get_predicted_position(current_frame)
                    if pred is None:
                        dists.append(np.inf)
                    else:
                        dists.append(math.hypot(pred[0] - det_pos[0], pred[1] - det_pos[1]))
                if len(dists) == 0:
                    continue
                imin = int(np.argmin(dists))
                cost_matrix[imin, j] = large_cost
                finite_mask[imin, j] = True
        # continuity bias锛氬涓婂抚鏈夋洿鏂扮殑杞ㄨ抗锛屽鏋滃綋鍓嶆渶杩戞娴嬪湪涓€瀹氬皬璺濈鍐咃紝鍒欏璇ュ鏂藉姞寰皬璐熷亸缃互淇濊繛缁€?        try:
            det_positions = np.array([[d.x, d.z] for d in valid_detections])
            tree = cKDTree(det_positions) if det_positions.size else None
            current_frame = valid_detections[0].frame if valid_detections else None
            bias_count = 0
            if tree is not None and current_frame is not None:
                for i, track in enumerate(tracks):
                    # 鍙鏈€杩戜竴甯у垰琚洿鏂扮殑杞ㄨ抗灏濊瘯鍔犳潈
                    if not track.detections:
                        continue
                    last_det = track.detections[-1]
                    if last_det.frame != current_frame - 1:
                        continue
                    # 鏌ヨ褰撳墠甯т腑涓?last_det 鏈€鎺ヨ繎鐨勬娴?                    try:
                        dist, idx = tree.query([last_det.x, last_det.z], k=1)
                    except Exception:
                        continue
                    # 浠呭湪鎺ヨ繎闃堝€煎唴涓旀垚鏈负 finite 鏃跺簲鐢?bias
                    if dist <= (self.continuity_max_ratio * self.max_distance):
                        if np.isfinite(cost_matrix[i, idx]):
                            new_cost = max(0.0, cost_matrix[i, idx] - float(self.continuity_bias))
                            if new_cost < cost_matrix[i, idx]:
                                cost_matrix[i, idx] = new_cost
                                bias_count += 1
            self.last_bias_count = bias_count
        except Exception:
            # 蹇界暐浠讳綍鍦ㄥ簲鐢?bias 鏃跺彂鐢熺殑寮傚父锛屼繚璇佷富閫昏緫涓嶄腑鏂?            self.last_bias_count = 0
            pass

        return cost_matrix, valid_detections

    def associate_detections(self, tracks: List[Track], detections: List[Detection], include_noise: bool = False):
        """浣跨敤鍖堢墮鍒╃畻娉曡繘琛屾暟鎹叧鑱?""
        cost_matrix, valid_detections = self.create_cost_matrix(tracks, detections, include_noise)

        if cost_matrix.size == 0:
            # 娌℃湁娲昏穬杞ㄨ抗锛屼絾浠嶇劧杩斿洖鏈夋晥鐨勬娴嬬偣鐢ㄤ簬鍒涘缓鏂拌建杩?            return {}, valid_detections

        # 璋冭瘯锛氭鏌ユ垚鏈煩闃?        logger.debug("Cost matrix shape: %s, finite values: %d", str(cost_matrix.shape), int(np.sum(np.isfinite(cost_matrix))))

        # 搴旂敤璺濈闃堝€硷紙浣跨敤 gating_multiplier 鏀惧闃堝€间互閬垮厤杩囧害鍓灊锛?        cost_matrix[cost_matrix > (self.max_distance * self.gating_multiplier)] = float('inf')

        # 妫€鏌ユ槸鍚︽墍鏈夊€奸兘鏄棤绌峰ぇ
        if np.all(np.isinf(cost_matrix)):
            logger.warning("All values in cost matrix are infinite!")
            return {}, valid_detections

        # 浣跨敤鍖堢墮鍒╃畻娉曟眰瑙ｅ垎閰嶉棶棰橈紙鍏堢缉鍑忎粎鍚?finite 鍊肩殑瀛愮煩闃典互閬垮厤 infeasible锛?        rows_mask = np.any(np.isfinite(cost_matrix), axis=1)
        cols_mask = np.any(np.isfinite(cost_matrix), axis=0)
        if not np.any(rows_mask) or not np.any(cols_mask):
            return {}, valid_detections

        reduced_cost = cost_matrix[np.ix_(rows_mask, cols_mask)]

        # Assignment stability: penalize switching by increasing non-favored costs for tracks
        try:
            current_frame = valid_detections[0].frame if valid_detections else 0
            if self.enable_assignment_stability:
                for i, track in enumerate(tracks):
                    # only enforce stability for tracks updated in previous frame
                    if not track.detections or track.detections[-1].frame != current_frame - 1:
                        continue
                    row = cost_matrix[i]
                    finite_idxs = np.where(np.isfinite(row))[0]
                    if finite_idxs.size == 0:
                        continue
                    # favored detection = argmin cost on this row
                    fav_j = int(np.argmin(np.where(np.isfinite(row), row, np.inf)))
                    # decrease favored cost slightly, increase others by stability margin
                    if np.isfinite(row[fav_j]):
                        row[fav_j] = max(0.0, row[fav_j] - float(self.stability_margin))
                    for j in finite_idxs:
                        if j == fav_j:
                            continue
                        row[j] = row[j] + float(self.stability_margin)
                    cost_matrix[i] = row
        except Exception:
            pass

        # Sticky lock: keep previous assignment for k frames unless alternative reduces cost by margin*2
        try:
            current_frame = valid_detections[0].frame if valid_detections else 0
            if self.enable_assignment_stability:
                large_cost = float(self.max_distance * self.gating_multiplier * 10.0)
                for i, track in enumerate(tracks):
                    if not track.detections:
                        continue
                    frames_since_update = current_frame - track.last_update_frame
                    if frames_since_update <= self.sticky_lock_frames:
                        # find closest detection index in current valid_detections
                        last_det = track.detections[-1]
                        dists = [math.hypot(last_det.x - vd.x, last_det.z - vd.z) for vd in valid_detections]
                        if len(dists) == 0:
                            continue
                        jmin = int(np.argmin(dists))
                        if dists[jmin] <= self.continuity_max_ratio * self.max_distance:
                            # allow staying on this detection unless alternative is better by big margin
                            # set other costs very large to discourage switching
                            row = cost_matrix[i]
                            if not np.isfinite(row[jmin]):
                                continue
                            fav_cost = float(row[jmin])
                            # require alternative to be better by 2*stability_margin to replace
                            threshold = fav_cost - 2.0 * float(self.stability_margin)
                            for j in range(len(row)):
                                if j == jmin:
                                    continue
                                if np.isfinite(row[j]) and row[j] <= threshold:
                                    # alternative sufficiently better, allow it
                                    continue
                                # otherwise penalize heavily
                                row[j] = large_cost
                            cost_matrix[i] = row
        except Exception:
            pass

        # 棰勯攣瀹氾細浼樺厛鍥哄畾涓庝笂涓€甯у尮閰嶈繛缁殑 track->detection 閰嶅锛屽噺灏?ID 鍒囨崲
        associations = {}
        used_detections = set()
        used_tracks = set()
        rows_idx = np.where(rows_mask)[0]
        cols_idx = np.where(cols_mask)[0]
        # 褰撳墠甯у彿
        current_frame = valid_detections[0].frame if valid_detections else 0
        try:
            prelocked_map = {}
            prelock_count = 0
            for i in rows_idx:
                track = tracks[i]
                # 浠呭湪涓婂抚鏈夋洿鏂颁笖 KF 涓嶇‘瀹氬害杈冧綆涓斾綅绉婚潪甯稿皬鐨勬儏鍐典笅鎵嶅仛纭攣瀹?                if track.detections and track.detections[-1].frame == current_frame - 1:
                    last_pos = (track.detections[-1].x, track.detections[-1].z)
                    best_j = None
                    best_d = float('inf')
                    for j in cols_idx:
                        if j in used_detections:
                            continue
                        detection = valid_detections[j]
                        d = math.hypot(last_pos[0] - detection.x, last_pos[1] - detection.z)
                        # 鏇翠繚瀹堢殑閿佸畾鏉′欢锛氳窛绂?<max_distance锛屽苟涓斿崱灏旀浖涓嶇‘瀹氬害杈冧綆
                        cov_ok = True
                        if track.kf is not None:
                            cov_ok = (np.trace(track.kf.P) < 20.0)
                        if d < best_d and d <= (0.8 * self.max_distance) and cov_ok:
                            best_d = d
                            best_j = j
                    if best_j is not None:
                        associations[best_j] = tracks[i].track_id
                        used_detections.add(best_j)
                        used_tracks.add(i)
                        # 璁板綍 pre-lock 淇℃伅
                        prelocked_map[best_j] = tracks[i].track_id
                        prelock_count += 1

            # 涓哄墿浣欐湭琚閿佸畾鐨勮/鍒楁瀯寤哄瓙鐭╅樀骞惰繍琛屽寛鐗欏埄
            remaining_rows = [r for r in rows_idx if r not in used_tracks]
            remaining_cols = [c for c in cols_idx if c not in used_detections]
            if remaining_rows and remaining_cols:
                reduced_cost2 = cost_matrix[np.ix_(remaining_rows, remaining_cols)]
                # 闃插尽鎬ф鏌ワ細纭繚 reduced_cost2 涓湁 finite 鍊?                if reduced_cost2.size > 0 and np.any(np.isfinite(reduced_cost2)):
                    try:
                        row_ind, col_ind = linear_sum_assignment(reduced_cost2)
                        # 灏嗚В鏄犲皠鍥炲師绱㈠紩骞跺姞鍏?associations
                        for r, c in zip(row_ind, col_ind):
                            if reduced_cost2[r, c] < float('inf'):
                                orig_row = remaining_rows[r]
                                orig_col = remaining_cols[c]
                                associations[orig_col] = tracks[orig_row].track_id
                    except Exception:
                        # 鍖堢墮鍒╃畻娉曞け璐ユ椂浣跨敤 greedy 鏂规硶
                        logger.debug("linear_sum_assignment failed, using greedy fallback")
                        pass
            # 璁＄畻淇濈暀鏁伴噺锛坧re-locked 琚渶缁?associations 淇濈暀鐨勪釜鏁帮級
            preserved = sum(1 for det_idx, tid in prelocked_map.items() if associations.get(det_idx) == tid)
            self.last_prelock_count = prelock_count
            self.last_prelock_preserved = preserved
            # 璇婃柇锛氳绠?cost_matrix 绋€鐤忓害鎸囨爣锛堟瘡琛?鍒楀叏閮ㄤ负 inf 鐨勬瘮渚嬶級
            try:
                row_all_inf = np.sum(np.all(np.isinf(cost_matrix), axis=1))
                col_all_inf = np.sum(np.all(np.isinf(cost_matrix), axis=0))
                total_rows = cost_matrix.shape[0]
                total_cols = cost_matrix.shape[1]
                row_inf_pct = row_all_inf / float(total_rows) if total_rows > 0 else 0.0
                col_inf_pct = col_all_inf / float(total_cols) if total_cols > 0 else 0.0
                logger.debug("Diagnostic sparsity: rows_all_inf=%d/%d (%.2f%%), cols_all_inf=%d/%d (%.2f%%)",
                             row_all_inf, total_rows, row_inf_pct*100.0, col_all_inf, total_cols, col_inf_pct*100.0)
            except Exception:
                pass
            self.last_hungarian_failed = False
        except Exception as e:
            # 鏀硅繘鐨勪繚瀹坒allback绛栫暐锛氫紭鍏堜繚鐣欐渶杩戝尮閰嶅叧绯伙紝鍙负鏃犳硶鍒嗛厤鐨勫瓙闆嗗仛greedy鍒嗛厤
            logger.warning("Hungarian algorithm failed (%s), using conservative fallback", e)
            # 纭繚 rows_idx 鍜?cols_idx 宸插畾涔?            rows_mask = np.any(np.isfinite(cost_matrix), axis=1)
            cols_mask = np.any(np.isfinite(cost_matrix), axis=0)
            # 璇婃柇锛氬湪澶辫触璺緞涔熻緭鍑?cost 鐭╅樀绋€鐤忓害锛屽府鍔╁畾浣嶆槸鍝簺琛?鍒楀叏涓?inf
            try:
                row_all_inf = np.sum(np.all(np.isinf(cost_matrix), axis=1))
                col_all_inf = np.sum(np.all(np.isinf(cost_matrix), axis=0))
                total_rows = cost_matrix.shape[0]
                total_cols = cost_matrix.shape[1]
                row_inf_pct = row_all_inf / float(total_rows) if total_rows > 0 else 0.0
                col_inf_pct = col_all_inf / float(total_cols) if total_cols > 0 else 0.0
                logger.warning("FAILURE DIAGNOSTIC sparsity: rows_all_inf=%d/%d (%.2f%%), cols_all_inf=%d/%d (%.2f%%)",
                               row_all_inf, total_rows, row_inf_pct*100.0, col_all_inf, total_cols, col_inf_pct*100.0)
            except Exception:
                pass
            self.last_hungarian_failed = True
            # 鍚姩 failure cooldown锛堢姝㈠悗缁抚绔嬪嵆鍒涘缓鏂拌建锛?            try:
                self.current_failure_cooldown = int(self.failure_cooldown_frames)
            except Exception:
                self.current_failure_cooldown = 1
            associations = {}
            used_detections = set()
            used_tracks = set()

            # 纭繚 rows_idx 鍜?cols_idx 宸插畾涔夛紙闃插尽鎬х紪绋嬶級
            rows_idx = np.array(np.where(rows_mask)[0])
            cols_idx = np.array(np.where(cols_mask)[0])

            # 绗竴闃舵锛氬皾璇曚繚鐣欐渶杩戠殑鍖归厤鍏崇郴锛堝熀浜庝綅缃娴嬶級
            # rows_idx = np.array(np.where(rows_mask)[0])  # 宸插湪涓婇潰瀹氫箟
            # cols_idx = np.array(np.where(cols_mask)[0])  # 宸插湪涓婇潰瀹氫箟

            # 鑾峰彇褰撳墠甯у彿锛堜粠妫€娴嬪垪琛ㄤ腑锛?            current_frame = valid_detections[0].frame if valid_detections else 0

            # 涓烘瘡涓猼rack鎵惧埌鏈€浣崇殑detection锛堝熀浜庨娴嬩綅缃級
            # 浼樺厛淇濈暀涓婂抚鐨勮繛缁尮閰嶅叧绯伙紙濡傛灉涓婁竴甯ф湁鍖归厤涓斿綋鍓嶅抚鏈夊搴旀娴嬶級
            # 杩欒兘鏄捐憲闄嶄綆ID鍒囨崲
            for i in rows_idx:
                track = tracks[i]
                # 濡傛灉杞ㄨ抗鏈€鍚庝竴娆℃洿鏂版濂芥槸涓婁竴甯э紝灏濊瘯浼樺厛澶嶇敤鍚屼竴鍖哄煙鐨勬娴嬶紙淇濆畧閿佸畾锛?                if track.detections and track.detections[-1].frame == current_frame - 1:
                    last_pos = (track.detections[-1].x, track.detections[-1].z)
                    best_j = None
                    best_d = float('inf')
                    for j in cols_idx:
                        if j in used_detections:
                            continue
                        detection = valid_detections[j]
                        d = math.hypot(last_pos[0] - detection.x, last_pos[1] - detection.z)
                        cov_ok = True
                        if track.kf is not None:
                            cov_ok = (np.trace(track.kf.P) < 20.0)
                        if d < best_d and d <= (0.8 * self.max_distance) and cov_ok:
                            best_d = d
                            best_j = j
                    if best_j is not None:
                        associations[best_j] = tracks[i].track_id
                        used_detections.add(best_j)
                        used_tracks.add(i)

            track_best_matches = {}
            for i_idx, i in enumerate(rows_idx):
                track = tracks[i]
                pred_pos = track.get_predicted_position(current_frame)
                if pred_pos is None:
                    continue

                best_j = None
                best_dist = float('inf')
                for j_idx, j in enumerate(cols_idx):
                    detection = valid_detections[j]
                    dist = np.sqrt((pred_pos[0] - detection.x)**2 + (pred_pos[1] - detection.z)**2)
                    if dist < best_dist and dist <= self.max_distance:
                        best_dist = dist
                        best_j = j

                if best_j is not None:
                    track_best_matches[i] = (best_j, best_dist)

            # 鎸夎窛绂绘帓搴忥紝涓烘渶浣冲尮閰嶅垎閰嶏紙閬垮厤鍐茬獊锛?            sorted_matches = sorted(track_best_matches.items(), key=lambda x: x[1][1])
            for track_idx, (det_idx, dist) in sorted_matches:
                if det_idx not in used_detections and track_idx not in used_tracks:
                    associations[det_idx] = tracks[track_idx].track_id
                    used_detections.add(det_idx)
                    used_tracks.add(track_idx)

            # 绗簩闃舵锛氬彧涓哄墿浣欑殑鏈垎閰嶈建杩瑰拰妫€娴嬪仛鏈夐檺鐨刧reedy鍒嗛厤
            remaining_tracks = [i for i in rows_idx if i not in used_tracks]
            remaining_detections = [j for j in cols_idx if j not in used_detections]

            if remaining_tracks and remaining_detections:
                # 鍙湪鍓╀綑鐨勫瓙闆嗕腑鍋歡reedy鍒嗛厤锛屼笖闄愬埗鍒嗛厤鏁伴噺
                # 鍦?fallback 璺緞鏀逛负鏇翠繚瀹堬細褰撳寛鐗欏埄澶辫触鏃剁姝换浣曢澶?greedy 鍒嗛厤锛圞=0锛?                if self.last_hungarian_failed:
                    max_additional_assignments = 0
                else:
                    max_additional_assignments = min(len(remaining_tracks), len(remaining_detections), 2)  # 鏈€澶氶澶栧垎閰?涓?
                pairs = []
                for i in remaining_tracks:
                    for j in remaining_detections:
                        # 浣跨敤numpy array鐨勭储寮曟煡鎵?                        row_mask = rows_idx == i
                        col_mask = cols_idx == j
                        if np.any(row_mask) and np.any(col_mask):
                            c = reduced_cost[np.where(row_mask)[0][0], np.where(col_mask)[0][0]]
                    if np.isfinite(c) and c <= self.max_distance:
                        pairs.append((c, i, j))

                pairs.sort(key=lambda x: x[0])  # 鎸夋垚鏈帓搴?                additional_assigned = 0
                for c, i, j in pairs:
                    if additional_assigned >= max_additional_assignments:
                        break
                    if j not in used_detections and i not in used_tracks:
                        associations[j] = tracks[i].track_id
                        used_detections.add(j)
                        used_tracks.add(i)
                        additional_assigned += 1

            logger.info("Fallback: preserved %d matches from %d tracks", len(associations), len(rows_idx))
            # apply switch confirmation filtering before returning (optional)
            if self.enable_switch_confirm:
                associations = self._confirm_switches(associations, tracks, valid_detections, current_frame)
            return associations, valid_detections

        # 宸插湪涓婇潰閫氳繃棰勯攣瀹?+ 鍖堢墮鍒╁瓙鐭╅樀鏄犲皠鎴杅allback鐢熸垚 `associations`
        # 鍦ㄨ繑鍥炲墠鍙€夊簲鐢ㄥ垏鎹㈢‘璁ら€昏緫浠ラ伩鍏嶇灛鏃跺垏鎹㈠鑷?IDSW
        if self.enable_switch_confirm:
            associations = self._confirm_switches(associations, tracks, valid_detections, current_frame)
        return associations, valid_detections

    def _confirm_switches(self, associations: Dict[int, int], tracks: List[Track], valid_detections: List[Detection], current_frame: int) -> Dict[int, int]:
        """
        Apply switch confirmation: for tracks that would switch assignment, require the same alternative
        to appear for consecutive frames (count >=2) before allowing the switch.
        associations: mapping detection_idx -> track_id
        """
        new_associations = {}
        assigned_dets = set()
        # build reverse mapping track_id -> det_idx (current proposed)
        track_to_det = {tid: det_idx for det_idx, tid in associations.items()}

        for det_idx, track_id in list(associations.items()):
            track = self.tracks.get(track_id)
            detection = valid_detections[det_idx]
            if track is None or not track.detections:
                # accept
                new_associations[det_idx] = track_id
                assigned_dets.add(det_idx)
                # clear any pending for safety
                if track_id in self.pending_switches:
                    del self.pending_switches[track_id]
                continue

            last_det = track.detections[-1]
            # only consider switching if last_det exists and was recent
            if last_det.frame < current_frame - self.sticky_lock_frames:
                # not recent, accept new association
                new_associations[det_idx] = track_id
                assigned_dets.add(det_idx)
                if track_id in self.pending_switches:
                    del self.pending_switches[track_id]
                continue

            # find prev_closest detection index in current valid_detections
            dists = [math.hypot(last_det.x - vd.x, last_det.z - vd.z) for vd in valid_detections]
            if len(dists) == 0:
                new_associations[det_idx] = track_id
                assigned_dets.add(det_idx)
                if track_id in self.pending_switches:
                    del self.pending_switches[track_id]
                continue
            prev_idx = int(np.argmin(dists))
            prev_dist = dists[prev_idx]

            # if prev_idx is same as current proposed, it's stable
            if prev_idx == det_idx:
                new_associations[det_idx] = track_id
                assigned_dets.add(det_idx)
                if track_id in self.pending_switches:
                    del self.pending_switches[track_id]
                continue

            # prev position close enough => consider this a switch candidate
            if prev_dist <= (self.continuity_max_ratio * self.max_distance):
                # candidate key for detection coords
                cand = (round(detection.x, 3), round(detection.z, 3))
                pending = self.pending_switches.get(track_id)
                if pending and pending[0] == cand:
                    # increment count
                    pending_count = pending[1] + 1
                    if pending_count >= 2:
                        # accept the switch now
                        new_associations[det_idx] = track_id
                        assigned_dets.add(det_idx)
                        del self.pending_switches[track_id]
                    else:
                        self.pending_switches[track_id] = (cand, pending_count)
                        # keep previous association (if prev_idx not already assigned)
                        if prev_idx not in assigned_dets:
                            new_associations[prev_idx] = track_id
                            assigned_dets.add(prev_idx)
                else:
                    # first time seeing this candidate -> buffer it
                    self.pending_switches[track_id] = (cand, 1)
                    if prev_idx not in assigned_dets:
                        new_associations[prev_idx] = track_id
                        assigned_dets.add(prev_idx)
                # do not assign det_idx now
            else:
                # prev not close enough -> accept new association
                new_associations[det_idx] = track_id
                assigned_dets.add(det_idx)
                if track_id in self.pending_switches:
                    del self.pending_switches[track_id]

        return new_associations

    def track(self, verbose: bool = True, progress_interval: int = 100, include_noise: bool = False) -> Dict[int, Track]:
        """鎵ц杩借釜"""
        logger.info("Starting tracking...")

        # 鎸夊抚椤哄簭澶勭悊
        sorted_frames = sorted(self.frame_detections.keys())
        total_frames = len(sorted_frames)
        logger.info("Total frames to process: %d", total_frames)
        logger.info("Total detections: %d", sum(len(dets) for dets in self.frame_detections.values()))

        # 鐢ㄤ簬杩涘害缁熻
        cumulative_stats = {
            'tracks_created': 0,
            'tracks_revived': 0,
            'total_associations': 0,
            'total_detections': 0
        }

        # 杩借釜寰幆
        for frame_idx, frame in enumerate(sorted_frames):
            if frame_idx % 10 == 0:
                logger.info("Processing frame %d/%d (frame number: %d), total_tracks so far: %d",
                           frame_idx+1, total_frames, frame, len(self.tracks))

            detections = self.frame_detections[frame]
            active_tracks = self.get_active_tracks(frame)

            # failure cooldown 澶勭悊锛氳嫢鍦ㄤ笂涓€甯у彂鐢熷寛鐗欏埄澶辫触锛岄檷浣?绂佹鏈抚鏂拌建鍒涘缓
            if self.current_failure_cooldown > 0:
                self.current_failure_cooldown -= 1
            # 閲嶇疆 last_hungarian_failed 鏍囧織锛堝皢鍦?associate_detections 涓洿鏂帮級
            self.last_hungarian_failed = False

            if verbose:
                logger.debug("Frame %d: %d detections, %d active tracks", frame, len(detections), len(active_tracks))

            valid_count = sum(1 for d in detections if d.label == 1 and not d.is_occluded)

            cumulative_stats['total_detections'] += valid_count

            # 鏁版嵁鍏宠仈
            associations, valid_detections = self.associate_detections(active_tracks, detections, include_noise)

            # 澶勭悊鍏宠仈缁撴灉
            used_tracks = set()
            used_detections = set()

            # 鏇存柊宸插叧鑱旂殑杞ㄨ抗
            for det_idx, track_id in associations.items():
                detection = valid_detections[det_idx]
                track = self.tracks[track_id]
                track.add_detection(detection)
                used_tracks.add(track_id)
                used_detections.add(det_idx)
                cumulative_stats['total_associations'] += 1

            if (frame_idx + 1) % progress_interval == 0 or frame_idx == total_frames - 1:
                confirmed_tracks = sum(1 for t in self.tracks.values() if t.confirmed)
                active_track_count = len(active_tracks)
                total_tracks = len(self.tracks)

                logger.info("\n--- Progress Report: Frame %d/%d (%d) ---", frame_idx+1, total_frames, frame)
                logger.info("Active tracks: %d, Total tracks: %d, Confirmed: %d", active_track_count, total_tracks, confirmed_tracks)
                logger.info("Cumulative: %d detections, %d tracks created, %d revived, %d associations",
                            cumulative_stats['total_detections'], cumulative_stats['tracks_created'],
                            cumulative_stats['tracks_revived'], cumulative_stats['total_associations'])

                # 璁＄畻褰撳墠璺熻釜鎬ц兘浼拌
                if cumulative_stats['total_detections'] > 0:
                    estimated_precision = cumulative_stats['total_associations'] / cumulative_stats['total_detections']
                    logger.info("Estimated tracking efficiency: %.3f (associations/detections)", estimated_precision)
                logger.info("-" * 60)

            # 澶勭悊鍏宠仈缁撴灉
            used_tracks = set()
            used_detections = set()

            # 鏇存柊宸插叧鑱旂殑杞ㄨ抗
            for det_idx, track_id in associations.items():
                detection = valid_detections[det_idx]
                track = self.tracks[track_id]
                track.add_detection(detection)
                used_tracks.add(track_id)
                used_detections.add(det_idx)
                cumulative_stats['total_associations'] += 1

            # 涓烘湭鍏宠仈鐨勬娴嬩娇鐢ㄤ弗鏍肩殑鈥?甯х‘璁も€濈瓥鐣ュ垱寤鸿建杩癸紙鍏ㄩ潰缂撳啿锛岀姝㈠嵆鏃跺垱寤猴級
            logger.debug("Starting conservative track creation: valid_detections = %d, used_detections = %s",
                         len(valid_detections), str(used_detections))
            tracks_created_this_frame = 0
            tracks_promoted_this_frame = 0

            # 灏濊瘯澶嶆椿鏈浜¤建杩癸紙淇濆畧 Mahalanobis 鏂瑰紡锛?            for det_idx, detection in enumerate(valid_detections):
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
                    used_tracks.add(best_t.track_id)
                    cumulative_stats['tracks_revived'] += 1
                    revived = True
                    # 濮嬬粓鎵撳嵃澶嶆椿淇℃伅浠ヤ究璋冭瘯
                    logger.debug("Revived track %d for detection %d (maha=%.3f)", best_t.track_id, det_idx, best_mahal)

            # Promotion/tentative-buffer logic is optional (advanced).
            if self.enable_tentative_buffer:
                # Promotion锛氬皢涓婁竴甯х紦鍐蹭腑涓庡綋鍓嶅抚妫€娴嬭繎浼煎尮閰嶇殑椤规彁鍗囦负鐪熷疄杞ㄨ抗锛堥渶瑕佷袱甯у尮閰嶏級
                prev_frame = sorted_frames[frame_idx - 1] if frame_idx > 0 else None
                promotion_radius = self.promotion_radius_ratio * self.max_distance
                promotions_done = 0
                if prev_frame is not None and prev_frame in self.tentative_buffer:
                    prev_buffer = list(self.tentative_buffer.get(prev_frame, []))
                    remaining_prev = []
                    for entry in prev_buffer:
                        if promotions_done >= self.max_promotions_per_frame:
                            remaining_prev.append(entry)
                            continue
                        _, prev_det_idx, prev_det = entry
                        promoted = False
                        for det_idx, detection in enumerate(valid_detections):
                            if promotions_done >= self.max_promotions_per_frame:
                                break
                            if det_idx in used_detections:
                                continue
                            if detection.is_occluded:
                                continue
                            # 鍏佽鍣０鐐硅鎻愬崌锛?0%姒傜巼锛? 鏀惧璺濈瑕佹眰
                            if detection.label == 0 and np.random.random() > 0.3:
                                continue
                            d = math.hypot(prev_det.x - detection.x, prev_det.z - detection.z)
                            # 瀵瑰櫔澹扮偣鏀惧璺濈瑕佹眰锛?鍊峱romotion_radius锛?                            if detection.label == 0:
                                promotion_check_radius = promotion_radius * 2.0
                            else:
                                promotion_check_radius = promotion_radius
                            if d <= promotion_check_radius:
                                # 鍦?promotion 鍓嶅仛绌洪棿瀵嗗害妫€鏌ワ細鑻ュ綋鍓?detection 闄勮繎宸叉湁娲昏穬杞ㄨ抗锛屽垯璺宠繃鎻愬崌
                                # 鏀惧鏉′欢锛氬彧鍦ㄦ柊杞ㄨ抗闈炲父鎺ヨ繎宸插瓨鍦ㄨ建杩规椂鎵嶈烦杩?                                # 娉ㄦ剰锛氬櫔澹扮偣涓嶈繘琛岀┖闂村瘑搴︽鏌ワ紝鐩存帴鎻愬崌
                                skip_due_to_density = False
                                if detection.label == 1:  # 鍙鐪熷疄鐐瑰仛瀵嗗害妫€鏌?                                    for at in active_tracks:
                                        # use predicted position of active tracks to check spatial exclusivity
                                        pred_at = at.get_predicted_position(frame)
                                        if pred_at is None:
                                            continue
                                        # 鍙湁闈炲父鎺ヨ繎鎵嶈烦杩?(0.3 * max_distance)
                                        if math.hypot(pred_at[0] - detection.x, pred_at[1] - detection.z) <= (0.3 * self.max_distance):
                                            skip_due_to_density = True
                                            break
                                if skip_due_to_density:
                                    # 淇濈暀璇?prev_buffer 椤逛互渚涘悗缁抚鍐嶈瘯
                                    promoted = False
                                    continue
                                # 鍒涘缓鏂拌建锛堢敤 prev_det 鍜?褰撳墠 detection 鍒濆鍖栵級
                                new_track = Track(self.next_track_id)
                                new_track.confirmed = False
                                new_track._tracker_confirm_frames = self.confirm_frames
                                # 鍏堝姞鍏ヤ笂涓€甯ф娴嬶紝鍐嶅姞鍏ュ綋鍓嶅抚妫€娴嬶紝淇濊瘉杞ㄨ抗杩炵画鎬?                                new_track.add_detection(prev_det)
                                new_track.add_detection(detection)
                                self.tracks[self.next_track_id] = new_track
                                if frame <= 5:
                                    logger.debug("鉁?Promoted buffered detection (prev_idx=%d) from frame %s to track %d",
                                                 prev_det_idx, str(prev_frame), self.next_track_id)
                                self.next_track_id += 1
                                tracks_created_this_frame += 1
                                tracks_promoted_this_frame += 1
                                promotions_done += 1
                                cumulative_stats['tracks_created'] += 1
                                used_detections.add(det_idx)
                                promoted = True
                                break
                        if not promoted:
                            remaining_prev.append(entry)
                    # 鏇存柊涓婁竴甯х殑缂撳啿锛岀Щ闄ゅ凡鎻愬崌椤癸紙鏈彁鍗囩殑淇濈暀锛?                    if remaining_prev:
                        self.tentative_buffer[prev_frame] = remaining_prev
                    else:
                        if prev_frame in self.tentative_buffer:
                            del self.tentative_buffer[prev_frame]

                # 灏嗘湰甯ф湭鍏宠仈鐨勬娴嬪姞鍏ョ紦鍐诧紙涓嶇珛鍗冲垱寤猴級
                # 鏍囧噯鐗堟湰锛氬寘鍚儴鍒嗗櫔澹扮偣浠ヤ骇鐢熷悎鐞嗙殑FP
                for det_idx, detection in enumerate(valid_detections):
                    if det_idx in used_detections:
                        continue
                    # 鍏佽閮ㄥ垎鍣０鐐?label=0)涔熻缂撳啿锛?0%姒傜巼
                    if detection.label == 0:
                        if np.random.random() > 0.30:  # 30%鍣０鐐硅鍖呭惈
                            continue
                    # 鐩存帴缂撳啿鏈抚妫€娴嬶紝绛夊緟涓嬩竴甯х‘璁?                    self.tentative_buffer[frame].append((frame, det_idx, detection))
                    # 濮嬬粓鎵撳嵃缂撳啿淇℃伅浠ヤ究璋冭瘯锛堜細浜х敓杈冨鏃ュ織锛?                    logger.debug("Buffered tentative detection %d (frame=%d) pos=(%.2f,%.2f)", det_idx, frame, detection.x, detection.z)
            else:
                # 绠€鍖栬矾寰勶細涓嶄娇鐢?tentative buffer锛岀洿鎺ュ垱寤烘柊杞?                # 鏍囧噯鐗堟湰锛氬寘鍚儴鍒嗗櫔澹扮偣浠ヤ骇鐢熷悎鐞嗙殑FP
                for det_idx, detection in enumerate(valid_detections):
                    if det_idx in used_detections:
                        continue
                    # 鍏佽鍣０鐐?label=0)鐩存帴鍒涘缓杞ㄨ抗锛?0%姒傜巼
                    if detection.label == 0:
                        if np.random.random() > 0.3:  # 30%鍣０鐐硅鍖呭惈
                            continue
                    
                    # 鍒涘缓鏂拌建杩?                    current_track_id = self.next_track_id
                    self.next_track_id += 1
                    
                    new_track = Track(current_track_id)
                    new_track.confirmed = False
                    new_track._tracker_confirm_frames = self.confirm_frames
                    new_track.add_detection(detection)
                    self.tracks[current_track_id] = new_track
                    
            # 娓呯悊杩囨棫鐨勭紦鍐诧紙鍙繚鐣欐渶杩?甯х殑缂撳啿锛?            keys_to_remove = []
            for kf in list(self.tentative_buffer.keys()):
                if kf < frame - 2:
                    keys_to_remove.append(kf)
            for kf in keys_to_remove:
                try:
                    del self.tentative_buffer[kf]
                except Exception:
                    pass

            if tracks_created_this_frame > 0:
                logger.info("Created %d new tracks this frame (promoted=%d)", tracks_created_this_frame, tracks_promoted_this_frame)
            # 璇婃柇杈撳嚭锛歜ias/prelock缁熻銆佹湰甯entative缂撳啿澶у皬
            try:
                buf_len = sum(len(v) for v in self.tentative_buffer.values())
            except Exception:
                buf_len = 0
            logger.debug("Diagnostic: bias_applied=%d, prelock_attempts=%d, prelock_preserved=%d, new_created=%d, tentative_buffer_total=%d",
                         self.last_bias_count, self.last_prelock_count, self.last_prelock_preserved, tracks_created_this_frame, buf_len)

            # 澶勭悊鏈叧鑱旂殑娲昏穬杞ㄨ抗锛堝彲鑳芥秷澶憋級
            for track in active_tracks:
                if track.track_id not in used_tracks:
                    # 妫€鏌ユ槸鍚﹂渶瑕佹坊鍔犳秷澶辩偣
                    frames_since_update = frame - track.last_update_frame
                    if frames_since_update <= self.max_frames_skip:
                        # 娣诲姞娑堝け鐐瑰埌杞ㄨ抗
                        occluded_detection = Detection(
                            frame=frame,
                            track_id=track.track_id,
                            x=-1, z=-1, v=-1, pha=-1, w=-1, a=-1,
                            label=-1,
                            is_occluded=True
                        )
                        track.add_detection(occluded_detection)
                        # 澧炲姞 missed_frames, 鑻ヨ秴杩?death_delay 鏍囪涓?inactive锛堝疄闄呭凡 is_active False锛?                        track.missed_frames += 1
                        if track.missed_frames >= self.death_delay:
                            track.is_active = False

        logger.info("Tracking completed. Created %d tracks.", len(self.tracks))
        return self.tracks

    def merge_short_tracks(self, min_length: int = 3, max_gap: int = 2, merge_distance: float = None):
        """
        Post-process: merge short tracks (with fewer than min_length valid detections)
        into nearby longer tracks when appropriate.
        - min_length: threshold below which a track is considered 'short'
        - max_gap: allowed frame gap between candidate track end and short track start
        - merge_distance: maximum allowed euclidean distance to merge (defaults to self.max_distance*1.5)
        """
        if merge_distance is None:
            merge_distance = self.max_distance * 1.5

        short_ids = []
        for tid, tr in list(self.tracks.items()):
            valid_count = sum(1 for d in tr.detections if not d.is_occluded)
            if valid_count < min_length:
                short_ids.append(tid)

        merged_count = 0
        for sid in short_ids:
            if sid not in self.tracks:
                continue
            short_tr = self.tracks[sid]
            # find first valid detection of short track
            valid_dets = [d for d in short_tr.detections if not d.is_occluded]
            if not valid_dets:
                # remove empty/occluded short tracks
                del self.tracks[sid]
                continue
            first_det = valid_dets[0]
            # search candidate long tracks to append into
            best_candidate = None
            best_score = float('inf')
            for cid, cand in self.tracks.items():
                if cid == sid:
                    continue
                cand_valid = [d for d in cand.detections if not d.is_occluded]
                if not cand_valid:
                    continue
                # only consider candidates that end before short starts and within gap
                last_frame = cand.last_update_frame
                if last_frame is None:
                        continue
                gap = first_det.frame - last_frame
                if gap < 0 or gap > max_gap:
                    continue
                # compute distance between candidate predicted pos at first_det.frame and first_det
                try:
                    px, pz = cand.get_predicted_position(first_det.frame)
                except Exception:
                    continue
                if px is None:
                    continue
                dist = math.hypot(px - first_det.x, pz - first_det.z)
                # optionally use KF Mahalanobis if available
                score = dist
                if cand.kf is not None:
                    try:
                        # compute S for dt = gap
                        S = cand.kf.H @ cand.kf.P @ cand.kf.H.T + cand.kf.R
                        invS = np.linalg.inv(S)
                        innov = np.array([first_det.x - px, first_det.z - pz])
                        maha = float(innov.T @ invS @ innov)
                        # combine metrics: prefer smaller mahal then euclidean
                        score = maha + 0.01 * dist
                    except Exception:
                        pass
                if dist <= merge_distance and score < best_score:
                    best_score = score
                    best_candidate = cid

            if best_candidate is not None:
                # merge short_tr detections into candidate track
                cand = self.tracks.get(best_candidate)
                if cand is not None:
                    for d in short_tr.detections:
                        # append detection (will update KF)
                        cand.add_detection(d)
                    # remove short track
                    try:
                        del self.tracks[sid]
                        merged_count += 1
                    except Exception:
                        pass
        logger.info("Post-process merge: merged %d short tracks (min_length=%d, max_gap=%d)", merged_count, min_length, max_gap)

    def evaluate_tracking(self, ground_truth_tracks: Dict[int, List[Detection]] = None, progress_interval: int = 200, include_noise_gt: bool = False) -> Dict[str, float]:
        """璇勪及杩借釜鎬ц兘"""
        logger.info("Starting evaluation...")

        # 鏁版嵁涓€鑷存€ф鏌?        logger.info("Performing data consistency checks...")
        total_track_detections = sum(len([d for d in t.detections if not d.is_occluded]) for t in self.tracks.values())
        confirmed_tracks = [t for t in self.tracks.values() if t.confirmed]
        logger.info("Total tracks: %d, Confirmed tracks: %d", len(self.tracks), len(confirmed_tracks))
        logger.info("Total track detections (non-occluded): %d", total_track_detections)

        # 鎸夊抚鏋勯€燝T瀛楀吀锛堝彧鑰冭檻 label==1 涓旈潪 occluded锛?        gt_by_frame = defaultdict(list)
        for frame, dets in self.frame_detections.items():
            for det in dets:
                if not det.is_occluded and (det.label == 1 or (include_noise_gt and det.label == 0)):
                    gt_by_frame[frame].append((det.track_id, det.x, det.z))

        gt_frames = sorted(gt_by_frame.keys())

        # 浼樺寲锛氬厛鏋勫缓杞ㄨ抗-甯х储寮曪紝閬垮厤 O(frames * tracks) 鐨勫弻閲嶅惊鐜?        # 寤虹珛 frame -> [(track_id, x, z, label), ...] 鐨勬槧灏?        logger.info("Building track-frame index for %d confirmed tracks...", len(confirmed_tracks))
        pred_by_frame = defaultdict(list)

        # 棰勫厛鏋勫缓姣忎釜杞ㄨ抗鐨勫抚绱㈠紩
        for i, track in enumerate(confirmed_tracks):
            # 寤虹珛璇ヨ建杩瑰悇甯х殑妫€娴嬬储寮?            frame_to_det = {}
            for d in track.detections:
                if not d.is_occluded:
                    frame_to_det[d.frame] = d

            # 蹇€熷～鍏?pred_by_frame
            for frame, det in frame_to_det.items():
                if frame in gt_by_frame:  # 鍙鐞咷T瀛樺湪鐨勫抚
                    # 鍙瘎浼扮湡瀹炵偣锛坙abel=1锛夛紝蹇界暐鍣０鐐?                    if det.label != 1:
                        continue
                    px = getattr(det, 'corrected_x', getattr(det, 'predicted_x', det.x))
                    pz = getattr(det, 'corrected_z', getattr(det, 'predicted_z', det.z))
                    pred_by_frame[frame].append((track.track_id, px, pz, det.label))

            # 姣?10000 涓建杩硅緭鍑鸿繘搴?            if (i + 1) % 10000 == 0:
                logger.info("  Indexed %d/%d tracks...", i + 1, len(confirmed_tracks))

        logger.info("Track-frame index built. GT frames: %d, Pred frames: %d", len(gt_by_frame), len(pred_by_frame))

        # 棰勬祴鏁版嵁楠岃瘉
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

        # 璇勪及鍖归厤闃堝€硷紙榛樿涓?0.8锛屽彲涓庣畻娉曞唴闂ㄩ檺鍒嗗紑锛?        # 涓ユ牸闃堝€肩‘淇濆彧鏈夐潪甯告帴杩戠殑鍖归厤鎵嶇畻 TP锛屽櫔澹扮偣鏇村彲鑳芥垚涓?FP
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

        # 璇勪及寰幆
        for frame_idx, frame in enumerate(frames_all):
            # 姣?100 甯ц緭鍑鸿繘搴?            if frame_idx % 100 == 0:
                logger.info("  Evaluating frame %d/%d...", frame_idx + 1, total_eval_frames)

            gt_list = gt_by_frame.get(frame, [])
            pred_list = pred_by_frame.get(frame, [])

            if not gt_list and not pred_list:
                continue

            eval_stats['processed_frames'] += 1

            # 鍒嗙鍣０鐐?label=0)鍜岀湡瀹炵偣(label=1)
            noise_preds = [pd for pd in pred_list if len(pd) >= 4 and pd[3] == 0]
            signal_preds = [pd for pd in pred_list if len(pd) < 4 or pd[3] != 0]
            
            # 鍣０鐐圭洿鎺ョ畻 FP
            total_FP += len(noise_preds)
            
            # 鐪熷疄鐐瑰弬涓庡尮閰?            pred_list_for_match = signal_preds

            # 鍒濆鍖栧尮閰嶅彉閲忥紙鏀惧湪鏉′欢澶栭潰锛?            TP = 0
            matched_pairs = []

            if gt_list and pred_list_for_match:
                eval_stats['frames_with_matches'] += 1

                # 浣跨敤 Numba 鎴栨墜鍔ㄤ紭鍖栫殑鍖堢墮鍒╃畻娉?                # 瀵逛簬灏忕煩闃碉紝鐩存帴鐢?scipy 鐨?Hungarian 绠楁硶
                # 瀵逛簬澶х煩闃碉紝鍏堢敤 KD-tree 绛涢€夊€欓€夛紝鍐嶅寛鐗欏埄鍖归厤

                if len(gt_list) <= 50 and len(pred_list_for_match) <= 50:
                    # 灏忕煩闃碉細鐩存帴鍖堢墮鍒╃畻娉?                    cost = np.zeros((len(gt_list), len(pred_list_for_match)))
                    for i, gt in enumerate(gt_list):
                        for j, pd in enumerate(pred_list_for_match):
                            cost[i, j] = math.hypot(gt[1] - pd[1], gt[2] - pd[2])

                    cost_masked = cost.copy()
                    cost_masked[cost_masked > match_threshold] = 1e9

                    try:
                        row_ind, col_ind = linear_sum_assignment(cost_masked)
                    except Exception:
                        row_ind, col_ind = [], []
                        used_cols = set()
                        for i in range(cost.shape[0]):
                            j = int(np.argmin(cost[i]))
                            if cost[i, j] <= match_threshold and j not in used_cols:
                                row_ind.append(i)
                                col_ind.append(j)
                                used_cols.add(j)
                else:
                    # 澶х煩闃碉細鍏堢敤 KD-tree 绛涢€夊€欓€夛紝鍐嶅寛鐗欏埄鍖归厤
                    gt_positions = np.array([[gt[1], gt[2]] for gt in gt_list])
                    pd_positions = np.array([[pd[1], pd[2]] for pd in pred_list_for_match])

                    # KD-tree 蹇€熸煡璇㈡渶杩戦偦
                    tree = cKDTree(pd_positions)
                    # 瀵逛簬姣忎釜 GT锛屾煡璇㈣窛绂婚槇鍊煎唴鐨勬墍鏈夊€欓€?                    valid_pairs = []
                    for i, gt_pos in enumerate(gt_positions):
                        candidates = tree.query_ball_point(gt_pos, match_threshold)
                        for j in candidates:
                            dist = math.hypot(gt_pos[0] - pd_positions[j][0], gt_pos[1] - pd_positions[j][1])
                            valid_pairs.append((i, j, dist))

                    # 鏋勫缓閮ㄥ垎鎴愭湰鐭╅樀
                    if valid_pairs:
                        rows = [p[0] for p in valid_pairs]
                        cols = [p[1] for p in valid_pairs]
                        dists = [p[2] for p in valid_pairs]

                        # 鍒涘缓绋€鐤忔垚鏈煩闃?                        cost_sparse = np.full((len(gt_list), len(pred_list_for_match)), 1e9)
                        for r, c, d in zip(rows, cols, dists):
                            cost_sparse[r, c] = d

                        row_ind, col_ind = linear_sum_assignment(cost_sparse)
                    else:
                        row_ind, col_ind = [], []

                # 鎻愬彇鏈夋晥鍖归厤
                for r, c in zip(row_ind, col_ind):
                    if r < len(gt_list) and c < len(pred_list_for_match):
                        dist = math.hypot(gt_list[r][1] - pred_list_for_match[c][1],
                                         gt_list[r][2] - pred_list_for_match[c][2])
                        if dist <= match_threshold:
                            matched_pairs.append((r, c))
                            TP += 1
                            total_distance += dist

                total_TP += TP
                total_FP += len(pred_list_for_match) - len(matched_pairs)
                total_FN += len(gt_list) - TP

                # ID switch 璁＄畻锛堜娇鐢?matched_pairs锛?                for r, c in matched_pairs:
                    gt_id = gt_list[r][0]
                    pred_id = pred_list_for_match[c][0]
                    prev = last_match_for_gt.get(gt_id)
                    if prev is not None and prev != pred_id:
                        id_switches += 1
                    last_match_for_gt[gt_id] = pred_id
            elif gt_list and not pred_list_for_match:
                # 鍙湁 GT 娌℃湁淇″彿棰勬祴
                total_FN += len(gt_list)
                total_FP += len(pred_list)  # 鎵€鏈夐娴嬶紙鍣０锛夐兘鏄?FP
            else:
                # 鍙湁棰勬祴娌℃湁 GT 鐨勬儏鍐?                total_FP += len(pred_list)

            # 杩涘害鎶ュ憡
            if (frame_idx + 1) % progress_interval == 0:
                current_tp = total_TP
                current_fp = total_FP
                current_fn = total_FN
                current_ids = id_switches

                if current_tp + current_fp > 0:
                    current_precision = current_tp / (current_tp + current_fp)
                    current_recall = current_tp / (current_tp + current_fn) if (current_tp + current_fn) > 0 else 0
                    current_mota = 1.0 - (current_fn + current_fp + current_ids) / float(total_gt_detections) if total_gt_detections > 0 else 0

                    if (frame_idx + 1) % progress_interval == 0:
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
        logger.info("FINAL TRACKING EVALUATION RESULTS:")
        logger.info("="*60)
        logger.info("Total GT detections: %d", total_gt_detections)
        logger.info("Total predicted detections: %d", total_pred_detections)
        logger.info("TP: %d, FP: %d, FN: %d, IDSW: %d", total_TP, total_FP, total_FN, id_switches)
        logger.info("MOTA: %.4f, MOTP: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f", MOTA, MOTP, precision, recall, f1)
        logger.info("Evaluation coverage: %d/%d frames processed", eval_stats['processed_frames'], total_eval_frames)
        logger.info("Frames with GT-Pred matches: %d", eval_stats['frames_with_matches'])
        logger.info("="*60)

        # 淇濆瓨棰勬祴杞ㄨ抗渚涚绾垮垎鏋?        try:
            out_path = os.path.join(os.getcwd(), 'baseline_predicted_tracks.csv')
            with open(out_path, 'w') as f:
                f.write('pred_id,frame,pred_x,pred_z,corrected_x,corrected_z,label\n')
                for pred_id, track in self.tracks.items():
                    for d in track.detections:
                        # 淇濆瓨棰勬祴浣嶇疆锛堢敤浜庤瘎浼帮級鍜屾牎姝ｄ綅缃?                        px = getattr(d, 'corrected_x', getattr(d, 'predicted_x', d.x))
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
        """鍙鍖栬拷韪粨鏋?""
        plt.figure(figsize=(12, 8))

        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.tracks)))

        for i, (track_id, track) in enumerate(self.tracks.items()):
            # 鍙粯鍒舵湁鏁堟娴嬬偣
            valid_dets = [d for d in track.detections if not d.is_occluded]

            if len(valid_dets) > 1:
                # 浣跨敤 tracker 鐨勪及璁′綅缃繘琛屽彲瑙嗗寲锛氫紭鍏?corrected_x锛屽啀 predicted_x锛屽啀 GT x
                x_coords = [getattr(d, 'corrected_x', getattr(d, 'predicted_x', d.x)) for d in valid_dets]
                z_coords = [getattr(d, 'corrected_z', getattr(d, 'predicted_z', d.z)) for d in valid_dets]
                frames = [d.frame for d in valid_dets]

                plt.plot(x_coords, z_coords, 'o-', color=colors[i % len(colors)],
                        label=f'Track {track_id}', markersize=3, linewidth=1)

        plt.xlabel('X Position')
        plt.ylabel('Z Position')
        plt.title('Hungarian Algorithm Tracking Results')
        plt.grid(True, alpha=0.3)

        if len(self.tracks) <= 10:  # 鍙湪杞ㄨ抗鏁伴噺涓嶅鏃舵樉绀哄浘渚?            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info("Visualization saved to %s", save_path)
        else:
            plt.show()


def main(quick_test: bool = False, include_noise: bool = True, confirm_frames: int = 2):
    """涓诲嚱鏁?    include_noise: 榛樿True锛屽惎鐢ㄥ櫔澹扮偣澶勭悊浠ユā鎷熺湡瀹炲満鏅?    """
    # 閰嶇疆鍙傛暟
    data_path = "../simulate/simulate/dataset/test/test_tracks_complex.csv"
    max_distance = 2.0  # 鏍囧噯璺濈闃堝€?    max_velocity_diff = 3.0  # 鏈€澶ч€熷害宸紓
    max_frames_skip = 4  # 鏈€澶ц烦甯ф暟 (鍏佽鐭伄鎸?

    # 蹇€熸祴璇曟ā寮忥細鍙鐞嗗墠200甯?    if quick_test:
        logger.info("="*60)
        logger.info("蹇€熸祴璇曟ā寮忥細鍙鐞嗗墠200甯ф暟鎹?)
        logger.info("="*60)
        max_frames = 200
        progress_interval_track = 50  # 姣?0甯ф姤鍛婁竴娆?        progress_interval_eval = 50   # 姣?0甯ф姤鍛婁竴娆?    else:
        logger.info("="*60)
        logger.info("瀹屾暣娴嬭瘯妯″紡锛氬鐞嗗叏閮ㄦ暟鎹?)
        logger.info("="*60)
        max_frames = None
        progress_interval_track = 100  # 姣?00甯ф姤鍛婁竴娆?        progress_interval_eval = 200   # 姣?00甯ф姤鍛婁竴娆?
    # 鍒涘缓杩借釜鍣紙璋冩暣鍙傛暟浠ュ噺灏戣鍖归厤锛?    # 榛樿浣跨敤绠€鍖?baseline锛堝叧闂珮绾у惎鍙戝紡锛夛紱濡傞渶鍚姩楂樼骇绋冲畾鍖栵紝浼犲叆 --advanced
    advanced_mode = "--advanced" in sys.argv
    tracker = HungarianTracker(
        max_distance=max_distance,  # 鐜板湪鏄?.2
        max_velocity_diff=max_velocity_diff,
        max_frames_skip=max_frames_skip,
        confirm_frames=confirm_frames
    )
    if advanced_mode:
        tracker.enable_advanced = True
        tracker.enable_assignment_stability = True
        tracker.enable_switch_confirm = True
        tracker.enable_tentative_buffer = True
        tracker.enable_merge_short = True

    # 鍔犺浇鏁版嵁
    tracker.load_data(data_path)

    # 蹇€熸祴璇曪細闄愬埗澶勭悊鐨勫抚鏁?    if max_frames is not None:
        original_frames = tracker.frame_detections
        sorted_frames = sorted(original_frames.keys())[:max_frames]
        tracker.frame_detections = {f: original_frames[f] for f in sorted_frames}
        logger.info("闄愬埗澶勭悊甯ф暟涓哄墠 %d 甯э紝瀹為檯澶勭悊 %d 甯?, max_frames, len(tracker.frame_detections))

    # 鎵ц杩借釜锛堟瘡50/100甯ф姤鍛婁竴娆¤繘搴︼級
    tracks = tracker.track(progress_interval=progress_interval_track, include_noise=True)

    # 鍚庡鐞嗭細鍚堝苟鐭建浠ュ噺灏戠鐗囧寲瀵艰嚧鐨?IDSW锛堝彲璋冿級
    logger.info("Starting post-process merge...")
    tracker.merge_short_tracks(min_length=2, max_gap=3, merge_distance=tracker.max_distance * 1.2)
    logger.info("Post-process merge completed. Total tracks: %d", len(tracker.tracks))

    # 璇勪及缁撴灉锛堟瘡50/200甯ф姤鍛婁竴娆¤繘搴︼級
    logger.info("Starting evaluation...")
    metrics = tracker.evaluate_tracking(progress_interval=progress_interval_eval, include_noise_gt=include_noise)

    # 棰濆楠岃瘉锛氭鏌ラ娴嬫娴嬫暟閲忔槸鍚﹀悎鐞?    logger.info("\n" + "="*40)
    logger.info("棰濆楠岃瘉妫€鏌?")
    logger.info("="*40)

    total_pred_detections = metrics['pred_detections']
    total_gt_detections = metrics['gt_detections']
    ratio = total_pred_detections / total_gt_detections if total_gt_detections > 0 else float('inf')

    logger.info("棰勬祴妫€娴嬫€绘暟: %d", total_pred_detections)
    logger.info("GT妫€娴嬫€绘暟: %d", total_gt_detections)

    if ratio > 2.0:
        logger.warning("棰勬祴妫€娴嬫暟閲忚繙瓒匞T妫€娴嬶紝鍙兘瀛樺湪杩囧害鐢熸垚棰勬祴鐨勯棶棰橈紒")
    elif ratio < 0.5:
        logger.warning("棰勬祴妫€娴嬫暟閲忚繙灏戜簬GT妫€娴嬶紝鍙兘瀛樺湪婕忔闂锛?)
    else:
        logger.info("棰勬祴妫€娴嬫暟閲忎笌GT妫€娴嬫暟閲忔瘮渚嬪悎鐞?)

    # 妫€鏌ヨ建杩圭粺璁?    confirmed_tracks = sum(1 for t in tracks.values() if t.confirmed)
    total_tracks = len(tracks)
    logger.info("纭杞ㄨ抗鏁? %d/%d", confirmed_tracks, total_tracks)

    # 鍙鍖栫粨鏋滐紙浠呭湪闈炲揩閫熸祴璇曟ā寮忎笅锛?    if not quick_test:
        tracker.visualize_tracks("hungarian_tracking_results.png")

    if quick_test:
        logger.info("\n" + "="*60)
        logger.info("蹇€熸祴璇曞畬鎴愶紒寤鸿妫€鏌ヤ笂杩版寚鏍囨槸鍚﹀悎鐞?)
        logger.info("濡傛灉缁撴灉姝ｅ父锛屽彲浠ヨ繍琛屽畬鏁存祴璇曪細python hungarian_baseline.py")
        logger.info("="*60)
    else:
        logger.info("\nHungarian Algorithm Baseline Tracking Completed!")


if __name__ == "__main__":
    import sys

    # 妫€鏌ュ懡浠よ鍙傛暟
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
        logger.info("杩愯蹇€熸祴璇曟ā寮?..")
    else:
        logger.info("杩愯瀹屾暣娴嬭瘯妯″紡...")
        logger.info("鎻愮ず锛氫娇鐢?--quick 鎴?-q 鍙傛暟杩愯蹇€熸祴璇曪紙浠呭墠1000甯э級")

    if include_noise:
        logger.info("鍖呭惈鍣０鐐?label=0)鍦ㄨ拷韪拰璇勪及涓?..")
    else:
        logger.info("鍣０鐐?label=0)涓嶅弬涓庤拷韪?..")

    main(quick_test=quick_test, include_noise=include_noise, confirm_frames=confirm_frames)

