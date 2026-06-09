#!/usr/bin/env python3
"""
robot_sim2real — Linux 統一測試腳本
=====================================
三種測試模式：

  policy  — ONNX 推論（假/真 IMU，可接或不接馬達）
  replay  — 從錄製 CSV 重播關節狀態，自動驗證 policy 輸出一致性
  check   — 快速自動檢查（N 步推論，驗證輸出範圍、無 NaN）

一鍵測試（從任意目錄）：
  python linux/test/test_policy.py                          # 快速 check
  python linux/test/test_policy.py --mode check             # 同上
  python linux/test/test_policy.py --mode policy --dry-run  # 假 IMU + 假馬達
  python linux/test/test_policy.py --mode policy --imu --dry-run        # 真 IMU
  python linux/test/test_policy.py --mode policy --imu --can can0       # 真 IMU + 馬達
  python linux/test/test_policy.py --mode replay --recording recordings/xxx.csv
  python linux/test/test_policy.py --mode replay            # 自動選最新錄製
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

# ── 路徑設定（從本檔案位置自動推算，不需手動改路徑）────────────────────────────
_HERE       = Path(__file__).resolve().parent          # linux/test/
_LINUX_IMU  = _HERE.parent / "imu"                    # linux/imu/
_REPO_ROOT  = _HERE.parent.parent                     # repo 根目錄（imu/）

# tools/ 模組
sys.path.insert(0, str(_REPO_ROOT / "tools"))
from spike_filter import SpikeFilter, JOINT_NAMES as _SF_JOINT_NAMES  # noqa: E402

# 預設策略檔：優先找 repo 內 models/
def _find_default_policy() -> Path:
    kinfers = sorted((_REPO_ROOT / "models").glob("*.kinfer"))
    if kinfers:
        return kinfers[-1]
    raise FileNotFoundError(f"找不到 .kinfer 模型，請將模型放入 {_REPO_ROOT / 'models'}/")

DEFAULT_POLICY = _find_default_policy()

# 錄製資料夾：repo 內的 recordings/（跨機器一致）
RECORDINGS_DIR = _REPO_ROOT / "recordings"


# ── CLI 輸出工具 ──────────────────────────────────────────────────────────────

def _banner(title: str, width: int = 60):
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)

def _section(title: str):
    print(f"\n  ── {title} ──")

def _ok(msg: str):   print(f"  [OK]   {msg}")
def _warn(msg: str): print(f"  [WARN] {msg}")
def _fail(msg: str): print(f"  [FAIL] {msg}")
def _info(msg: str): print(f"  {msg}")

# ── 策略關節順序（20-dim，ONNX 模型輸入/輸出順序）────────────────────────────────
POLICY_JOINT_NAMES = [
    "dof_right_shoulder_pitch_03", "dof_right_shoulder_roll_03",
    "dof_right_shoulder_yaw_02",   "dof_right_elbow_02",
    "dof_right_wrist_00",
    "dof_left_shoulder_pitch_03",  "dof_left_shoulder_roll_03",
    "dof_left_shoulder_yaw_02",    "dof_left_elbow_02",
    "dof_left_wrist_00",
    "dof_right_hip_pitch_04",  "dof_right_hip_roll_03",  "dof_right_hip_yaw_03",
    "dof_right_knee_04",       "dof_right_ankle_02",
    "dof_left_hip_pitch_04",   "dof_left_hip_roll_03",   "dof_left_hip_yaw_03",
    "dof_left_knee_04",        "dof_left_ankle_02",
]

# 錄製 CSV 的關節順序（10 腿，train_v1/test_policy.py）
RECORDING_JOINT_NAMES = [
    "dof_right_hip_pitch_04", "dof_right_hip_roll_03", "dof_right_hip_yaw_03",
    "dof_right_knee_04",      "dof_right_ankle_02",
    "dof_left_hip_pitch_04",  "dof_left_hip_roll_03",  "dof_left_hip_yaw_03",
    "dof_left_knee_04",       "dof_left_ankle_02",
]
# 錄製關節 → policy 20-dim 索引的對應（也是腿部 10 關節在 20-dim 中的位置）
_REC_TO_POL = [POLICY_JOINT_NAMES.index(n) for n in RECORDING_JOINT_NAMES]
_LEG_INDICES = np.array(_REC_TO_POL, dtype=np.int32)  # 10 個腿部索引

# ── 馬達 CAN 介面對照表（由 main() 填入，供 run_* 顯示用）──────────────────────
_MID_CAN: dict = {}

# ── CAN ID → 關節名稱、型號、PD 增益（10 腿部馬達）────────────────────────────
MOTOR_CONFIG = {
    # torque_cap：初次安全測試用（Nm）。走路測試用 --torque-cap 0 移除限制
    # 參考 MuJoCo 模擬峰值：hip_pitch~81Nm, knee~84Nm, ankle~11.9Nm
    31: {"name": "dof_left_hip_pitch_04",  "type": "04", "kp": 150.0, "kd": 24.722, "torque_cap": 60.0},
    32: {"name": "dof_left_hip_roll_03",   "type": "03", "kp": 200.0, "kd": 26.387, "torque_cap": 30.0},
    33: {"name": "dof_left_hip_yaw_03",    "type": "03", "kp": 100.0, "kd":  3.419, "torque_cap": 30.0},
    34: {"name": "dof_left_knee_04",       "type": "04", "kp": 150.0, "kd":  8.654, "torque_cap": 60.0},
    35: {"name": "dof_left_ankle_02",      "type": "02", "kp":  40.0, "kd":  0.990, "torque_cap": 10.0},
    41: {"name": "dof_right_hip_pitch_04", "type": "04", "kp": 150.0, "kd": 24.722, "torque_cap": 60.0},
    42: {"name": "dof_right_hip_roll_03",  "type": "03", "kp": 200.0, "kd": 26.387, "torque_cap": 30.0},
    43: {"name": "dof_right_hip_yaw_03",   "type": "03", "kp": 100.0, "kd":  3.419, "torque_cap": 30.0},
    44: {"name": "dof_right_knee_04",      "type": "04", "kp": 150.0, "kd":  8.654, "torque_cap": 60.0},
    45: {"name": "dof_right_ankle_02",     "type": "02", "kp":  40.0, "kd":  0.990, "torque_cap": 10.0},
}
ACTUATOR_TYPE_MAP = {"02": "Robstride02", "03": "Robstride03", "04": "Robstride04"}
MAX_TORQUE = {"04": 84.0, "03": 42.0, "02": 11.9}

# ── Raw CAN 狀態讀取器（仿 Rust firmware read_responses_update）─────────────────
# 解決 PyRobstrideDriver.get_actuator_state() 的根本問題：
#   - 多顆馬達同時廣播，讀到的幀可能來自其他馬達 → "CAN ID mismatch" exception
#   - mux=0x15 (unsolicited fault) → Python driver 直接 panic，Rust 版本有處理
# 解法：直接開 raw socketcan socket 讀所有幀，按 motor_id 分類存入字典
import socket as _socket
import struct as _struct
import threading as _threading

# 各型號物理量程（來自 deploye_robot/firmware/src/robstride_utils.rs）
_MOTOR_PHYS_RANGES = {
    "04": {"angle": (-4 * math.pi, 4 * math.pi), "vel": (-15.0, 15.0)},
    "03": {"angle": (-4 * math.pi, 4 * math.pi), "vel": (-20.0, 20.0)},
    "02": {"angle": (-4 * math.pi, 4 * math.pi), "vel": (-44.0, 44.0)},
}
_CAN_FRAME_SIZE = 16  # socketcan struct can_frame 大小（bytes）


def _can_scale(raw_u16: int, phys_min: float, phys_max: float) -> float:
    """CAN u16 [0, 65535] → 物理量（同 Rust RangeSet::scale_value）。"""
    return phys_min + (raw_u16 / 65535.0) * (phys_max - phys_min)


class CanStateReader:
    """背景執行緒持續讀取所有 CAN 幀並按 motor_id 分類。

    幀格式（socketcan struct can_frame = 16 bytes）：
      byte[0]   host_id
      byte[1]   actuator_can_id  ← motor ID
      byte[2]   fault_flags / mode
      byte[3]   mux（低 5 bits）  0x02=feedback  0x15=fault
      byte[4-7] len/pad
      byte[8-9]   angle_be    (big-endian u16)
      byte[10-11] vel_be      (big-endian u16)
      byte[12-13] torque_be   (big-endian u16)
      byte[14-15] temp_be     (big-endian u16)
    """

    def __init__(self, interfaces: list, motor_config: dict):
        self._lock   = _threading.Lock()
        self._state: dict = {}  # motor_id → {"pos": float, "vel": float}
        self._mconfig = motor_config
        self._stop   = _threading.Event()

        for iface in interfaces:
            t = _threading.Thread(target=self._read_loop, args=(iface,),
                                  daemon=True, name=f"can-reader-{iface}")
            t.start()

    def _read_loop(self, iface: str):
        try:
            sock = _socket.socket(_socket.AF_CAN, _socket.SOCK_RAW, _socket.CAN_RAW)
            sock.bind((iface,))
            sock.settimeout(0.1)
        except OSError as e:
            print(f"  [CanReader] {iface} 開啟失敗: {e}")
            return

        try:
            while not self._stop.is_set():
                try:
                    frame = sock.recv(_CAN_FRAME_SIZE)
                except _socket.timeout:
                    continue
                except OSError:
                    break

                if len(frame) < _CAN_FRAME_SIZE:
                    continue

                mux = frame[3] & 0x1F   # byte[3] 低 5 bits = mux
                if mux != 0x02:         # 只處理 feedback（0x15 fault 靜默忽略）
                    continue

                motor_id = frame[1]     # actuator_can_id 在 byte[1]
                if motor_id not in self._mconfig:
                    continue

                mtype  = self._mconfig[motor_id]["type"]
                ranges = _MOTOR_PHYS_RANGES.get(mtype, _MOTOR_PHYS_RANGES["03"])

                raw_angle = _struct.unpack_from(">H", frame, 8)[0]
                raw_vel   = _struct.unpack_from(">H", frame, 10)[0]
                pos = _can_scale(raw_angle, *ranges["angle"])
                vel = _can_scale(raw_vel,   *ranges["vel"])

                with self._lock:
                    self._state[motor_id] = {"pos": pos, "vel": vel}
        finally:
            sock.close()

    def get(self, motor_id: int):
        """回傳最新 (pos_rad, vel_rad_s)。若尚無資料回傳 (None, None)。"""
        with self._lock:
            s = self._state.get(motor_id)
        if s is None:
            return None, None
        return s["pos"], s["vel"]

    def stop(self):
        self._stop.set()


_can_reader: "CanStateReader | None" = None

# 正弦波參數（與舊版 test_motor_policy.py 一致，已實測通過）
SINE_AMP_RAD = {
    "04": math.radians(15),   # hip pitch / knee ±15°
    "03": math.radians(10),   # hip roll / yaw   ±10°
    "02": math.radians(8),    # ankle             ±8°
}
SINE_FREQ_HZ = 0.3   # 預設頻率，可由 --sine-freq 覆蓋


def _sine_peak_torque(mid: int, freq_hz: float) -> tuple[float, float]:
    """估算指定頻率下的峰值扭矩與佔最大值比例。
    主要來源：kd × 振幅 × ω（速度項，頻率越高越大）。
    """
    cfg   = MOTOR_CONFIG[mid]
    amp   = SINE_AMP_RAD[cfg["type"]]
    omega = 2 * math.pi * freq_hz
    peak  = cfg["kd"] * amp * omega          # kd 速度項（主導）
    peak += cfg["kp"] * amp * 0.05           # kp 位置誤差估算（5% 相位差）
    max_t = MAX_TORQUE[cfg["type"]]
    return peak, peak / max_t

# 安全關節限制（policy 輸出的 hard clip）
_ZEROS_DEG = {
    "dof_right_hip_pitch_04": -20.0, "dof_right_hip_roll_03":  0.0,
    "dof_right_hip_yaw_03":    0.0,  "dof_right_knee_04":    -50.0,
    "dof_right_ankle_02":     30.0,  "dof_left_hip_pitch_04": 20.0,
    "dof_left_hip_roll_03":    0.0,  "dof_left_hip_yaw_03":    0.0,
    "dof_left_knee_04":       50.0,  "dof_left_ankle_02":    -30.0,
}
# 各關節安全範圍：對齊 MJCF joint limit（不讓指令超出物理限位）
# ankle 物理限位比 ±55° 窄（MJCF right: -13°~+72°, left: -72°~+13°），需單獨限制
SAFE_MIN = {
    "dof_right_hip_pitch_04": math.radians(-20.0 - 55),  # -75°
    "dof_right_hip_roll_03":  math.radians(  0.0 - 55),  # -55°
    "dof_right_hip_yaw_03":   math.radians(  0.0 - 55),  # -55°
    "dof_right_knee_04":      math.radians(-50.0 - 55),  # -105°
    "dof_right_ankle_02":     math.radians(-13.0),        # MJCF 物理下限
    "dof_left_hip_pitch_04":  math.radians( 20.0 - 55),  # -35°
    "dof_left_hip_roll_03":   math.radians(  0.0 - 55),  # -55°
    "dof_left_hip_yaw_03":    math.radians(  0.0 - 55),  # -55°
    "dof_left_knee_04":       math.radians( 50.0 - 55),  # -5°
    "dof_left_ankle_02":      math.radians(-72.0),        # MJCF 物理下限
}
SAFE_MAX = {
    "dof_right_hip_pitch_04": math.radians(-20.0 + 55),  # +35°
    "dof_right_hip_roll_03":  math.radians(  0.0 + 55),  # +55°
    "dof_right_hip_yaw_03":   math.radians(  0.0 + 55),  # +55°
    "dof_right_knee_04":      math.radians(-50.0 + 55),  # +5°
    "dof_right_ankle_02":     math.radians(+72.0),        # MJCF 物理上限
    "dof_left_hip_pitch_04":  math.radians( 20.0 + 55),  # +75°
    "dof_left_hip_roll_03":   math.radians(  0.0 + 55),  # +55°
    "dof_left_hip_yaw_03":    math.radians(  0.0 + 55),  # +55°
    "dof_left_knee_04":       math.radians( 50.0 + 55),  # +105°
    "dof_left_ankle_02":      math.radians(+13.0),        # MJCF 物理上限
}

# policy 20-dim 的安全限制陣列（腿部有限制，手臂用極大值）
_SAFE_MIN_ARR = np.array([
    SAFE_MIN.get(n, -math.pi * 2) for n in POLICY_JOINT_NAMES], dtype=np.float32)
_SAFE_MAX_ARR = np.array([
    SAFE_MAX.get(n,  math.pi * 2) for n in POLICY_JOINT_NAMES], dtype=np.float32)


# ── 工具函數 ───────────────────────────────────────────────────────────────────

def motor_id_to_policy_idx(mid: int) -> int:
    return POLICY_JOINT_NAMES.index(MOTOR_CONFIG[mid]["name"])


def load_kinfer(path: str | Path):
    """載入 .kinfer（tar.gz 包含 init_fn.onnx + step_fn.onnx + metadata.json）。"""
    import json
    with tempfile.TemporaryDirectory() as d:
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(d)
        with open(os.path.join(d, "metadata.json")) as f:
            meta = json.load(f)
        init_sess = ort.InferenceSession(os.path.join(d, "init_fn.onnx"))
        step_sess = ort.InferenceSession(os.path.join(d, "step_fn.onnx"))
    return init_sess, step_sess, meta


def _n_joints(step_sess) -> int:
    """回傳模型期望的關節數（10 腿部 or 20 全身）。"""
    for inp in step_sess.get_inputs():
        if inp.name == "joint_angles":
            return int(inp.shape[0])
    return 20


def expand_actions(actions, joint_pos) -> np.ndarray:
    """把 10-dim 腿部 actions 展開為 20-dim；20-dim 直接返回。"""
    if len(actions) == 20:
        return actions
    out = joint_pos.copy()          # 手臂保持當前位置
    for leg_i, pol_i in enumerate(_LEG_INDICES):
        out[pol_i] = actions[leg_i]
    return out


def build_policy_feed(step_sess, joint_pos, joint_vel, carry, num_commands, sim_t,
                      bridge=None):
    """組裝 policy 輸入字典。bridge 不為 None 時讀真實 IMU。"""
    import threading
    names = {i.name for i in step_sess.get_inputs()}
    n = _n_joints(step_sess)
    if n == 10:
        jpos = joint_pos[_LEG_INDICES].astype(np.float32)
        jvel = joint_vel[_LEG_INDICES].astype(np.float32)
    else:
        jpos = joint_pos.astype(np.float32)
        jvel = joint_vel.astype(np.float32)
    feed = {
        "joint_angles":             jpos,
        "joint_angular_velocities": jvel,
        "carry":                    carry,
    }
    if bridge is not None:
        with bridge._imu_lock:
            acc  = bridge.IMU_STATE["acc"].copy().astype(np.float32)
            gyro = bridge.IMU_STATE["gyro"].copy().astype(np.float32)
            quat = bridge.IMU_STATE["quat"].copy()
        proj_grav = np.array(bridge.proj_gravity_from_quat(*quat), dtype=np.float32)
    else:
        acc       = np.array([0.0, 0.0, -9.81], dtype=np.float32)
        gyro      = np.zeros(3, dtype=np.float32)
        proj_grav = np.array([0.0, 0.0, -1.0],  dtype=np.float32)

    if "projected_gravity" in names:
        feed["projected_gravity"] = proj_grav
    if "imu_gyro" in names:
        feed["imu_gyro"] = gyro
    elif "gyroscope" in names:
        feed["gyroscope"] = gyro
    if "imu_acc" in names:
        feed["imu_acc"] = acc
    elif "accelerometer" in names:
        feed["accelerometer"] = acc
    if "time" in names:
        feed["time"] = np.array([sim_t], dtype=np.float32)
    if "command" in names:
        feed["command"] = np.zeros(max(num_commands, 1), dtype=np.float32)
    return feed


def calc_torque(target, cur, vel, kp, kd, max_t):
    t = kp * (target - cur) + kd * (-vel)
    return max(-max_t, min(max_t, t))


# ── IMU bridge 載入（可選）────────────────────────────────────────────────────

def load_bridge_module(imu_name: str = "WT901BLE67"):
    """動態載入 linux/imu/bridge_h30.py 或 BLE bridge。"""
    # 預設用 H30 USB bridge
    bridge_path = _LINUX_IMU / "bridge_h30.py"
    if not bridge_path.exists():
        raise FileNotFoundError(f"找不到 {bridge_path}")
    spec   = importlib.util.spec_from_file_location("bridge", bridge_path)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    return bridge


def start_imu(imu_port: str = "/dev/ttyACM0", imu_baud: int = 460800):
    """啟動 H30 bridge thread，等待第一筆資料。"""
    import threading
    bridge = load_bridge_module()

    def _run():
        bridge.run_imu_only(imu_port, imu_baud)

    t = threading.Thread(target=_run, daemon=True, name="h30-bridge")
    t.start()

    _info(f"啟動 H30 bridge {imu_port} @ {imu_baud} baud，等待資料...")
    for _ in range(200):
        time.sleep(0.1)
        with bridge._imu_lock:
            if bridge.IMU_STATE["updated"]:
                _ok(f"IMU 就緒（{imu_port} @ {imu_baud}）")
                return bridge
    _warn("20 秒內未收到 IMU 資料，使用假值繼續")
    return bridge


# ── 馬達驅動器 ─────────────────────────────────────────────────────────────────

def setup_driver(can_assignment: dict) -> dict:
    """建立馬達驅動器映射。

    Phase 1: ping 全部馬達
    Phase 2: enable 全部馬達（04 送 3 次確保收到）
    Phase 3: 背景持續發送保持指令，讓你摸馬達確認鎖住。
             輸入已確認的 ID → 從待確認清單移除。
             r ID → 對指定馬達重新 enable。
             Enter → 全部跳過繼續。q → 中止。
    """
    import threading
    from robstride_driver import PyRobstrideDriver, PyRobstrideActuatorType, PyActuatorCommand

    iface_drivers: dict = {}
    for iface in set(can_assignment.values()):
        d = PyRobstrideDriver(iface)
        d.connect(iface)
        iface_drivers[iface] = d

    driver_map: dict = {}

    # Phase 1: ping
    print(f"\n  [Phase 1] Ping 全部馬達")
    for mid in sorted(can_assignment.keys()):
        iface = can_assignment[mid]
        d     = iface_drivers[iface]
        cfg   = MOTOR_CONFIG[mid]
        atype = getattr(PyRobstrideActuatorType, ACTUATOR_TYPE_MAP[cfg["type"]])
        d.add_actuator(can_id=mid, actuator_type=atype)
        time.sleep(0.05)
        driver_map[mid] = d
        print(f"    ping 馬達 {mid:2d} ({cfg['name']:<28}) [{iface}]")
    time.sleep(0.15)

    # Phase 2: enable（04 重送 3 次）
    print(f"\n  [Phase 2] Enable 全部馬達")
    for mid in sorted(can_assignment.keys()):
        cfg    = MOTOR_CONFIG[mid]
        d      = driver_map[mid]
        repeat = 3 if cfg["type"] == "04" else 1
        for _ in range(repeat):
            try:
                d.enable_actuator(actuator_id=mid)
            except Exception:
                pass
            time.sleep(0.1 if cfg["type"] == "04" else 0.05)
        print(f"    enable 馬達 {mid:2d} ({cfg['name']:<28})")
    time.sleep(0.3)

    # 啟動 raw CAN reader
    # Robstride 馬達不主動廣播，只有收到指令後才回傳 feedback（mux=0x02）
    # → 先送一次 kp=0 kd=0 無力指令讓每顆馬達回傳第一幀，reader 才能捕捉到位置
    global _can_reader
    ifaces = sorted(set(can_assignment.values()))
    _can_reader = CanStateReader(ifaces, MOTOR_CONFIG)
    print(f"\n  [CAN Reader] 已啟動 raw 幀讀取: {ifaces}")
    def _prime_once():
        """對所有馬達送一次 kp=0 kd=0，觸發 feedback 回傳。"""
        for mid in sorted(can_assignment.keys()):
            try:
                driver_map[mid].send_command(
                    actuator_id=mid,
                    command=PyActuatorCommand(position=0.0, velocity=0.0, torque=0.0,
                                              kp=0.0, kd=0.0),
                )
            except Exception:
                pass

    print(f"  送初始化指令讓馬達回傳位置...")
    _prime_once()
    time.sleep(0.15)

    # 等最多 2 秒收齊；沒回應的馬達重新 enable 再 prime（最多 3 輪）
    for attempt in range(3):
        t_end = time.time() + 2.0
        while time.time() < t_end:
            if all(_can_reader.get(mid)[0] is not None for mid in can_assignment):
                break
            time.sleep(0.05)

        missing = [mid for mid in can_assignment if _can_reader.get(mid)[0] is None]
        if not missing:
            break

        print(f"  [嘗試 {attempt+1}/3] 以下馬達未回應，重新 enable: {missing}")
        for mid in missing:
            cfg = MOTOR_CONFIG[mid]
            repeat = 3 if cfg["type"] == "04" else 1
            for _ in range(repeat):
                try:
                    driver_map[mid].enable_actuator(actuator_id=mid)
                except Exception:
                    pass
                time.sleep(0.1)
        _prime_once()
        time.sleep(0.15)

    ready   = [mid for mid in can_assignment if _can_reader.get(mid)[0] is not None]
    missing = [mid for mid in can_assignment if _can_reader.get(mid)[0] is None]
    print(f"  完成（{len(ready)}/{len(can_assignment)} 顆）")
    if missing:
        _warn(f"以下馬達仍未回傳資料（可能在保護模式）: {missing}")

    # Phase 3: 背景發送 + 互動確認
    print(f"\n  [Phase 3] 持續送指令 — 摸馬達確認鎖住後輸入 ID")
    print(f"  格式：")
    print(f"    31,34       → 確認這些馬達已鎖住")
    print(f"    r 31,34     → 對指定馬達重新 enable")
    print(f"    Enter       → 全部跳過，直接繼續")
    print(f"    q           → 中止程式")
    print()

    pending   = sorted(can_assignment.keys())
    confirmed: list = []

    def _send_loop(stop_event):
        while not stop_event.is_set():
            for mid in sorted(can_assignment.keys()):
                cfg = MOTOR_CONFIG[mid]
                cur, _ = _can_reader.get(mid)
                if cur is None:
                    continue  # 尚無資料，跳過，不送 0° 避免瞬間施力
                try:
                    driver_map[mid].send_command(
                        actuator_id=mid,
                        command=PyActuatorCommand(
                            position=cur, velocity=0.0, torque=0.0,
                            kp=cfg["kp"], kd=cfg["kd"]),
                    )
                except Exception:
                    pass
            time.sleep(0.02)

    stop_evt = threading.Event()
    threading.Thread(target=_send_loop, args=(stop_evt,), daemon=True).start()

    try:
        while pending:
            print(f"  待確認: {pending}")
            print(f"  已確認: {confirmed}")
            print("  > ", end="", flush=True)
            try:
                ans = input().strip()
            except EOFError:
                ans = ""

            if ans.lower() == "q":
                stop_evt.set()
                raise SystemExit("使用者中止")

            if not ans:
                _warn(f"跳過 {pending}，繼續啟動")
                confirmed.extend(pending)
                pending.clear()
                break

            # r ID：重新 enable
            if ans.lower().startswith("r "):
                for tok in ans[2:].split(","):
                    tok = tok.strip()
                    if tok.isdigit():
                        mid = int(tok)
                        cfg = MOTOR_CONFIG.get(mid)
                        if cfg:
                            repeat = 3 if cfg["type"] == "04" else 1
                            for _ in range(repeat):
                                try:
                                    driver_map[mid].enable_actuator(actuator_id=mid)
                                except Exception:
                                    pass
                                time.sleep(0.1)
                            print(f"    重新 enable 馬達 {mid}")
                continue

            # ID：標記確認
            for tok in ans.split(","):
                tok = tok.strip()
                if tok.isdigit():
                    mid = int(tok)
                    if mid in pending:
                        pending.remove(mid)
                        confirmed.append(mid)
                        _ok(f"馬達 {mid:2d} ({MOTOR_CONFIG[mid]['name']}) 確認鎖住")
                    else:
                        print(f"    [忽略] {mid} 不在待確認清單")

        _ok(f"確認完成: {sorted(confirmed)}")
    finally:
        stop_evt.set()

    return driver_map


def send_cmd(driver_map, mid: int, position: float, kp: float, kd: float):
    from robstride_driver import PyActuatorCommand
    driver_map[mid].send_command(
        actuator_id=mid,
        command=PyActuatorCommand(position=position, velocity=0.0, torque=0.0, kp=kp, kd=kd),
    )


def disable_all(driver_map, motor_ids: list):
    from robstride_driver import PyActuatorCommand
    for mid in motor_ids:
        try:
            driver_map[mid].send_command(
                actuator_id=mid,
                command=PyActuatorCommand(position=0.0, velocity=0.0, torque=0.0, kp=0.0, kd=0.0),
            )
        except Exception:
            pass


def _is_mismatch(e: Exception) -> bool:
    return "mismatch" in str(e).lower()


def init_joint_pos_from_reader(motor_ids: list, joint_pos: np.ndarray,
                               joint_vel: np.ndarray, id_to_idx: dict,
                               timeout: float = 3.0):
    """從 CanStateReader 讀取所有馬達的實際當前位置，填入 joint_pos。
    在 home_ramp 開始前呼叫，確保從真實位置出發，避免突然施力。
    """
    if _can_reader is None:
        return
    print("  [Init] 讀取馬達實際位置...")
    t_end = time.time() + timeout
    ready = set()
    while time.time() < t_end and len(ready) < len(motor_ids):
        for mid in motor_ids:
            if mid in ready:
                continue
            pos, vel = _can_reader.get(mid)
            if pos is not None:
                joint_pos[id_to_idx[mid]] = pos
                joint_vel[id_to_idx[mid]] = vel
                ready.add(mid)
        if len(ready) < len(motor_ids):
            time.sleep(0.05)

    if ready:
        print("  [Init] 已讀到位置: " + "  ".join(
            f"{MOTOR_CONFIG[m]['name'].replace('dof_','')[:8]}="
            f"{math.degrees(joint_pos[id_to_idx[m]]):+.1f}°"
            for m in motor_ids if m in ready))
    missing = [m for m in motor_ids if m not in ready]
    if missing:
        _warn(f"以下馬達未收到位置資料，從 0° 出發: {missing}")


def read_states(driver_map, motor_ids, joint_pos, joint_vel, id_to_idx, retries: int = 3):
    """讀取馬達位置與速度。優先使用 raw CAN reader（完全無 mismatch 問題）。"""
    if _can_reader is not None:
        for mid in motor_ids:
            pos, vel = _can_reader.get(mid)
            if pos is not None:
                joint_pos[id_to_idx[mid]] = pos
                joint_vel[id_to_idx[mid]] = vel
        return

    # fallback: PyRobstrideDriver（會有 mismatch 警告但不影響控制）
    for mid in motor_ids:
        idx = id_to_idx[mid]
        for attempt in range(retries):
            try:
                s = driver_map[mid].get_actuator_state(actuator_id=mid)
                joint_pos[idx] = s.position
                joint_vel[idx] = s.velocity
                break
            except Exception as e:
                if attempt == retries - 1 and not _is_mismatch(e):
                    print(f"[WARN] 馬達 {mid} 讀取失敗（{retries}次）: {e}")
                elif not _is_mismatch(e):
                    time.sleep(0.003)


def send_and_read(driver_map, mid: int, step_pos: float, kp: float, kd: float,
                  joint_pos: np.ndarray, joint_vel: np.ndarray, idx: int,
                  retries: int = 3):
    """送指令後讀回狀態。優先使用 raw CAN reader（無 mismatch）。"""
    send_cmd(driver_map, mid, step_pos, kp, kd)

    if _can_reader is not None:
        time.sleep(0.003)
        pos, vel = _can_reader.get(mid)
        if pos is not None:
            joint_pos[idx] = pos
            joint_vel[idx] = vel
        return

    # fallback
    time.sleep(0.003)
    for attempt in range(retries):
        try:
            s = driver_map[mid].get_actuator_state(actuator_id=mid)
            joint_pos[idx] = s.position
            joint_vel[idx] = s.velocity
            return
        except Exception as e:
            if attempt == retries - 1 and not _is_mismatch(e):
                print(f"[WARN] 馬達 {mid} 讀取失敗（{retries}次）: {e}")
            elif not _is_mismatch(e):
                time.sleep(0.003)


# ═══════════════════════════════════════════════════════════════════════════════
# 模式 1：policy  — ONNX 推論
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_step_rad(mid: int, torque_limit_ratio: float) -> float:
    """依關節 kp / max_torque 計算安全步長（rad）。
    保證單步最大扭矩 ≤ torque_limit_ratio × max_torque。
    """
    cfg = MOTOR_CONFIG[mid]
    max_t = MAX_TORQUE[cfg["type"]]
    return (max_t * torque_limit_ratio) / cfg["kp"]


def torque_ratio(target_rad: float, cur_rad: float, vel_rad: float, mid: int) -> float:
    """估算 PD 扭矩佔最大值的比例（0~1+）。"""
    cfg   = MOTOR_CONFIG[mid]
    max_t = MAX_TORQUE[cfg["type"]]
    tau   = cfg["kp"] * (target_rad - cur_rad) + cfg["kd"] * (-vel_rad)
    return abs(tau) / max_t


RAMP_DURATION_S = 3.0   # home_ramp 固定耗時（秒），調整此值控制快慢


def home_ramp(motor_ids: list, driver, id_to_idx: dict,
              joint_pos: np.ndarray, joint_vel: np.ndarray,
              torque_limit_ratio: float = 0.05,
              custom_targets: dict | None = None):
    """從當前位置以固定時間線性插值移動到目標姿態。
    custom_targets: {motor_id: rad}，若為 None 則移到 ZEROS 站姿。
    固定跑完 RAMP_DURATION_S 秒，不以誤差判斷提前結束，確保平順。
    """
    ctrl_dt = 0.02

    home_targets = {mid: math.radians(_ZEROS_DEG[MOTOR_CONFIG[mid]["name"]])
                    for mid in motor_ids if MOTOR_CONFIG[mid]["name"] in _ZEROS_DEG}

    # 等到所有馬達都有實際位置資料（最多 3 秒）
    print("  等待馬達位置資料...", end="", flush=True)
    t_wait = time.time() + 3.0
    while time.time() < t_wait:
        if _can_reader is None:
            break
        if all(_can_reader.get(mid)[0] is not None for mid in motor_ids):
            break
        time.sleep(0.05)
    print(" 完成")

    # 讀取起始位置（插值的起點）
    for mid in motor_ids:
        if _can_reader is not None:
            pos, vel = _can_reader.get(mid)
            if pos is not None:
                joint_pos[id_to_idx[mid]] = pos
                joint_vel[id_to_idx[mid]] = vel

    start_pos = {mid: joint_pos[id_to_idx[mid]] for mid in motor_ids}

    _section(f"Home Ramp — {RAMP_DURATION_S:.0f}秒線性插值至初始姿態")
    _info("起點: " + "  ".join(
        f"{MOTOR_CONFIG[m]['name'].replace('dof_','')[:10]}={math.degrees(start_pos[m]):+.1f}°"
        for m in motor_ids if m in home_targets))
    _info("目標: " + "  ".join(
        f"{MOTOR_CONFIG[m]['name'].replace('dof_','')[:10]}={math.degrees(home_targets[m]):+.0f}°"
        for m in motor_ids if m in home_targets))

    t_start = time.time()
    step = 0
    while True:
        t0      = time.time()
        elapsed = t0 - t_start
        alpha   = min(elapsed / RAMP_DURATION_S, 1.0)   # 0.0 → 1.0

        # 更新實際位置（用於顯示，不影響插值路徑）
        if _can_reader is not None:
            for mid in motor_ids:
                pos, vel = _can_reader.get(mid)
                if pos is not None:
                    joint_pos[id_to_idx[mid]] = pos
                    joint_vel[id_to_idx[mid]] = vel

        for mid in motor_ids:
            if mid not in home_targets:
                continue
            interp = start_pos[mid] + alpha * (home_targets[mid] - start_pos[mid])
            send_cmd(driver, mid, interp, MOTOR_CONFIG[mid]["kp"], MOTOR_CONFIG[mid]["kd"])

        if step % 25 == 0:
            print(f"  [{elapsed:4.1f}/{RAMP_DURATION_S:.0f}s]  "
                  + "  ".join(f"{MOTOR_CONFIG[m]['name'].replace('dof_','')[:8]}"
                               f"={math.degrees(joint_pos[id_to_idx[m]]):+.1f}°"
                               for m in motor_ids if m in home_targets))

        if alpha >= 1.0:
            _ok(f"Home Ramp 完成（{elapsed:.1f}s）")
            break

        step += 1
        slp = ctrl_dt - (time.time() - t0)
        if slp > 0:
            time.sleep(slp)


def run_policy(args, motor_ids: list, active_ids: list, driver, bridge):
    _banner("Policy 模式 — ONNX 即時推論")
    _section("載入策略")
    _info(f"檔案: {Path(args.policy).name}")
    init_sess, step_sess, meta = load_kinfer(args.policy)
    input_names  = [i.name for i in step_sess.get_inputs()]
    num_commands = meta.get("num_commands", 0) or 0
    carry_size   = meta["carry_size"]
    _info(f"輸入: {input_names}")
    _info(f"carry_size={carry_size}  commands={num_commands}")
    if active_ids != motor_ids:
        active_names = [MOTOR_CONFIG[m]["name"].replace("dof_","") for m in active_ids]
        _warn(f"只有 {active_ids} ({active_names}) 接收 policy 指令")

    carry_init = init_sess.run(None, {})
    carry = carry_init[0] if carry_init else np.zeros(carry_size, dtype=np.float32)

    # MLP frame-stack warm-up：carry_size==465 代表 15幀×31D frame buffer
    # 以初始 ZEROS 觀測填滿 buffer，避免全零 carry 讓 policy 第一步輸出極端動作
    if carry_size == 465:
        _info("MLP frame-stack 偵測（carry=465）：以 ZEROS 初始觀測填滿 frame buffer")
        # ZEROS 關節角度，依 RECORDING_JOINT_NAMES 順序（= policy 輸入順序）
        _ZEROS_ORDER = [
            "dof_right_hip_pitch_04", "dof_right_hip_roll_03", "dof_right_hip_yaw_03",
            "dof_right_knee_04",      "dof_right_ankle_02",
            "dof_left_hip_pitch_04",  "dof_left_hip_roll_03",  "dof_left_hip_yaw_03",
            "dof_left_knee_04",       "dof_left_ankle_02",
        ]
        zeros_jp = np.array([math.radians(_ZEROS_DEG.get(n, 0.0)) for n in _ZEROS_ORDER],
                            dtype=np.float32)  # 10D
        zeros_obs = np.concatenate([
            np.array([np.sin(0.0), np.cos(0.0)], np.float32),  # sin/cos time
            zeros_jp,                                            # joint_angles 10D
            np.zeros(10, np.float32),                           # joint_vel 10D
            np.array([0.0, 0.0, -1.0], np.float32),            # proj_grav（直立）
            np.array([0.0, 0.0,  9.81], np.float32),           # imu_acc（MuJoCo 靜止值）
            np.zeros(3, np.float32),                            # imu_gyro
        ])  # 31D
        carry = np.tile(zeros_obs, 15).astype(np.float32)  # 465D

    id_to_idx = {mid: motor_id_to_policy_idx(mid) for mid in motor_ids}
    ctrl_dt   = 0.02
    sim_t     = 0.0
    step_cnt  = 0

    joint_pos = np.zeros(20, dtype=np.float32)
    joint_vel = np.zeros(20, dtype=np.float32)

    # Dry-run：把全部 10 個腿部關節都初始化到 ZEROS（包含非 active 的右腿）
    # 若只初始化 active motors，policy 會看到右膝=0°（應是-50°）等異常輸入 → 輸出爆炸
    if args.dry_run:
        for mid, cfg in MOTOR_CONFIG.items():
            name = cfg["name"]
            if name in _ZEROS_DEG:
                joint_pos[motor_id_to_policy_idx(mid)] = math.radians(_ZEROS_DEG[name])

    imu_info = "真實 H30 IMU" if bridge is not None else "假 IMU（直立靜止）"
    _section(f"運行配置")
    _info(f"IMU   : {imu_info}")
    ifaces = sorted(set(_MID_CAN.get(m, args.can) for m in motor_ids))
    _info(f"CAN   : {'DRY RUN（不送指令）' if args.dry_run else 'LIVE — ' + str(ifaces)}")
    _info(f"馬達  : {motor_ids}")
    _info(f"Active: {active_ids}")
    _section("腿部關節安全限制")
    _print_joint_limits_table()

    # Home ramp：先緩移到初始姿態，避免從零位暴衝
    if driver and not args.skip_home_ramp:
        init_joint_pos_from_reader(motor_ids, joint_pos, joint_vel, id_to_idx)
        home_ramp(motor_ids, driver, id_to_idx, joint_pos, joint_vel, 0.05)
        input("\n  [確認] 已到達初始姿態，按 Enter 開始 Policy 推論...")
    elif args.skip_home_ramp:
        _warn("--skip-home-ramp：跳過 Home ramp，確認機器人已在初始姿態")

    # ── 第一幀緩移：計算 policy 第一步輸出，若偏差 > 3° 先緩慢移動過去 ─────────
    # 避免 ZEROS → policy_t0 之間有 10-20° 落差導致瞬間施力
    if driver and not args.dry_run:
        _section("第一幀緩移（policy pre-ramp）")
        _dummy_pos = joint_pos.copy()
        _dummy_vel = joint_vel.copy()
        _feed0 = build_policy_feed(step_sess, _dummy_pos, _dummy_vel, carry,
                                   num_commands, 0.0, bridge)
        _first_out = step_sess.run(None, _feed0)
        _first_act = np.clip(expand_actions(_first_out[0], _dummy_pos),
                             _SAFE_MIN_ARR, _SAFE_MAX_ARR)
        # 計算各 active 關節的最大偏差
        _max_dev_deg = max(
            abs(math.degrees(float(_first_act[motor_id_to_policy_idx(m)])
                             - joint_pos[motor_id_to_policy_idx(m)]))
            for m in active_ids
        )
        if _max_dev_deg > 3.0:
            _info(f"偵測到第一幀偏差 {_max_dev_deg:.1f}°，先緩移 2s 到 policy 第一幀位置...")
            _pre_ramp_start = joint_pos.copy()
            _pre_ramp_dur   = 2.0
            _t_pre = time.time()
            while True:
                _t0 = time.time()
                _alpha = min((_t0 - _t_pre) / _pre_ramp_dur, 1.0)
                if _can_reader is not None:
                    for mid in motor_ids:
                        _p, _v = _can_reader.get(mid)
                        if _p is not None:
                            joint_pos[id_to_idx[mid]] = _p
                            joint_vel[id_to_idx[mid]] = _v
                for mid in active_ids:
                    _idx  = motor_id_to_policy_idx(mid)
                    _tgt  = _pre_ramp_start[_idx] + _alpha * (_first_act[_idx] - _pre_ramp_start[_idx])
                    send_cmd(driver, mid, float(_tgt),
                             MOTOR_CONFIG[mid]["kp"], MOTOR_CONFIG[mid]["kd"])
                if _alpha >= 1.0:
                    _ok("第一幀緩移完成")
                    break
                _slp = 0.02 - (time.time() - _t0)
                if _slp > 0:
                    time.sleep(_slp)
        else:
            _ok(f"第一幀偏差 {_max_dev_deg:.1f}° < 3°，直接開始")

    record_secs = getattr(args, "record_secs", 0)
    rec_tgt  = {mid: [] for mid in motor_ids}   # 記錄 target positions
    rec_tau  = {mid: [] for mid in motor_ids}   # 記錄 torques
    rec_pg      = []                             # 記錄 projected gravity
    rec_acc     = []                             # 記錄 acc
    rec_gyro    = []                             # 記錄 gyro
    rec_csv_rows = []                            # 記錄每幀 target（用於 replay）

    if record_secs > 0:
        _info(f"記錄模式：自動跑 {record_secs}s 後輸出統計分析，並存 CSV 供 replay")
    _info("Ctrl+C 停止")
    try:
        while True:
            t0 = time.time()
            if driver:
                read_states(driver, motor_ids, joint_pos, joint_vel, id_to_idx)

            feed    = build_policy_feed(step_sess, joint_pos, joint_vel, carry,
                                        num_commands, sim_t, bridge)
            outputs = step_sess.run(None, feed)
            actions = np.clip(expand_actions(outputs[0], joint_pos), _SAFE_MIN_ARR, _SAFE_MAX_ARR)
            carry   = outputs[1]

            if not args.dry_run and driver:
                for mid in motor_ids:
                    idx = motor_id_to_policy_idx(mid)
                    cfg = MOTOR_CONFIG[mid]
                    tgt = float(actions[idx]) if mid in active_ids else float(joint_pos[idx])
                    kp  = cfg["kp"]
                    kd  = cfg["kd"]
                    cap = args.torque_cap if args.torque_cap > 0 else cfg.get("torque_cap", 0.0)
                    if cap > 0:
                        cur     = float(joint_pos[idx])
                        vel     = float(joint_vel[idx])
                        # kd 項也計入總扭力預算：(cap - kd*|vel|) 剩餘給 kp 項
                        kd_tau  = kd * abs(vel)
                        kp_budget = max(0.0, cap - kd_tau)
                        max_err = kp_budget / kp if kp > 0 else 0.0
                        tgt     = cur + float(np.clip(tgt - cur, -max_err, max_err))
                    send_cmd(driver, mid, tgt, kp, kd)
            elif args.dry_run:
                joint_pos[:] = actions

            # 記錄資料
            if record_secs > 0 or True:
                for mid in motor_ids:
                    idx    = motor_id_to_policy_idx(mid)
                    cfg    = MOTOR_CONFIG[mid]
                    tgt    = float(actions[idx])
                    cur    = float(joint_pos[idx])
                    vel    = float(joint_vel[idx])
                    max_t  = MAX_TORQUE[cfg["type"]]
                    tau    = calc_torque(tgt, cur, vel, cfg["kp"], cfg["kd"], max_t)
                    rec_tgt[mid].append(math.degrees(tgt))
                    rec_tau[mid].append(abs(tau))
                # CSV row for replay
                if record_secs > 0:
                    row = {"time_s": f"{sim_t:.4f}"}
                    for name in RECORDING_JOINT_NAMES:
                        pol_i = POLICY_JOINT_NAMES.index(name)
                        row[f"target_{name}"] = f"{float(actions[pol_i]):.6f}"
                        row[f"pos_{name}"]    = f"{float(joint_pos[pol_i]):.6f}"
                        row[f"vel_{name}"]    = "0.000000"
                    rec_csv_rows.append(row)
                if bridge is not None:
                    with bridge._imu_lock:
                        _acc  = bridge.IMU_STATE["acc"].copy()
                        _gyro = bridge.IMU_STATE["gyro"].copy()
                        _quat = bridge.IMU_STATE["quat"].copy()
                    rec_acc.append(_acc.copy())
                    rec_gyro.append(_gyro.copy())
                    rec_pg.append(bridge.proj_gravity_from_quat(*_quat))

            if step_cnt % 50 == 0:
                overload_flags = []
                print(f"\n[t={sim_t:6.2f}s  step={step_cnt}]")
                if bridge is not None and rec_pg:
                    _pg = rec_pg[-1]
                    _acc = rec_acc[-1]
                    _gyro = rec_gyro[-1]
                    print(f"  IMU acc=[{_acc[0]:+.3f} {_acc[1]:+.3f} {_acc[2]:+.3f}]m/s²  "
                          f"gyro=[{_gyro[0]:+.3f} {_gyro[1]:+.3f} {_gyro[2]:+.3f}]rad/s  "
                          f"pg=[{_pg[0]:+.3f} {_pg[1]:+.3f} {_pg[2]:+.3f}]")
                for mid in motor_ids:
                    idx    = motor_id_to_policy_idx(mid)
                    cfg    = MOTOR_CONFIG[mid]
                    cur    = joint_pos[idx]
                    vel    = joint_vel[idx]
                    tgt    = float(actions[idx])
                    max_t  = MAX_TORQUE[cfg["type"]]
                    torque = calc_torque(tgt, cur, vel, cfg["kp"], cfg["kd"], max_t)
                    pct    = abs(torque) / max_t * 100
                    over   = pct >= args.torque_limit * 100
                    flag   = " !OVER" if over else ""
                    hold   = "" if mid in set(active_ids) else " [hold]"
                    if over:
                        overload_flags.append(cfg["name"].replace("dof_", ""))
                    print(f"  Actuator {mid:2d} ({_MID_CAN.get(mid,'?')}):"
                          f"  pos={cur:+7.3f}rad ({math.degrees(cur):+6.1f}°)"
                          f"  vel={vel:+6.3f}"
                          f"  torque={torque:+7.2f}Nm ({pct:4.1f}%)"
                          f"  tgt={tgt:+7.3f}rad ({math.degrees(tgt):+6.1f}°){flag}{hold}")
                if overload_flags:
                    _warn(f"扭矩過載: {overload_flags}")

            sim_t    += ctrl_dt
            step_cnt += 1
            slp = ctrl_dt - (time.time() - t0)
            if slp > 0:
                time.sleep(slp)

            if record_secs > 0 and sim_t >= record_secs:
                print(f"\n[記錄完成 {record_secs}s]")
                break

    except KeyboardInterrupt:
        print("\n停止")

    # ── 統計分析輸出 ────────────────────────────────────────────────────────────
    if rec_tgt[motor_ids[0]]:
        _banner("推論統計分析")
        print(f"\n  {'關節':<28} {'目標min':>8} {'目標max':>8} {'最大扭矩':>10} {'占比':>7}  {'超上限?':>6}")
        print(f"  {'-'*70}")
        for mid in motor_ids:
            cfg    = MOTOR_CONFIG[mid]
            name   = cfg["name"].replace("dof_", "")
            max_t  = MAX_TORQUE[cfg["type"]]
            cap    = args.torque_cap if args.torque_cap > 0 else cfg.get("torque_cap", 0.0)
            tmin   = min(rec_tgt[mid])
            tmax   = max(rec_tgt[mid])
            pk_tau = max(rec_tau[mid])
            pct    = pk_tau / max_t * 100
            safe_min_d = math.degrees(SAFE_MIN.get(cfg["name"], -math.pi*2))
            safe_max_d = math.degrees(SAFE_MAX.get(cfg["name"],  math.pi*2))
            pos_warn = " !" if (tmin < safe_min_d or tmax > safe_max_d) else "  "
            cap_warn = " !" if (cap > 0 and pk_tau > cap) else "  "
            print(f"  {name:<28} {tmin:>+7.1f}° {tmax:>+7.1f}°  {pk_tau:>7.2f}Nm  {pct:>5.1f}%  {pos_warn}{cap_warn}")
        if rec_pg:
            pg_arr = np.array(rec_pg)
            print(f"\n  ── IMU projected_gravity 範圍 ──")
            print(f"  X: [{pg_arr[:,0].min():+.4f}, {pg_arr[:,0].max():+.4f}]")
            print(f"  Y: [{pg_arr[:,1].min():+.4f}, {pg_arr[:,1].max():+.4f}]")
            print(f"  Z: [{pg_arr[:,2].min():+.4f}, {pg_arr[:,2].max():+.4f}]")
            pg_mean = pg_arr.mean(axis=0)
            print(f"  平均: [{pg_mean[0]:+.4f} {pg_mean[1]:+.4f} {pg_mean[2]:+.4f}]")
            if pg_mean[2] > 0.5:
                _warn("pg Z 平均為正值（+1方向）；訓練時直立應為 -1。建議確認 IMU 安裝方向是否需要翻轉。")
        if rec_acc:
            acc_arr = np.array(rec_acc)
            print(f"\n  ── IMU 加速度範圍（m/s²）──")
            for i, ax in enumerate("XYZ"):
                print(f"  {ax}: [{acc_arr[:,i].min():+.3f}, {acc_arr[:,i].max():+.3f}]  mean={acc_arr[:,i].mean():+.3f}")
        if rec_gyro:
            gyro_arr = np.array(rec_gyro)
            print(f"\n  ── IMU 角速度範圍（rad/s）──")
            for i, ax in enumerate("XYZ"):
                print(f"  {ax}: [{gyro_arr[:,i].min():+.3f}, {gyro_arr[:,i].max():+.3f}]  mean={gyro_arr[:,i].mean():+.3f}")

    # 儲存 CSV 供 replay
    if rec_csv_rows:
        import datetime
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = RECORDINGS_DIR / f"{ts}_actions.csv"
        fieldnames = ["time_s"] + [f"target_{n}" for n in RECORDING_JOINT_NAMES] \
                                 + [f"pos_{n}"    for n in RECORDING_JOINT_NAMES] \
                                 + [f"vel_{n}"    for n in RECORDING_JOINT_NAMES]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rec_csv_rows)
        _ok(f"已儲存 {len(rec_csv_rows)} 幀到 {csv_path}")
        print(f"\n  replay 指令（--no-policy 直接送錄製目標）：")
        print(f"  python linux/test/test_policy.py --mode replay --no-policy "
              f"--recording {csv_path} --ids {','.join(str(m) for m in motor_ids)} "
              f"--left-can can1\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 模式 2：zero  — 所有馬達歸零位（上電安全確認）
# ═══════════════════════════════════════════════════════════════════════════════

def run_zero(args, motor_ids: list, driver):
    """將所有指定馬達保持在零位（組裝確認、上電安全測試）。"""
    ctrl_dt   = 0.02
    step_cnt  = 0
    id_to_idx = {mid: motor_id_to_policy_idx(mid) for mid in motor_ids}
    joint_pos = np.zeros(20, dtype=np.float32)
    joint_vel = np.zeros(20, dtype=np.float32)

    _banner("零位模式 — 所有馬達保持 0°")
    _info(f"{'DRY RUN（不送指令）' if args.dry_run else 'LIVE CAN — 馬達上電中'}")
    _info("Ctrl+C 停止")
    _section("腿部關節安全限制")
    _print_joint_limits_table()

    # 先緩移到零位，避免從非零位突然跳到 0°
    if driver and not args.skip_home_ramp:
        _warn("zero 模式：先緩移到 0°，確認機器人不會撞到東西")
        init_joint_pos_from_reader(motor_ids, joint_pos, joint_vel, id_to_idx)
        # 臨時用 _ZEROS_DEG 全設 0 的版本做 ramp
        _orig = dict(_ZEROS_DEG)
        for k in _ZEROS_DEG:
            _ZEROS_DEG[k] = 0.0
        home_ramp(motor_ids, driver, id_to_idx, joint_pos, joint_vel, 0.05)
        _ZEROS_DEG.update(_orig)

    try:
        while True:
            t0 = time.time()
            if driver:
                read_states(driver, motor_ids, joint_pos, joint_vel, id_to_idx)
            if not args.dry_run and driver:
                for mid in motor_ids:
                    send_cmd(driver, mid, 0.0, MOTOR_CONFIG[mid]["kp"], MOTOR_CONFIG[mid]["kd"])

            if step_cnt % 50 == 0:
                print(f"[t={step_cnt*ctrl_dt:6.1f}s]")
                for mid in motor_ids:
                    idx    = id_to_idx[mid]
                    cfg    = MOTOR_CONFIG[mid]
                    name   = cfg["name"].replace("dof_", "")
                    cur    = math.degrees(joint_pos[idx])
                    max_t  = MAX_TORQUE[cfg["type"]]
                    torque = calc_torque(0.0, joint_pos[idx], joint_vel[idx],
                                        cfg["kp"], cfg["kd"], max_t)
                    flag   = " !!OVERLOAD" if abs(torque) >= max_t * 0.95 else ""
                    print(f"  {name:<28}  cur={cur:6.1f}°  tgt=  0.0°  "
                          f"τ={torque:6.1f}/{max_t:.0f}Nm{flag}")

            step_cnt += 1
            slp = ctrl_dt - (time.time() - t0)
            if slp > 0:
                time.sleep(slp)
    except KeyboardInterrupt:
        print("\n停止")


# ═══════════════════════════════════════════════════════════════════════════════
# 模式 3：sine  — 正弦波運動（確認馬達響應，與舊版實測相同參數）
# ═══════════════════════════════════════════════════════════════════════════════

def run_stand(args, motor_ids: list, driver):
    """移動到直立站姿（ZEROS）並保持。
    測試個別關節前的標準起始點，相當於 firmware Home state 到達後的狀態。
    """
    ctrl_dt   = 0.02
    step_cnt  = 0
    id_to_idx = {mid: motor_id_to_policy_idx(mid) for mid in motor_ids}
    joint_pos = np.zeros(20, dtype=np.float32)
    joint_vel = np.zeros(20, dtype=np.float32)

    _banner("站姿保持模式 — 移動到 ZEROS 直立站姿")
    _info(f"{'DRY RUN（不送指令）' if args.dry_run else 'LIVE CAN — 馬達上電中'}")
    _info("目標：ZEROS 直立站姿（膝彎 ±50°，髖 ±20°，踝 ±30°）")
    _info("Ctrl+C 停止")
    print("  ── 各關節目標位置 ──")
    for mid in motor_ids:
        name = MOTOR_CONFIG[mid]["name"]
        tgt  = _ZEROS_DEG.get(name, 0.0)
        print(f"    {name.replace('dof_',''):<28}  tgt={tgt:+6.1f}°")
    print()

    if driver and not args.skip_home_ramp:
        init_joint_pos_from_reader(motor_ids, joint_pos, joint_vel, id_to_idx)
        home_ramp(motor_ids, driver, id_to_idx, joint_pos, joint_vel, 0.05)

    try:
        while True:
            t0 = time.time()
            if driver:
                read_states(driver, motor_ids, joint_pos, joint_vel, id_to_idx)
            if not args.dry_run and driver:
                for mid in motor_ids:
                    name = MOTOR_CONFIG[mid]["name"]
                    tgt  = math.radians(_ZEROS_DEG.get(name, 0.0))
                    send_cmd(driver, mid, tgt, MOTOR_CONFIG[mid]["kp"], MOTOR_CONFIG[mid]["kd"])

            if step_cnt % 50 == 0:
                print(f"[t={step_cnt*ctrl_dt:6.1f}s]")
                for mid in motor_ids:
                    idx   = id_to_idx[mid]
                    cfg   = MOTOR_CONFIG[mid]
                    name  = cfg["name"]
                    tgt_d = _ZEROS_DEG.get(name, 0.0)
                    cur   = math.degrees(joint_pos[idx])
                    err   = abs(tgt_d - cur)
                    max_t = MAX_TORQUE[cfg["type"]]
                    torque = calc_torque(math.radians(tgt_d), joint_pos[idx],
                                        joint_vel[idx], cfg["kp"], cfg["kd"], max_t)
                    flag  = " !!OVERLOAD" if abs(torque) >= max_t * 0.95 else ""
                    print(f"  {name.replace('dof_',''):<28}  cur={cur:+6.1f}°  "
                          f"tgt={tgt_d:+6.1f}°  err={err:4.1f}°  τ={torque:6.1f}/{max_t:.0f}Nm{flag}")

            step_cnt += 1
            slp = ctrl_dt - (time.time() - t0)
            if slp > 0:
                time.sleep(slp)
    except KeyboardInterrupt:
        print("\n停止")


def run_sine(args, motor_ids: list, active_ids: list, driver):
    """正弦波運動（以 ZEROS 站姿為中心，非馬達機械零點）。
    --sine-freq 控制頻率（Hz），自動計算預估峰值扭矩並警告是否超限。
    非 active_ids 的馬達保持 ZEROS 站姿位置。
    """
    freq_hz   = getattr(args, "sine_freq", SINE_FREQ_HZ)
    ctrl_dt   = 0.02
    step_cnt  = 0
    omega     = 2 * math.pi * freq_hz
    id_to_idx = {mid: motor_id_to_policy_idx(mid) for mid in motor_ids}
    joint_pos = np.zeros(20, dtype=np.float32)
    joint_vel = np.zeros(20, dtype=np.float32)

    _zeros_rad = {mid: math.radians(_ZEROS_DEG.get(MOTOR_CONFIG[mid]["name"], 0.0))
                  for mid in motor_ids}

    active_set = set(active_ids)
    _banner(f"正弦波模式 — {freq_hz:.2f} Hz，中心=ZEROS 站姿")
    _info(f"{'DRY RUN（不送指令）' if args.dry_run else 'LIVE CAN — 馬達上電中'}")
    _info(f"頻率: {freq_hz:.2f} Hz   振幅: type04=±15°  type03=±10°  type02=±8°")
    _info("中心：ZEROS 站姿（不是馬達機械 0°）")
    _info("Ctrl+C 停止")
    if active_set != set(motor_ids):
        active_names = [MOTOR_CONFIG[m]["name"].replace("dof_","") for m in active_ids]
        print(f"  [限制] 只有 {active_ids} ({active_names}) 做正弦波，其餘保持 ZEROS 站姿")

    # 預估峰值扭矩表
    print(f"\n  ── 各關節 sine 實際範圍 & 預估峰值扭矩（{freq_hz:.2f} Hz）──")
    print(f"  {'關節':<26} {'ZEROS':>6} {'最小':>8} {'最大':>8}  {'峰值扭矩':>10}  {'佔比':>6}")
    over_torque = []
    for mid in motor_ids:
        cfg   = MOTOR_CONFIG[mid]
        name  = cfg["name"].replace("dof_", "")
        z     = _ZEROS_DEG.get(cfg["name"], 0.0)
        amp   = math.degrees(SINE_AMP_RAD[cfg["type"]])
        tag   = "" if mid in active_set else " [hold]"
        if mid in active_set:
            peak, ratio = _sine_peak_torque(mid, freq_hz)
            warn = " !!OVER" if ratio >= args.torque_limit else ""
            if ratio >= args.torque_limit:
                over_torque.append((mid, name, ratio))
            torque_str = f"{peak:5.1f}/{MAX_TORQUE[cfg['type']]:.0f}Nm  {ratio*100:4.0f}%{warn}"
        else:
            torque_str = f"{'[hold]':>17}"
        print(f"  {name:<26} {z:>+6.0f}°  {z-amp:>+7.1f}°  {z+amp:>+7.1f}°  {torque_str}{tag}")
    if over_torque:
        _warn(f"以下關節預估扭矩超過 {args.torque_limit*100:.0f}% 上限，建議降低頻率：")
        for mid, name, ratio in over_torque:
            _warn(f"  {name}  {ratio*100:.0f}%")
        # 計算不超限的最大頻率
        max_safe_freq = min(
            (args.torque_limit * MAX_TORQUE[MOTOR_CONFIG[m]["type"]] /
             (MOTOR_CONFIG[m]["kd"] * SINE_AMP_RAD[MOTOR_CONFIG[m]["type"]] * 2 * math.pi))
            for m, _, _ in over_torque
        )
        _warn(f"建議最大頻率: {max_safe_freq:.2f} Hz")

    # 先移到站姿，再開始 sine
    if driver and not args.skip_home_ramp:
        init_joint_pos_from_reader(motor_ids, joint_pos, joint_vel, id_to_idx)
        home_ramp(motor_ids, driver, id_to_idx, joint_pos, joint_vel, 0.05)
        input("\n  [確認] 已到站姿，按 Enter 開始 sine 測試...")

    try:
        while True:
            t0    = time.time()
            sim_t = step_cnt * ctrl_dt

            if driver:
                read_states(driver, motor_ids, joint_pos, joint_vel, id_to_idx)

            for mid in motor_ids:
                cfg    = MOTOR_CONFIG[mid]
                zero   = _zeros_rad[mid]
                if mid in active_set:
                    amp    = SINE_AMP_RAD[cfg["type"]]
                    target = zero + amp * math.sin(omega * sim_t)  # 以 ZEROS 為中心
                else:
                    target = zero                                   # 非 active → 保持站姿
                if not args.dry_run and driver:
                    send_cmd(driver, mid, target, cfg["kp"], cfg["kd"])

            if step_cnt % 50 == 0:
                print(f"[t={sim_t:6.1f}s]")
                for mid in motor_ids:
                    idx    = id_to_idx[mid]
                    cfg    = MOTOR_CONFIG[mid]
                    name   = cfg["name"].replace("dof_", "")
                    zero   = _zeros_rad[mid]
                    if mid in active_set:
                        amp = SINE_AMP_RAD[cfg["type"]]
                        tgt = zero + amp * math.sin(omega * sim_t)
                    else:
                        tgt = zero
                    cur    = joint_pos[idx]
                    max_t  = MAX_TORQUE[cfg["type"]]
                    torque = calc_torque(tgt, cur, joint_vel[idx],
                                        cfg["kp"], cfg["kd"], max_t)
                    flag   = " !!OVERLOAD" if abs(torque) >= max_t * 0.95 else ""
                    tag    = "" if mid in active_set else " [hold]"
                    print(f"  {name:<28}  tgt={math.degrees(tgt):+6.1f}°  "
                          f"cur={math.degrees(cur):+6.1f}°  "
                          f"τ={torque:6.1f}/{max_t:.0f}Nm{flag}{tag}")

            step_cnt += 1
            slp = ctrl_dt - (time.time() - t0)
            if slp > 0:
                time.sleep(slp)
    except KeyboardInterrupt:
        print("\n停止")


# ═══════════════════════════════════════════════════════════════════════════════
# 模式 4：replay  — 從錄製 CSV 重播並驗證
# ═══════════════════════════════════════════════════════════════════════════════

def _latest_recording() -> Path | None:
    """回傳 recordings/ 目錄中最新的 _actions.csv。"""
    if not RECORDINGS_DIR.exists():
        return None
    csvs = sorted(RECORDINGS_DIR.glob("*_actions.csv"))
    return csvs[-1] if csvs else None


def _replay_estop(reason: str, motor_ids: list, driver, id_to_idx: dict,
                  joint_pos: np.ndarray, joint_vel: np.ndarray):
    """緊急停止：印錯誤訊息並緩移回 ZEROS。"""
    _fail(f"緊急停止（E-STOP）: {reason}")
    _warn("正在緩移回站姿 ZEROS，請勿拔電…")
    if driver:
        try:
            home_ramp(motor_ids, driver, id_to_idx, joint_pos, joint_vel,
                      torque_limit_ratio=0.1)
        except Exception as e:
            _warn(f"home_ramp 失敗: {e}")
    _fail("已停止，請檢查機器人狀態後再繼續。")


def run_replay(args, motor_ids: list, active_ids: list, driver, bridge):
    # 選擇錄製檔
    rec_path = Path(args.recording) if args.recording else _latest_recording()
    if rec_path is None or not rec_path.exists():
        print(f"[ERROR] 找不到錄製 CSV（{rec_path}）")
        print(f"  錄製目錄: {RECORDINGS_DIR}")
        avail = sorted(RECORDINGS_DIR.glob("*_actions.csv")) if RECORDINGS_DIR.exists() else []
        if avail:
            print("  可用錄製：")
            for p in avail[-5:]:
                print(f"    {p}")
        sys.exit(1)

    active_set = set(active_ids)
    _banner("Replay 模式 — 從錄製 CSV 重播")
    _info(f"錄製: {rec_path.name}")
    _info(f"完整路徑: {rec_path}")
    if active_set != set(motor_ids):
        active_names = [MOTOR_CONFIG[m]["name"].replace("dof_","") for m in active_ids]
        _warn(f"只有 {active_ids} ({active_names}) 接收錄製指令，其餘保持零位")

    # 載入 CSV
    with open(rec_path, newline="") as f:
        rows = list(csv.DictReader(f))
    _ok(f"載入 {len(rows)} 幀，時長 {float(rows[-1]['time_s']):.2f}s")

    # 速度縮放：每幀重複送 n_reps 次，降低有效關節速度
    replay_speed = max(0.1, min(1.0, getattr(args, "replay_speed", 1.0)))
    n_reps = max(1, round(1.0 / replay_speed))
    if n_reps > 1:
        _info(f"速度縮放 {replay_speed:.0%}：每幀重複 {n_reps} 次（有效速度降為 {1/n_reps:.0%}）")

    # 保護閾值
    estop_vel     = getattr(args, "estop_vel",     20.0)   # rad/s
    estop_pos_err = getattr(args, "estop_pos_err", 15.0)   # degrees
    estop_torque  = getattr(args, "estop_torque",  0.85)   # fraction of hw_max
    CONSEC_LIMIT  = 3                                       # 連續超扭矩幀才觸發

    _section("保護機制設定")
    use_cap = args.torque_cap > 0 or any(c.get("torque_cap", 0) > 0 for c in MOTOR_CONFIG.values())
    _info(f"Torque cap  : {'啟用（args.torque_cap 或 MOTOR_CONFIG.torque_cap）' if use_cap else '未啟用（可加 --torque-cap <Nm>）'}")
    _info(f"E-STOP vel  : {estop_vel:.1f} rad/s （--estop-vel）")
    _info(f"E-STOP Δpos : {estop_pos_err:.1f} °  （--estop-pos-err，幀 5 後生效）")
    _info(f"E-STOP τ    : {estop_torque*100:.0f}% × hw_max 連續 {CONSEC_LIMIT} 幀（--estop-torque）")

    # 載入 policy（--no-policy 時跳過）
    if args.no_policy:
        _info("模式：CSV 直播（--no-policy），直接送錄製 target，不跑 policy 推論")
        step_sess = num_commands = carry = None
    else:
        _section("載入策略")
        _info(f"檔案: {Path(args.policy).name}")
        init_sess, step_sess, meta = load_kinfer(args.policy)
        num_commands = meta.get("num_commands", 0) or 0
        carry_init   = init_sess.run(None, {})
        carry_size   = meta["carry_size"]
        carry        = carry_init[0] if carry_init else np.zeros(carry_size, dtype=np.float32)

    id_to_idx  = {mid: motor_id_to_policy_idx(mid) for mid in motor_ids}
    ctrl_dt    = 0.02
    sim_t      = 0.0

    # Home ramp：先移到 ZEROS 站姿，再緩移到 CSV 第一幀位置
    if driver and not args.skip_home_ramp:
        _ramp_pos = np.zeros(20, dtype=np.float32)
        _ramp_vel = np.zeros(20, dtype=np.float32)
        # 第一段：移到 ZEROS 站姿
        home_ramp(motor_ids, driver, id_to_idx, _ramp_pos, _ramp_vel, 0.05)
        # 第二段：從站姿緩移到 CSV 第一幀（消除啟動衝擊）
        first_row = rows[0]
        csv_frame0 = {}
        for mid in motor_ids:
            col = f"target_{MOTOR_CONFIG[mid]['name']}"
            if col in first_row:
                csv_frame0[mid] = float(first_row[col])
        if csv_frame0:
            _info("銜接 Ramp：從站姿移到 CSV 起始姿態（2秒）...")
            t_start   = time.time()
            ramp_dur  = 2.0
            read_states(driver, motor_ids, _ramp_pos, _ramp_vel, id_to_idx)
            start_pos = {mid: float(_ramp_pos[id_to_idx[mid]]) for mid in motor_ids}
            while True:
                alpha = min((time.time() - t_start) / ramp_dur, 1.0)
                for mid in motor_ids:
                    if mid in csv_frame0:
                        tgt = start_pos[mid] + alpha * (csv_frame0[mid] - start_pos[mid])
                        cfg = MOTOR_CONFIG[mid]
                        send_cmd(driver, mid, tgt, cfg["kp"], cfg["kd"])
                if alpha >= 1.0:
                    break
                time.sleep(0.02)
        input("\n  [確認] 已到達 CSV 起始姿態，按 Enter 開始 Replay...")

    errors        = []   # per-leg per-frame |policy_output - recorded_target|
    torque_ratios = []   # per-leg per-frame estimated torque ratio
    positions     = []   # per-leg per-frame commanded target position (deg)
    oob_cnt       = 0
    nan_cnt       = 0
    tlimit        = args.torque_limit
    consec_over   = 0    # 連續超扭矩幀計數

    # 列印間隔：live 模式更密集
    print_every = 10 if (driver and not args.dry_run) else 50

    _section(f"重播中（{'DRY RUN' if args.dry_run else 'LIVE CAN'}  速度={replay_speed:.0%}）")
    print(f"  {'幀':>5}  {'時間':>6}  {'τmax%':>7}  E-STOP 速度/位置誤差監控中")

    joint_pos = np.zeros(20, dtype=np.float32)
    joint_vel = np.zeros(20, dtype=np.float32)

    # 即時突波過濾：單幀突波自動平滑，連續突波觸發 E-STOP
    _spike_filter = SpikeFilter(
        n_joints=20,
        max_jump_deg=35.0,
        max_consecutive=5,
        joint_names=[MOTOR_CONFIG[mid]["name"] if mid in MOTOR_CONFIG else f"j{i}"
                     for i, mid in enumerate(sorted(MOTOR_CONFIG))],
    )

    for row_i, row in enumerate(rows):

        # 從錄製讀取關節狀態（用於 dry-run 及初始值）
        rec_targets_20 = np.zeros(20, dtype=np.float32)
        for rec_i, name in enumerate(RECORDING_JOINT_NAMES):
            pol_i = _REC_TO_POL[rec_i]
            joint_pos[pol_i]      = float(row.get(f"pos_{name}", 0))
            joint_vel[pol_i]      = float(row.get(f"vel_{name}", 0))
            rec_targets_20[pol_i] = float(row.get(f"target_{name}", 0))

        # 預讀下一幀（用於幀內插值，消除幀間跳變）
        if row_i + 1 < len(rows):
            next_row = rows[row_i + 1]
            next_rec = np.zeros(20, dtype=np.float32)
            for rec_i, name in enumerate(RECORDING_JOINT_NAMES):
                pol_i = _REC_TO_POL[rec_i]
                next_rec[pol_i] = float(next_row.get(f"target_{name}", 0))
            next_clipped = np.clip(next_rec, _SAFE_MIN_ARR, _SAFE_MAX_ARR)
        else:
            next_clipped = None

        # ── 計算本幀目標（只算一次，重複送 n_reps 次）────────────────────────
        if args.no_policy:
            actions = rec_targets_20
            clipped = np.clip(actions, _SAFE_MIN_ARR, _SAFE_MAX_ARR)
        else:
            feed    = build_policy_feed(step_sess, joint_pos, joint_vel, carry,
                                        num_commands, sim_t, bridge)
            outputs = step_sess.run(None, feed)
            actions = expand_actions(outputs[0], joint_pos)
            carry   = outputs[1]
            if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
                nan_cnt += 1
            clipped = np.clip(actions, _SAFE_MIN_ARR, _SAFE_MAX_ARR)
            if not np.allclose(actions, clipped, atol=1e-6):
                oob_cnt += 1

        # ── 突波過濾（單幀平滑，連續突波 E-STOP）──────────────────────────
        clipped, spike_abort = _spike_filter.update(clipped)
        if spike_abort:
            _replay_estop(
                f"突波過濾器中止：{spike_abort}",
                motor_ids, driver, id_to_idx, joint_pos, joint_vel)
            return

        # ── 重複送指令（速度縮放 + 幀間線性插值）────────────────────────────
        for rep in range(n_reps):
            # 幀內插值：alpha 從 0 線性到 1，消除幀間跳變
            if next_clipped is not None and n_reps > 1:
                alpha = rep / n_reps
                interp_clipped = clipped + alpha * (next_clipped - clipped)
            else:
                interp_clipped = clipped
            t0 = time.time()

            # 讀取硬體回饋（每次重複都重讀）
            if driver:
                read_states(driver, motor_ids, joint_pos, joint_vel, id_to_idx)

            # ── E-STOP 1：速度過高（前 3 幀跳過，等馬達從無控切換穩定）────────
            if driver and row_i >= 3:
                for mid in motor_ids:
                    idx = motor_id_to_policy_idx(mid)
                    v   = abs(float(joint_vel[idx]))
                    if v > estop_vel:
                        _replay_estop(
                            f"Actuator {mid}（{MOTOR_CONFIG[mid]['name']}）"
                            f" 速度 {v:.2f} rad/s > 閾值 {estop_vel} rad/s",
                            motor_ids, driver, id_to_idx, joint_pos, joint_vel)
                        return

            # ── E-STOP 2：位置偏差過大（幀 5 後生效，讓馬達先追上 ZEROS）────
            # 用插值目標（interp_clipped）比較，避免插值期間的誤報
            if driver and row_i >= 5:
                for mid in active_ids:
                    idx     = motor_id_to_policy_idx(mid)
                    err_deg = abs(math.degrees(float(joint_pos[idx]))
                                  - math.degrees(float(interp_clipped[idx])))
                    if err_deg > estop_pos_err:
                        _replay_estop(
                            f"Actuator {mid}（{MOTOR_CONFIG[mid]['name']}）"
                            f" 位置偏差 {err_deg:.1f}° > 閾值 {estop_pos_err}°",
                            motor_ids, driver, id_to_idx, joint_pos, joint_vel)
                        return

            # ── 送馬達（含 torque cap）───────────────────────────────────────
            frame_torque_over = False
            if not args.dry_run and driver:
                for mid in motor_ids:
                    idx = motor_id_to_policy_idx(mid)
                    cfg = MOTOR_CONFIG[mid]
                    pos = float(interp_clipped[idx]) if mid in active_set else \
                          math.radians(_ZEROS_DEG.get(cfg["name"], 0.0))

                    # kd-aware torque cap（與 policy 模式相同邏輯）
                    cap = args.torque_cap if args.torque_cap > 0 else cfg.get("torque_cap", 0.0)
                    if cap > 0:
                        cur      = float(joint_pos[idx])
                        vel      = float(joint_vel[idx])
                        kd_tau   = cfg["kd"] * abs(vel)
                        kp_bdgt  = max(0.0, cap - kd_tau)
                        max_err  = kp_bdgt / cfg["kp"] if cfg["kp"] > 0 else 0.0
                        pos      = cur + float(np.clip(pos - cur, -max_err, max_err))

                    # E-STOP 3 預檢：估算送出後扭矩
                    hw_max = MAX_TORQUE[cfg["type"]]
                    tau    = calc_torque(pos, float(joint_pos[idx]),
                                         float(joint_vel[idx]),
                                         cfg["kp"], cfg["kd"], hw_max)
                    if abs(tau) / hw_max > estop_torque:
                        frame_torque_over = True

                    send_cmd(driver, mid, pos, cfg["kp"], cfg["kd"])

                # 連續超扭矩幀計數
                if frame_torque_over:
                    consec_over += 1
                else:
                    consec_over = 0

                if consec_over >= CONSEC_LIMIT:
                    _replay_estop(
                        f"連續 {CONSEC_LIMIT} 幀估算扭矩 > {estop_torque*100:.0f}% hw_max",
                        motor_ids, driver, id_to_idx, joint_pos, joint_vel)
                    return

            slp = ctrl_dt - (time.time() - t0)
            if slp > 0:
                time.sleep(slp)

        # ── 統計（每幀只算一次）──────────────────────────────────────────────
        frame_err = []
        frame_tau = []
        for rec_i, name in enumerate(RECORDING_JOINT_NAMES):
            pol_i      = _REC_TO_POL[rec_i]
            frame_err.append(abs(float(actions[pol_i]) - float(rec_targets_20[pol_i])))
            mid_for_name = next((m for m, c in MOTOR_CONFIG.items() if c["name"] == name), None)
            if mid_for_name:
                frame_tau.append(torque_ratio(float(actions[pol_i]), joint_pos[pol_i],
                                              joint_vel[pol_i], mid_for_name))
            else:
                frame_tau.append(0.0)
        errors.append(frame_err)
        torque_ratios.append(frame_tau)
        positions.append([math.degrees(float(actions[_REC_TO_POL[i]]))
                          for i in range(len(RECORDING_JOINT_NAMES))])

        # ── 狀態列印 ─────────────────────────────────────────────────────────
        if row_i % print_every == 0:
            max_tau  = max(frame_tau) if frame_tau else 0.0
            over_tag = f" [OVER {max_tau*100:.0f}%]" if max_tau > tlimit else ""
            print(f"\n  [frame {row_i:4d}/{len(rows)}  t={sim_t:6.2f}s  τmax={max_tau*100:.0f}%{over_tag}]")
            for mid in motor_ids:
                idx  = motor_id_to_policy_idx(mid)
                cfg  = MOTOR_CONFIG[mid]
                cur  = joint_pos[idx]
                vel  = joint_vel[idx]
                hw_max_disp = MAX_TORQUE[cfg["type"]]
                # 用已套 cap + 插值的目標計算扭矩（與實際送出一致）
                tgt_raw = float(interp_clipped[idx])
                cap_disp = args.torque_cap if args.torque_cap > 0 else cfg.get("torque_cap", 0.0)
                if cap_disp > 0:
                    kd_tau_d  = cfg["kd"] * abs(float(vel))
                    kp_bdgt_d = max(0.0, cap_disp - kd_tau_d)
                    max_err_d = kp_bdgt_d / cfg["kp"] if cfg["kp"] > 0 else 0.0
                    tgt = float(cur) + float(np.clip(tgt_raw - float(cur), -max_err_d, max_err_d))
                else:
                    tgt = tgt_raw
                tau  = calc_torque(tgt, cur, vel, cfg["kp"], cfg["kd"], hw_max_disp)
                pct  = abs(tau) / hw_max_disp * 100
                flag = " !OVER" if pct >= tlimit * 100 else ""
                print(f"  Actuator {mid:2d} ({_MID_CAN.get(mid,'?')}):"
                      f"  pos={cur:+7.3f}rad ({math.degrees(cur):+6.1f}°)"
                      f"  vel={vel:+6.3f}"
                      f"  torque={tau:+7.2f}Nm ({pct:4.1f}%)"
                      f"  tgt={tgt:+7.3f}rad ({math.degrees(tgt):+6.1f}°){flag}")

        sim_t += ctrl_dt

    # ── 摘要報告 ────────────────────────────────────────────────────────────
    errors_arr = np.array(errors)   # (N_frames, 10_legs)
    _banner("Replay 摘要報告")
    _info(f"總幀數  : {len(rows)}")
    if nan_cnt == 0:
        _ok(f"NaN 幀  : {nan_cnt}")
    else:
        _fail(f"NaN 幀  : {nan_cnt}")
    if oob_cnt == 0:
        _ok(f"超範圍幀: {oob_cnt}")
    else:
        _warn(f"超範圍幀: {oob_cnt}")

    tau_arr = np.array(torque_ratios)  # (N_frames, 10_legs)
    pos_arr = np.array(positions)      # (N_frames, 10_legs) in degrees

    _section(f"各馬達位置行程（{float(rows[-1]['time_s']):.1f}s 內）")
    print(f"  {'ID':>3}  {'關節':<26}  {'起始°':>7}  {'最小°':>7}  {'最大°':>7}  {'均值°':>7}  {'行程°':>7}")
    for i, name in enumerate(RECORDING_JOINT_NAMES):
        col     = pos_arr[:, i]
        mid     = next(m for m, c in MOTOR_CONFIG.items() if c["name"] == name)
        lo, hi  = SAFE_MIN[name], SAFE_MAX[name]
        start   = col[0]
        mn, mx, mean = col.min(), col.max(), col.mean()
        travel  = mx - mn
        # 超界標記
        oob_tag = ""
        if mn < math.degrees(lo) - 0.1 or mx > math.degrees(hi) + 0.1:
            oob_tag = " [OOB!]"
        print(f"  {mid:>3}  {name.replace('dof_',''):<26}  {start:>+7.1f}  "
              f"{mn:>+7.1f}  {mx:>+7.1f}  {mean:>+7.1f}  {travel:>7.1f}{oob_tag}")

    _section(f"估算扭矩比（限制 {tlimit*100:.0f}%）")
    print(f"  {'關節':<32}  {'均值%':>6}  {'最大%':>6}  {'P90%':>6}  狀態")
    any_over = False
    for i, name in enumerate(RECORDING_JOINT_NAMES):
        col   = tau_arr[:, i] * 100
        over  = col.max() > tlimit * 100
        if over:
            any_over = True
        status = f"[OVER {col.max():.0f}%]" if over else "[OK]  "
        print(f"  {name:<32}  {col.mean():6.1f}  {col.max():6.1f}  "
              f"{np.percentile(col, 90):6.1f}  {status}")
    print()
    if any_over:
        _warn(f"部分關節超過 {tlimit*100:.0f}% 扭矩限制，走路時建議降低策略輸出幅度或提高扭矩限制")
    else:
        _ok(f"所有關節扭矩均在 {tlimit*100:.0f}% 限制內")

    if not args.no_policy:
        _section("policy 輸出 vs 錄製 target 偏差（deg）")
        print(f"  {'關節':<32}  {'均值':>6}  {'最大':>6}  {'P90':>6}  狀態")
        for i, name in enumerate(RECORDING_JOINT_NAMES):
            col = errors_arr[:, i] * 180 / math.pi
            status = "[OK] " if col.max() < 5.0 else "[WARN]"
            print(f"  {name:<32}  {col.mean():6.2f}  {col.max():6.2f}  "
                  f"{np.percentile(col, 90):6.2f}  {status}")
        overall = errors_arr.flatten() * 180 / math.pi
        print()
        _info(f"整體均值誤差: {overall.mean():.2f}°  最大: {overall.max():.2f}°")

    passed = nan_cnt == 0 and oob_cnt == 0
    print()
    if passed:
        _ok("結果: PASS ✓")
    else:
        _warn("結果: WARN 需確認")

    # ── 正常完播後回 ZEROS（馬達停在最後幀位置可能是彎膝狀態）────────────────
    if driver and not args.dry_run:
        print()
        _warn("Replay 完成，馬達停在最後幀位置。自動緩移回站姿 ZEROS…")
        home_ramp(motor_ids, driver, id_to_idx, joint_pos, joint_vel,
                  torque_limit_ratio=0.05)


# ═══════════════════════════════════════════════════════════════════════════════
# 模式 3：check  — 快速自動驗證
# ═══════════════════════════════════════════════════════════════════════════════

def _print_joint_limits_table():
    """印出所有腿部關節的安全上下限表格。"""
    print(f"\n  {'關節':<30}  {'下限(°)':>8}  {'上限(°)':>8}  {'範圍(°)':>8}  {'零位(°)':>8}")
    print("  " + "-" * 68)
    for name in RECORDING_JOINT_NAMES:
        lo = math.degrees(SAFE_MIN[name])
        hi = math.degrees(SAFE_MAX[name])
        z  = _ZEROS_DEG[name]
        print(f"  {name:<30}  {lo:8.1f}  {hi:8.1f}  {hi-lo:8.1f}  {z:8.1f}")
    print()


def run_check(args, bridge):
    _banner(f"自動檢查模式 — {args.steps} 步推論")
    _info(f"策略: {Path(args.policy).name}")
    imu_str = "真實 H30 IMU" if bridge else "假 IMU（直立靜止）"
    _info(f"IMU : {imu_str}")

    # 印出關節安全限制表
    _section("腿部關節安全限制")
    _print_joint_limits_table()

    init_sess, step_sess, meta = load_kinfer(args.policy)
    num_commands = meta.get("num_commands", 0) or 0
    carry_init   = init_sess.run(None, {})
    carry_size   = meta["carry_size"]
    carry        = carry_init[0] if carry_init else np.zeros(carry_size, dtype=np.float32)

    # 從 ZEROS 站姿開始（模擬 home_ramp 已完成），讓扭矩檢查反映真實運行狀況
    joint_pos = np.zeros(20, dtype=np.float32)
    joint_vel = np.zeros(20, dtype=np.float32)
    _leg_mid_all = [31, 32, 33, 34, 35, 41, 42, 43, 44, 45]
    for mid in _leg_mid_all:
        idx = motor_id_to_policy_idx(mid)
        joint_pos[idx] = math.radians(_ZEROS_DEG.get(MOTOR_CONFIG[mid]["name"], 0.0))
    _info("起始姿態：ZEROS 站姿（模擬 home_ramp 完成後）")

    # 記錄每個腿部關節的 policy 輸出歷程
    _n_legs = len(RECORDING_JOINT_NAMES)
    history = []   # list of float32 arrays (20-dim)

    results = {
        "步數": args.steps,
        "NaN/Inf": 0,
        "超安全範圍": 0,
        "扭矩過載": 0,
        "錯誤": [],
    }

    # CAN ID → policy index，只取腿部馬達
    _leg_mid_order = _leg_mid_all
    _leg_idx = [motor_id_to_policy_idx(m) for m in _leg_mid_order]

    for step in range(args.steps):
        sim_t = step * 0.02
        feed  = build_policy_feed(step_sess, joint_pos, joint_vel, carry,
                                  num_commands, sim_t, bridge)
        try:
            outputs = step_sess.run(None, feed)
        except Exception as e:
            results["錯誤"].append(f"step {step}: {e}")
            continue

        actions = expand_actions(outputs[0], joint_pos)
        carry   = outputs[1]
        history.append(actions.copy())

        if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
            results["NaN/Inf"] += 1

        clipped = np.clip(actions, _SAFE_MIN_ARR, _SAFE_MAX_ARR)
        if not np.allclose(actions, clipped, atol=1e-6):
            results["超安全範圍"] += 1

        # 扭矩過載檢查（以 torque_limit 為閾值）
        for mid in _leg_mid_order:
            idx = motor_id_to_policy_idx(mid)
            tr  = torque_ratio(float(actions[idx]), joint_pos[idx], 0.0, mid)
            if tr >= args.torque_limit:
                results["扭矩過載"] += 1

        # 假設靜止：下一步 joint_pos 微移（模擬馬達跟隨）
        for i, name in enumerate(POLICY_JOINT_NAMES):
            if name in SAFE_MIN:
                joint_pos[i] += (float(actions[i]) - joint_pos[i]) * 0.1

        if (step + 1) % 25 == 0:
            range_ok = np.allclose(actions, clipped, atol=1e-6)
            vals = " ".join(f"{math.degrees(float(actions[i])):+5.1f}°" for i in _leg_idx[:5])
            imu_str = ""
            if bridge is not None:
                with bridge._imu_lock:
                    _pg = bridge.proj_gravity_from_quat(*bridge.IMU_STATE["quat"])
                imu_str = (f"  pg=[{_pg[0]:+.2f} {_pg[1]:+.2f} {_pg[2]:+.2f}]")
            range_tag = "[OK]  " if range_ok else "[WARN]"
            print(f"  step {step+1:4d}  {range_tag}  R_leg=[{vals}]{imu_str}")

    # ── 每個腿部關節的 policy 輸出統計 ────────────────────────────────────────
    if history:
        hist_arr = np.array(history)   # (steps, 20)
        print(f"\n  ── Policy 輸出統計（{len(history)} 步） ──")
        print(f"  {'關節':<30}  {'最小(°)':>8}  {'最大(°)':>8}  {'均值(°)':>8}  "
              f"{'安全下限':>8}  {'安全上限':>8}  {'狀態':>6}")
        print("  " + "-" * 88)
        any_oob = False
        for mid in _leg_mid_order:
            idx  = motor_id_to_policy_idx(mid)
            name = MOTOR_CONFIG[mid]["name"]
            col  = hist_arr[:, idx]
            vmin = math.degrees(col.min())
            vmax = math.degrees(col.max())
            vmean= math.degrees(col.mean())
            lo   = math.degrees(SAFE_MIN[name])
            hi   = math.degrees(SAFE_MAX[name])
            oob  = (vmin < lo - 0.1) or (vmax > hi + 0.1)
            if oob:
                any_oob = True
            status = "[OOB!]" if oob else "[OK]  "
            print(f"  {name:<30}  {vmin:8.1f}  {vmax:8.1f}  {vmean:8.1f}  "
                  f"{lo:8.1f}  {hi:8.1f}  {status}")

    # ── 最終摘要 ────────────────────────────────────────────────────────────────
    _banner("Check 摘要報告")
    passed = results["NaN/Inf"] == 0 and not results["錯誤"]
    _info(f"策略檔  : {Path(args.policy).name}")
    _info(f"總步數  : {results['步數']}")
    if results["NaN/Inf"] == 0:
        _ok(f"NaN/Inf : {results['NaN/Inf']}")
    else:
        _fail(f"NaN/Inf : {results['NaN/Inf']}")
    if results["超安全範圍"] == 0:
        _ok(f"超安全範圍: {results['超安全範圍']}")
    else:
        _warn(f"超安全範圍: {results['超安全範圍']}")
    if results["扭矩過載"] == 0:
        _ok(f"扭矩過載  : {results['扭矩過載']}  （閾值 {args.torque_limit*100:.0f}%）")
    else:
        _warn(f"扭矩過載  : {results['扭矩過載']}  （超過 {args.torque_limit*100:.0f}% 最大扭矩）")
    if results["錯誤"]:
        for e in results["錯誤"]:
            _fail(f"錯誤: {e}")
    print()
    if passed:
        _ok("結果: PASS ✓")
    else:
        _fail("結果: FAIL ✗")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="robot_sim2real 統一測試腳本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
模式：
  check  (預設) — 快速自動驗證 N 步推論（無硬體）
  policy        — 連續推論，可接真實 IMU 和馬達
  replay        — 從錄製 CSV 重播，自動比對 policy 輸出
  zero          — 所有馬達歸零位（上電安全確認）
  sine          — 正弦波運動測試（±15°/±10°/±8°, 0.3Hz，已實測通過）

範例：
  python linux/test/test_policy.py
  python linux/test/test_policy.py --mode check --steps 200
  python linux/test/test_policy.py --mode policy --dry-run
  python linux/test/test_policy.py --mode policy --imu --dry-run
  python linux/test/test_policy.py --mode policy --imu --can can0 --ids 31,32,33,34,35
  python linux/test/test_policy.py --mode zero   --dry-run
  python linux/test/test_policy.py --mode sine   --dry-run
  python linux/test/test_policy.py --mode sine   --can can0 --ids 31,32,33,34,35
  python linux/test/test_policy.py --mode replay
  python linux/test/test_policy.py --mode replay --recording {RECORDINGS_DIR}/xxx.csv

可用錄製：
{chr(10).join('  ' + str(p) for p in sorted(RECORDINGS_DIR.glob('*_actions.csv'))[-5:]) if RECORDINGS_DIR.exists() else '  （尚無錄製）'}
        """,
    )

    # 模式
    parser.add_argument("--mode", default="check",
                        choices=["check", "policy", "replay", "zero", "stand", "sine"],
                        help="測試模式（預設: check）")

    # 策略檔
    parser.add_argument("--policy", default=str(DEFAULT_POLICY),
                        help=f"kinfer 路徑（預設: {DEFAULT_POLICY.name}）")

    # 安全限制
    parser.add_argument("--torque-limit", type=float, default=0.5,
                        help="扭矩安全上限（佔最大值比例，預設 0.5 = 50%%）；"
                             "用於 home_ramp 步長計算與 replay/check 過載判定")
    parser.add_argument("--torque-cap", type=float, default=0.0,
                        help="policy/replay 模式每顆馬達最大扭力上限（Nm，0=用 MOTOR_CONFIG 預設）；"
                             "例如 --torque-cap 20 限制全部馬達輸出 ≤ 20Nm")
    parser.add_argument("--replay-speed", type=float, default=1.0,
                        help="replay 播放速度（0.1~1.0，預設 1.0=原速）；"
                             "0.5 = 半速，每幀重複送 2 次，有效關節速度降一半")
    parser.add_argument("--estop-vel", type=float, default=20.0,
                        help="E-STOP 速度閾值（rad/s，預設 20.0）；"
                             "任意關節速度超過此值立即停止並回 ZEROS")
    parser.add_argument("--estop-pos-err", type=float, default=15.0,
                        help="E-STOP 位置偏差閾值（degree，預設 15.0）；"
                             "硬體位置與目標偏差超過此值立即停止（幀 5 後生效）")
    parser.add_argument("--estop-torque", type=float, default=0.85,
                        help="E-STOP 扭矩閾值（佔 hw_max 比例，預設 0.85=85%%）；"
                             "連續 3 幀估算扭矩超過此比例立即停止")
    parser.add_argument("--record-secs", type=float, default=0.0,
                        help="policy 模式跑滿 N 秒後自動停止並輸出統計分析（0=不限制）")

    # 錄製
    parser.add_argument("--recording", default=None,
                        help="replay 模式的 CSV 路徑（不指定則自動選最新）")

    # 馬達
    parser.add_argument("--can", default="can0",
                        help="右腿 CAN 介面（預設: can0；41~45）")
    parser.add_argument("--left-can", default=None,
                        help="左腿 CAN 介面（預設：與 --can 相同；31~39 使用此介面，"
                             "雙 CAN 時設為 can1）")
    parser.add_argument("--ids", default="31,32,33,34,35",
                        help="連接的馬達 CAN ID（預設: 31,32,33,34,35）")
    parser.add_argument("--active-ids", default=None,
                        help="實際送指令的馬達 ID，其餘保持零位（不指定=等同 --ids）"
                             "  例: --active-ids 34,44 只動兩個膝關節")
    parser.add_argument("--dry-run", action="store_true",
                        help="不送 CAN 指令，只印推論結果")
    parser.add_argument("--no-policy", action="store_true",
                        help="replay 模式：直接送 CSV 錄製 target，不跑 policy 推論"
                             "（用於確認錄製動作本身的安全性，排除 model 版本差異影響）")
    parser.add_argument("--skip-home-ramp", action="store_true",
                        help="跳過 Home ramp 階段（只在機器人已在初始姿態時使用）")

    # IMU
    parser.add_argument("--imu", action="store_true",
                        help="啟用真實 H30 Mini IMU")
    parser.add_argument("--imu-port", default="/dev/ttyACM0",
                        help="H30 串口（預設: /dev/ttyACM0）")
    parser.add_argument("--imu-baud", type=int, default=460800,
                        help="H30 波特率（預設: 460800）")

    # sine 模式
    parser.add_argument("--sine-freq", type=float, default=SINE_FREQ_HZ,
                        help=f"sine 模式頻率（Hz，預設: {SINE_FREQ_HZ}）；"
                             "頻率越高扭矩越大，自動警告超限")

    # check 模式
    parser.add_argument("--steps", type=int, default=100,
                        help="check 模式的推論步數（預設: 100）")

    args = parser.parse_args()

    # 解析馬達 ID
    motor_ids = [int(x) for x in args.ids.split(",")]
    for mid in motor_ids:
        if mid not in MOTOR_CONFIG:
            _fail(f"不認識的 motor ID: {mid}（可用：{sorted(MOTOR_CONFIG.keys())}）")
            sys.exit(1)

    # 解析 active-ids（實際送指令的子集）
    if args.active_ids:
        active_ids = [int(x) for x in args.active_ids.split(",")]
        for mid in active_ids:
            if mid not in MOTOR_CONFIG:
                _fail(f"--active-ids 中不認識的 ID: {mid}")
                sys.exit(1)
            if mid not in motor_ids:
                _fail(f"--active-ids {mid} 不在 --ids 列表中（需先連接才能啟用）")
                sys.exit(1)
    else:
        active_ids = motor_ids

    # ── 建立 CAN 分配表（左腿 31~39 → left_can，右腿 41~49 → right_can）────────
    right_can = args.can
    left_can  = args.left_can or args.can
    can_assignment = {}
    for mid in motor_ids:
        can_assignment[mid] = left_can if 31 <= mid <= 39 else right_can
    # 填入模組級顯示表
    _MID_CAN.update(can_assignment)

    # ── 啟動橫幅 ──────────────────────────────────────────────────────────────
    _banner("robot_sim2real — Pi 部署測試腳本")
    _section("啟動設定")
    _info(f"模式    : {args.mode.upper()}")
    _info(f"策略    : {Path(args.policy).name}")
    _info(f"策略路徑: {args.policy}")
    _info(f"CAN右腿 : {right_can}  （{'DRY RUN' if args.dry_run else 'LIVE'}）")
    _info(f"CAN左腿 : {left_can}")
    _info(f"馬達 IDs: {motor_ids}")
    if args.active_ids:
        _info(f"Active  : {active_ids}  （其餘保持零位）")
    _info(f"IMU     : {'H30 USB — ' + args.imu_port if args.imu else '假 IMU（靜止直立）'}")
    _info(f"扭矩限制: {args.torque_limit*100:.0f}%  （home_ramp 步長 + replay/check 警告閾值）")
    _section("馬達配置")
    print(f"  {'ID':>3}  {'CAN':>4}  {'關節名稱':<28}  {'型號':>4}  {'kp':>6}  {'kd':>6}  {'最大扭矩':>8}")
    for mid in motor_ids:
        cfg  = MOTOR_CONFIG[mid]
        mt   = MAX_TORQUE[cfg["type"]]
        tag  = " ← active" if mid in active_ids else ""
        iface = can_assignment[mid]
        print(f"  {mid:>3}  {iface:>4}  {cfg['name']:<28}  {cfg['type']:>4}  "
              f"{cfg['kp']:>6.0f}  {cfg['kd']:>6.3f}  {mt:>7.1f}Nm{tag}")

    # IMU
    bridge = None
    if args.imu:
        _section("啟動 IMU")
        bridge = start_imu(args.imu_port, args.imu_baud)

    # 馬達驅動器（check 不需要；其他模式 dry_run 時也不建立）
    driver = None
    if not args.dry_run and args.mode not in ("check",):
        ifaces = sorted(set(can_assignment.values()))
        _section(f"連接 CAN: {ifaces}")
        driver = setup_driver(can_assignment)
        _ok(f"所有馬達已啟用")
        print()

    try:
        if args.mode == "check":
            ok = run_check(args, bridge)
            sys.exit(0 if ok else 1)
        elif args.mode == "policy":
            run_policy(args, motor_ids, active_ids, driver, bridge)
        elif args.mode == "replay":
            run_replay(args, motor_ids, active_ids, driver, bridge)
        elif args.mode == "zero":
            run_zero(args, motor_ids, driver)
        elif args.mode == "stand":
            run_stand(args, motor_ids, driver)
        elif args.mode == "sine":
            run_sine(args, motor_ids, active_ids, driver)
    finally:
        if driver:
            print("停用馬達...")
            disable_all(driver, motor_ids)


if __name__ == "__main__":
    main()
