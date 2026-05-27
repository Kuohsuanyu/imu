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
    31: {"name": "dof_left_hip_pitch_04",  "type": "04", "kp": 150.0, "kd": 24.722},
    32: {"name": "dof_left_hip_roll_03",   "type": "03", "kp": 200.0, "kd": 26.387},
    33: {"name": "dof_left_hip_yaw_03",    "type": "03", "kp": 100.0, "kd":  3.419},
    34: {"name": "dof_left_knee_04",       "type": "04", "kp": 150.0, "kd":  8.654},
    35: {"name": "dof_left_ankle_02",      "type": "02", "kp":  40.0, "kd":  0.990},
    41: {"name": "dof_right_hip_pitch_04", "type": "04", "kp": 150.0, "kd": 24.722},
    42: {"name": "dof_right_hip_roll_03",  "type": "03", "kp": 200.0, "kd": 26.387},
    43: {"name": "dof_right_hip_yaw_03",   "type": "03", "kp": 100.0, "kd":  3.419},
    44: {"name": "dof_right_knee_04",      "type": "04", "kp": 150.0, "kd":  8.654},
    45: {"name": "dof_right_ankle_02",     "type": "02", "kp":  40.0, "kd":  0.990},
}
ACTUATOR_TYPE_MAP = {"02": "Robstride02", "03": "Robstride03", "04": "Robstride04"}
MAX_TORQUE = {"04": 84.0, "03": 42.0, "02": 11.9}

# 正弦波參數（與舊版 test_motor_policy.py 一致，已實測通過）
SINE_AMP_RAD = {
    "04": math.radians(15),   # hip pitch / knee ±15°
    "03": math.radians(10),   # hip roll / yaw   ±10°
    "02": math.radians(8),    # ankle             ±8°
}
SINE_FREQ_HZ = 0.3   # 0.3 Hz 慢速確認響應

# 安全關節限制（policy 輸出的 hard clip）
_ZEROS_DEG = {
    "dof_right_hip_pitch_04": -20.0, "dof_right_hip_roll_03":  0.0,
    "dof_right_hip_yaw_03":    0.0,  "dof_right_knee_04":    -50.0,
    "dof_right_ankle_02":     30.0,  "dof_left_hip_pitch_04": 20.0,
    "dof_left_hip_roll_03":    0.0,  "dof_left_hip_yaw_03":    0.0,
    "dof_left_knee_04":       50.0,  "dof_left_ankle_02":    -30.0,
}
_MARGIN_DEG = 55.0
SAFE_MIN = {n: math.radians(z - _MARGIN_DEG) for n, z in _ZEROS_DEG.items()}
SAFE_MAX = {n: math.radians(z + _MARGIN_DEG) for n, z in _ZEROS_DEG.items()}

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


# ── 馬達診斷追蹤（per-motor 失敗統計）────────────────────────────────────────

_motor_stats: dict = {}   # mid → {"ok": int, "mismatch": int, "other": int, "consec_fail": int}

def _stats(mid: int) -> dict:
    if mid not in _motor_stats:
        _motor_stats[mid] = {"ok": 0, "mismatch": 0, "other": 0, "consec_fail": 0}
    return _motor_stats[mid]

def _record_ok(mid: int):
    s = _stats(mid)
    s["ok"] += 1
    s["consec_fail"] = 0

def _record_fail(mid: int, e: Exception):
    s   = _stats(mid)
    msg = str(e)
    if "mismatch" in msg.lower():
        s["mismatch"] += 1
    else:
        s["other"] += 1
    s["consec_fail"] += 1
    return msg

def print_motor_health(motor_ids: list, label: str = ""):
    """印出每顆馬達的讀取成功/失敗統計。"""
    tag = f"  [{label}] " if label else "  "
    print(f"\n{tag}── 馬達健康報告 ──")
    print(f"  {'ID':>3}  {'名稱':<24}  {'成功':>6}  {'mismatch':>9}  {'其他錯誤':>9}  {'連續失敗':>9}  狀態")
    for mid in motor_ids:
        s    = _stats(mid)
        cfg  = MOTOR_CONFIG[mid]
        name = cfg["name"].replace("dof_", "")
        total = s["ok"] + s["mismatch"] + s["other"]
        ok_pct = s["ok"] / total * 100 if total else 0
        if s["consec_fail"] >= 10:
            status = "!! 疑似斷線"
        elif ok_pct < 50 and total > 10:
            status = "⚠  不穩定"
        else:
            status = "OK"
        print(f"  {mid:>3}  {name:<24}  {s['ok']:>6}  {s['mismatch']:>9}  {s['other']:>9}  "
              f"{s['consec_fail']:>9}  {status}")
    print()


# ── 馬達驅動器 ─────────────────────────────────────────────────────────────────

def setup_driver(can_assignment: dict) -> dict:
    """建立馬達驅動器映射 {mid: PyRobstrideDriver}。
    can_assignment = {mid: can_iface_str, ...}，支援多 CAN 介面（左右腿分開）。
    先 add_actuator（ping 全部），再統一 enable，避免已啟用的馬達 CAN 幀干擾後續 ping。
    """
    from robstride_driver import PyRobstrideDriver, PyRobstrideActuatorType
    iface_drivers: dict = {}
    for iface in set(can_assignment.values()):
        print(f"  [CAN] 開啟介面 {iface} ...")
        try:
            d = PyRobstrideDriver(iface)
            d.connect(iface)
            iface_drivers[iface] = d
            _ok(f"CAN 介面 {iface} 連接成功")
        except Exception as e:
            _fail(f"CAN 介面 {iface} 連接失敗: {e!r}")
            raise
    driver_map: dict = {}

    # Phase 1: add_actuator (ping) all motors without enabling
    print(f"\n  [Phase 1] Ping 全部馬達（不啟用）")
    ping_failed = []
    for mid in sorted(can_assignment.keys()):
        iface = can_assignment[mid]
        d     = iface_drivers[iface]
        cfg   = MOTOR_CONFIG[mid]
        atype = getattr(PyRobstrideActuatorType, ACTUATOR_TYPE_MAP[cfg["type"]])
        print(f"    ping 馬達 {mid:2d} ({cfg['name']:<28}) [{iface}] type={cfg['type']} ...", end=" ", flush=True)
        try:
            d.add_actuator(can_id=mid, actuator_type=atype)
            print("OK")
        except Exception as e:
            print(f"FAIL: {e!r}")
            ping_failed.append(mid)
        time.sleep(0.05)
        driver_map[mid] = d

    if ping_failed:
        _warn(f"Phase 1 ping 失敗的馬達: {ping_failed}")
    else:
        _ok(f"Phase 1 完成：所有 {len(driver_map)} 顆馬達 ping 成功")

    time.sleep(0.1)   # settle before enabling

    # Phase 2: enable all motors, Robstride04 gets longer delay
    print(f"\n  [Phase 2] 啟用全部馬達")
    enable_failed = []
    for mid in sorted(can_assignment.keys()):
        cfg        = MOTOR_CONFIG[mid]
        d          = driver_map[mid]
        init_delay = 0.2 if cfg["type"] == "04" else 0.05
        print(f"    enable 馬達 {mid:2d} ({cfg['name']:<28}) [{can_assignment[mid]}] ...", end=" ", flush=True)
        try:
            d.enable_actuator(actuator_id=mid)
            print(f"OK (等 {init_delay*1000:.0f}ms)", end=" ", flush=True)
        except Exception as e:
            print(f"FAIL: {e!r}")
            enable_failed.append(mid)
            time.sleep(init_delay)
            continue
        time.sleep(init_delay)
        # 啟用後立刻讀取狀態確認
        for attempt in range(3):
            try:
                s = d.get_actuator_state(actuator_id=mid)
                print(f"→ pos={math.degrees(s.position):+.1f}°  vel={s.velocity:+.3f}  "
                      f"temp={s.temperature:.0f}°C  faults=0x{getattr(s,'fault_code', 0):04X}")
                _record_ok(mid)
                break
            except Exception as e:
                msg = str(e)
                if attempt == 2:
                    print(f"\n    [WARN] 讀取失敗({attempt+1}/3): {msg!r}")
                    _record_fail(mid, e)
                else:
                    time.sleep(0.01)

    if enable_failed:
        _warn(f"Phase 2 enable 失敗的馬達: {enable_failed}")
    else:
        _ok(f"Phase 2 完成：所有馬達已啟用")

    # Phase 3: 啟用後等待 0.3s，再做一次全體狀態確認
    print(f"\n  [Phase 3] 啟用後穩定確認（等 300ms）")
    time.sleep(0.3)
    all_ok = True
    for mid in sorted(can_assignment.keys()):
        cfg = MOTOR_CONFIG[mid]
        d   = driver_map[mid]
        try:
            s = d.get_actuator_state(actuator_id=mid)
            fault = getattr(s, 'fault_code', 0)
            fault_tag = f"  ⚠ faults=0x{fault:04X}" if fault else ""
            _ok(f"馬達 {mid:2d} ({cfg['name']:<28}) [{can_assignment[mid]}]  "
                f"pos={math.degrees(s.position):+.1f}°  temp={s.temperature:.0f}°C{fault_tag}")
            _record_ok(mid)
        except Exception as e:
            _warn(f"馬達 {mid:2d} ({cfg['name']:<28}) 最終確認失敗: {e!r}")
            _record_fail(mid, e)
            all_ok = False

    if not all_ok:
        _warn("部分馬達最終確認失敗，後續指令可能不穩定")
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


def _drain_until(driver, mid: int, timeout_s: float = 0.004):
    """持續消耗 CAN buffer 中其他馬達的幀，直到拿到 mid 的幀或超時。
    每次嘗試之間加 0.2ms 間隔，避免洗爆 CAN socket buffer (ENOBUFS)。
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            return driver.get_actuator_state(actuator_id=mid)
        except Exception as e:
            msg = str(e)
            if "mismatch" in msg.lower():
                _record_fail(mid, e)
                time.sleep(0.0002)   # 200µs 讓 buffer 喘息
                continue
            if "buffer" in msg.lower() or "105" in msg:
                # ENOBUFS：buffer 已滿，等久一點再試
                time.sleep(0.001)
                continue
            raise   # 其他真實錯誤才往上拋
    return None   # 超時，用上次的值


def read_states(driver_map, motor_ids, joint_pos, joint_vel, id_to_idx, retries: int = 3):
    for mid in motor_ids:
        idx = id_to_idx[mid]
        s = _drain_until(driver_map[mid], mid)
        if s is not None:
            joint_pos[idx] = s.position
            joint_vel[idx] = s.velocity
            _record_ok(mid)
        else:
            s_info = _stats(mid)
            if s_info["consec_fail"] % 10 == 0:   # 每 10 次才印一次，避免洗版
                print(f"[WARN] 馬達 {mid} 讀取逾時（連續={s_info['consec_fail']}）")


def send_and_read(driver_map, mid: int, step_pos: float, kp: float, kd: float,
                  joint_pos: np.ndarray, joint_vel: np.ndarray, idx: int,
                  retries: int = 3):
    """送指令後消耗 CAN buffer 直到拿到同一顆馬達的回應幀。"""
    send_cmd(driver_map, mid, step_pos, kp, kd)
    s = _drain_until(driver_map[mid], mid, timeout_s=0.003)
    if s is not None:
        joint_pos[idx] = s.position
        joint_vel[idx] = s.velocity
        _record_ok(mid)
    else:
        s_info = _stats(mid)
        if s_info["consec_fail"] % 10 == 0:
            print(f"[WARN] 馬達 {mid} 讀取逾時（連續={s_info['consec_fail']}）")


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


def home_ramp(motor_ids: list, driver, id_to_idx: dict,
              joint_pos: np.ndarray, joint_vel: np.ndarray,
              torque_limit_ratio: float = 0.5):
    """從當前位置緩慢移動到 ZEROS 初始姿態。
    每顆馬達依自身 kp / max_torque 自動計算安全步長，確保任何時刻扭矩 ≤ torque_limit_ratio。
    避免從任意位置開機後暴衝到初始姿態造成過載。
    """
    ctrl_dt = 0.02
    home_targets = {mid: math.radians(_ZEROS_DEG[MOTOR_CONFIG[mid]["name"]])
                    for mid in motor_ids if MOTOR_CONFIG[mid]["name"] in _ZEROS_DEG}
    safe_steps   = {mid: _safe_step_rad(mid, torque_limit_ratio) for mid in motor_ids}

    _section(f"Home Ramp — 緩移至初始姿態（扭矩限制 ≤{torque_limit_ratio*100:.0f}%）")
    _info("目標: " + "  ".join(
        f"{MOTOR_CONFIG[m]['name'].replace('dof_','')[:10]}={math.degrees(home_targets[m]):+.0f}°"
        for m in motor_ids if m in home_targets))
    _info("安全步長: " + "  ".join(
        f"{MOTOR_CONFIG[m]['name'].replace('dof_','')[:10]}≤{math.degrees(safe_steps[m]):.1f}°"
        for m in motor_ids))

    MAX_RAMP_STEPS = 400   # 最多 400 步（~8s），避免讀取失敗時無限循環
    step = 0
    while True:
        t0 = time.time()

        max_err = 0.0
        max_torque_ratio = 0.0
        for mid in motor_ids:
            if mid not in home_targets:
                continue
            idx      = id_to_idx[mid]
            cfg      = MOTOR_CONFIG[mid]
            err      = home_targets[mid] - joint_pos[idx]
            max_err  = max(max_err, abs(err))
            clamp    = safe_steps[mid]
            step_pos = joint_pos[idx] + math.copysign(min(abs(err), clamp), err)
            est_torque = cfg["kp"] * abs(step_pos - joint_pos[idx])
            max_torque_ratio = max(max_torque_ratio,
                                   est_torque / MAX_TORQUE[cfg["type"]])
            # send 完立刻 read 同一顆，避免多馬達 CAN 回應交錯
            send_and_read(driver, mid, step_pos, cfg["kp"], cfg["kd"],
                          joint_pos, joint_vel, idx)

        if step % 25 == 0:
            print(f"  [{step*ctrl_dt:5.1f}s] max_err={math.degrees(max_err):.1f}°  "
                  f"max_τ={max_torque_ratio*100:.0f}%  "
                  + "  ".join(f"{MOTOR_CONFIG[m]['name'].replace('dof_','')[:8]}"
                               f"={math.degrees(joint_pos[id_to_idx[m]]):+.1f}°"
                               for m in motor_ids if m in home_targets))

        if max_err < 0.1:
            _ok(f"Home Ramp 完成：誤差 {math.degrees(max_err):.2f}°，共 {step} 步 ({step*ctrl_dt:.1f}s)，最大扭矩比 {max_torque_ratio*100:.0f}%")
            break

        if step >= MAX_RAMP_STEPS:
            _warn(f"Home Ramp 超時（{step} 步），最大殘差 {math.degrees(max_err):.1f}°，繼續執行")
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

    id_to_idx = {mid: motor_id_to_policy_idx(mid) for mid in motor_ids}
    ctrl_dt   = 0.02
    sim_t     = 0.0
    step_cnt  = 0

    joint_pos = np.zeros(20, dtype=np.float32)
    joint_vel = np.zeros(20, dtype=np.float32)

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
        home_ramp(motor_ids, driver, id_to_idx, joint_pos, joint_vel, args.torque_limit)
        input("\n  [確認] 已到達初始姿態，按 Enter 開始 Policy 推論...")
    elif args.skip_home_ramp:
        _warn("--skip-home-ramp：跳過 Home ramp，確認機器人已在初始姿態")

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
                    pos = float(actions[idx]) if mid in active_ids else 0.0
                    send_cmd(driver, mid, pos, MOTOR_CONFIG[mid]["kp"], MOTOR_CONFIG[mid]["kd"])

            if step_cnt % 50 == 0:
                overload_flags = []
                print(f"\n[t={sim_t:6.2f}s  step={step_cnt}]")
                # IMU 資料
                if bridge is not None:
                    with bridge._imu_lock:
                        _acc  = bridge.IMU_STATE["acc"].copy()
                        _gyro = bridge.IMU_STATE["gyro"].copy()
                        _quat = bridge.IMU_STATE["quat"].copy()
                    _pg = bridge.proj_gravity_from_quat(*_quat)
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

            # 每 100 步印健康摘要
            if step_cnt % 100 == 99 and driver:
                print_motor_health(motor_ids, f"t={sim_t:.0f}s")

            sim_t    += ctrl_dt
            step_cnt += 1
            slp = ctrl_dt - (time.time() - t0)
            if slp > 0:
                time.sleep(slp)

    except KeyboardInterrupt:
        print("\n停止")
        if driver:
            print_motor_health(motor_ids, "最終統計")


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
        # 臨時用 _ZEROS_DEG 全設 0 的版本做 ramp
        _orig = dict(_ZEROS_DEG)
        for k in _ZEROS_DEG:
            _ZEROS_DEG[k] = 0.0
        home_ramp(motor_ids, driver, id_to_idx, joint_pos, joint_vel, args.torque_limit)
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
        home_ramp(motor_ids, driver, id_to_idx, joint_pos, joint_vel, args.torque_limit)

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
    用於確認每顆馬達在站姿附近的響應與扭矩輸出：
      type04 ±15° @ 0.3 Hz（hip pitch / knee）
      type03 ±10° @ 0.3 Hz（hip roll / yaw）
      type02  ±8° @ 0.3 Hz（ankle）
    非 active_ids 的馬達保持 ZEROS 站姿位置。
    """
    ctrl_dt   = 0.02
    step_cnt  = 0
    omega     = 2 * math.pi * SINE_FREQ_HZ
    id_to_idx = {mid: motor_id_to_policy_idx(mid) for mid in motor_ids}
    joint_pos = np.zeros(20, dtype=np.float32)
    joint_vel = np.zeros(20, dtype=np.float32)

    # ZEROS 在 rad（sine 的中心點）
    _zeros_rad = {mid: math.radians(_ZEROS_DEG.get(MOTOR_CONFIG[mid]["name"], 0.0))
                  for mid in motor_ids}

    active_set = set(active_ids)
    _banner(f"正弦波模式 — {SINE_FREQ_HZ} Hz，中心=ZEROS 站姿")
    _info(f"{'DRY RUN（不送指令）' if args.dry_run else 'LIVE CAN — 馬達上電中'}")
    _info("振幅：type04=±15°  type03=±10°  type02=±8°")
    _info("中心：ZEROS 站姿（不是馬達機械 0°）")
    _info("Ctrl+C 停止")
    if active_set != set(motor_ids):
        active_names = [MOTOR_CONFIG[m]["name"].replace("dof_","") for m in active_ids]
        print(f"  [限制] 只有 {active_ids} ({active_names}) 做正弦波，其餘保持 ZEROS 站姿")
    print("\n  ── 各關節 sine 實際範圍 ──")
    print(f"  {'關節':<26} {'ZEROS':>6} {'振幅':>6} {'最小':>8} {'最大':>8}")
    for mid in motor_ids:
        cfg  = MOTOR_CONFIG[mid]
        name = cfg["name"].replace("dof_","")
        z    = _ZEROS_DEG.get(cfg["name"], 0.0)
        amp  = math.degrees(SINE_AMP_RAD[cfg["type"]])
        tag  = "" if mid in active_set else " [hold]"
        print(f"  {name:<26} {z:>+6.0f}° {amp:>+6.0f}°  {z-amp:>+7.1f}°  {z+amp:>+7.1f}°{tag}")

    # 先移到站姿，再開始 sine
    if driver and not args.skip_home_ramp:
        home_ramp(motor_ids, driver, id_to_idx, joint_pos, joint_vel, args.torque_limit)
        input("\n  [確認] 已到站姿，按 Enter 開始 sine 測試...")

    try:
        while True:
            t0    = time.time()
            sim_t = step_cnt * ctrl_dt

            for mid in motor_ids:
                cfg    = MOTOR_CONFIG[mid]
                zero   = _zeros_rad[mid]
                idx    = id_to_idx[mid]
                if mid in active_set:
                    amp    = SINE_AMP_RAD[cfg["type"]]
                    target = zero + amp * math.sin(omega * sim_t)
                else:
                    target = zero
                if not args.dry_run and driver:
                    # send_and_read：送指令後馬達立刻回應，比周期廣播幀優先到達
                    send_and_read(driver, mid, target, cfg["kp"], cfg["kd"],
                                  joint_pos, joint_vel, idx)
                elif driver:
                    # dry_run：只讀不送
                    read_states(driver, [mid], joint_pos, joint_vel, id_to_idx)

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
                    st     = _stats(mid)
                    ok_pct = (st["ok"] / max(1, st["ok"] + st["mismatch"] + st["other"])) * 100
                    print(f"  {name:<28}  tgt={math.degrees(tgt):+6.1f}°  "
                          f"cur={math.degrees(cur):+6.1f}°  "
                          f"τ={torque:6.1f}/{max_t:.0f}Nm  "
                          f"ok={ok_pct:.0f}% cf={st['consec_fail']}{flag}{tag}")

            # 每 100 步（2s）印一次健康摘要
            if step_cnt % 100 == 99 and driver:
                print_motor_health(motor_ids, f"t={sim_t:.0f}s")

            step_cnt += 1
            slp = ctrl_dt - (time.time() - t0)
            if slp > 0:
                time.sleep(slp)
    except KeyboardInterrupt:
        print("\n停止")
        if driver:
            print_motor_health(motor_ids, "最終統計")


# ═══════════════════════════════════════════════════════════════════════════════
# 模式 4：replay  — 從錄製 CSV 重播並驗證
# ═══════════════════════════════════════════════════════════════════════════════

def _latest_recording() -> Path | None:
    """回傳 recordings/ 目錄中最新的 _actions.csv。"""
    if not RECORDINGS_DIR.exists():
        return None
    csvs = sorted(RECORDINGS_DIR.glob("*_actions.csv"))
    return csvs[-1] if csvs else None


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

    # Home ramp：讀取當前位置，緩移到站姿後再開始播放
    if driver and not args.skip_home_ramp:
        _ramp_pos = np.zeros(20, dtype=np.float32)
        _ramp_vel = np.zeros(20, dtype=np.float32)
        home_ramp(motor_ids, driver, id_to_idx, _ramp_pos, _ramp_vel, args.torque_limit)
        input("\n  [確認] 已到達站姿，按 Enter 開始 Replay...")

    errors        = []   # per-leg per-frame |policy_output - recorded_target|
    torque_ratios = []   # per-leg per-frame estimated torque ratio
    positions     = []   # per-leg per-frame commanded target position (deg)
    oob_cnt  = 0
    nan_cnt  = 0
    tlimit   = args.torque_limit

    _section(f"重播中（{'DRY RUN' if args.dry_run else 'LIVE CAN'}）")
    print(f"  {'幀':>5}  {'時間':>6}  {'均誤差':>7}  {'超界':>4}  {'NaN':>4}  關節(cur→tgt)概覽")

    for row_i, row in enumerate(rows):
        t0 = time.time()

        # 從錄製讀取關節狀態
        joint_pos = np.zeros(20, dtype=np.float32)
        joint_vel = np.zeros(20, dtype=np.float32)
        rec_targets_20 = np.zeros(20, dtype=np.float32)
        for rec_i, name in enumerate(RECORDING_JOINT_NAMES):
            pol_i = _REC_TO_POL[rec_i]
            joint_pos[pol_i]     = float(row.get(f"pos_{name}", 0))
            joint_vel[pol_i]     = float(row.get(f"vel_{name}", 0))
            rec_targets_20[pol_i] = float(row.get(f"target_{name}", 0))

        # 如果有真實馬達，也讀取回饋（overlay 真實值）
        if driver:
            read_states(driver, motor_ids, joint_pos, joint_vel, id_to_idx)

        if args.no_policy:
            # ── CSV 直播模式：直接送錄製 target，不跑 policy ────────────────
            actions = rec_targets_20
            clipped = np.clip(actions, _SAFE_MIN_ARR, _SAFE_MAX_ARR)
        else:
            # ── Policy 推論模式：用 v20 policy 重新計算目標 ─────────────────
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

        # 比對錄製 target + 計算扭矩比
        frame_err   = []
        frame_tau   = []
        for rec_i, name in enumerate(RECORDING_JOINT_NAMES):
            pol_i      = _REC_TO_POL[rec_i]
            rec_target = rec_targets_20[pol_i]
            pol_target = float(actions[pol_i])
            frame_err.append(abs(pol_target - rec_target))
            mid_for_name = next((m for m, c in MOTOR_CONFIG.items() if c["name"] == name), None)
            if mid_for_name:
                # 扭矩估算：從當前位置到目標的誤差
                frame_tau.append(torque_ratio(pol_target, joint_pos[pol_i],
                                              joint_vel[pol_i], mid_for_name))
            else:
                frame_tau.append(0.0)
        errors.append(frame_err)
        torque_ratios.append(frame_tau)
        positions.append([math.degrees(float(actions[_REC_TO_POL[i]]))
                          for i in range(len(RECORDING_JOINT_NAMES))])

        # 送馬達
        if not args.dry_run and driver:
            for mid in motor_ids:
                idx = motor_id_to_policy_idx(mid)
                pos = float(clipped[idx]) if mid in active_set else \
                      math.radians(_ZEROS_DEG.get(MOTOR_CONFIG[mid]["name"], 0.0))
                send_cmd(driver, mid, pos, MOTOR_CONFIG[mid]["kp"], MOTOR_CONFIG[mid]["kd"])

        if row_i % 50 == 0:
            max_tau  = max(frame_tau) if frame_tau else 0.0
            over_tag = f" [OVER {max_tau*100:.0f}%]" if max_tau > tlimit else ""
            print(f"\n  [frame {row_i:4d}/{len(rows)}  t={sim_t:6.2f}s  τmax={max_tau*100:.0f}%{over_tag}]")
            for mid in motor_ids:
                idx  = motor_id_to_policy_idx(mid)
                cfg  = MOTOR_CONFIG[mid]
                cur  = joint_pos[idx]
                vel  = joint_vel[idx]
                tgt  = float(clipped[idx])
                tau  = calc_torque(tgt, cur, vel, cfg["kp"], cfg["kd"], MAX_TORQUE[cfg["type"]])
                pct  = abs(tau) / MAX_TORQUE[cfg["type"]] * 100
                flag = " !OVER" if pct >= tlimit * 100 else ""
                print(f"  Actuator {mid:2d} ({_MID_CAN.get(mid,'?')}):"
                      f"  pos={cur:+7.3f}rad ({math.degrees(cur):+6.1f}°)"
                      f"  vel={vel:+6.3f}"
                      f"  torque={tau:+7.2f}Nm ({pct:4.1f}%)"
                      f"  tgt={tgt:+7.3f}rad ({math.degrees(tgt):+6.1f}°){flag}")

        sim_t += ctrl_dt
        slp = ctrl_dt - (time.time() - t0)
        if slp > 0:
            time.sleep(slp)

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
