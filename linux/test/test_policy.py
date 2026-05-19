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
_REPO_ROOT  = _HERE.parent.parent                     # repo 根目錄
_KSIM_ROOT  = Path("/home/andykuo/ksim-gym")          # ksim-gym 根目錄

# 預設策略檔：優先找 repo 內 models/，其次 fallback 到 ksim-gym/kbot_robot/Policies/
def _find_default_policy() -> Path:
    models_dir = _REPO_ROOT / "models"
    kinfers = sorted(models_dir.glob("*.kinfer"))
    if kinfers:
        return kinfers[-1]
    return _KSIM_ROOT / "kbot_robot" / "Policies" / "kbot_zero_position.kinfer"

DEFAULT_POLICY = _find_default_policy()

# 錄製資料夾（來自 train_v1/test_policy.py 的錄製）
RECORDINGS_DIR = _KSIM_ROOT / "recordings"


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
# 錄製關節 → policy 20-dim 索引的對應
_REC_TO_POL = [POLICY_JOINT_NAMES.index(n) for n in RECORDING_JOINT_NAMES]

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


def build_policy_feed(step_sess, joint_pos, joint_vel, carry, num_commands, sim_t,
                      bridge=None):
    """組裝 policy 輸入字典。bridge 不為 None 時讀真實 IMU。"""
    import threading
    names = {i.name for i in step_sess.get_inputs()}
    feed = {
        "joint_angles":             joint_pos.astype(np.float32),
        "joint_angular_velocities": joint_vel.astype(np.float32),
        "carry":                    carry,
    }
    if bridge is not None:
        with bridge._imu_lock:
            acc  = bridge.IMU_STATE["acc"].copy()
            gyro = bridge.IMU_STATE["gyro"].copy()
            quat = bridge.IMU_STATE["quat"].copy()
        proj_grav = bridge.proj_gravity_from_quat(*quat)
    else:
        acc       = np.array([0.0, 0.0, -9.81], dtype=np.float32)
        gyro      = np.zeros(3, dtype=np.float32)
        proj_grav = np.array([0.0, 0.0, -1.0],  dtype=np.float32)

    if "projected_gravity" in names:
        feed["projected_gravity"] = proj_grav
    if "gyroscope" in names:
        feed["gyroscope"] = gyro
    if "accelerometer" in names:
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

def setup_driver(can_iface: str, motor_ids: list):
    from robstride_driver import PyRobstrideDriver, PyRobstrideActuatorType
    driver = PyRobstrideDriver(can_iface)
    driver.connect(can_iface)
    for mid in motor_ids:
        cfg   = MOTOR_CONFIG[mid]
        atype = getattr(PyRobstrideActuatorType, ACTUATOR_TYPE_MAP[cfg["type"]])
        driver.add_actuator(can_id=mid, actuator_type=atype)
        driver.enable_actuator(actuator_id=mid)
        _ok(f"馬達 {mid:2d} ({cfg['name']:<28}) 已啟用")
    return driver


def send_cmd(driver, mid: int, position: float, kp: float, kd: float):
    from robstride_driver import PyActuatorCommand
    driver.send_command(
        actuator_id=mid,
        command=PyActuatorCommand(position=position, velocity=0.0, torque=0.0, kp=kp, kd=kd),
    )


def disable_all(driver, motor_ids: list):
    from robstride_driver import PyActuatorCommand
    for mid in motor_ids:
        try:
            driver.send_command(
                actuator_id=mid,
                command=PyActuatorCommand(position=0.0, velocity=0.0, torque=0.0, kp=0.0, kd=0.0),
            )
        except Exception:
            pass


def read_states(driver, motor_ids, joint_pos, joint_vel, id_to_idx):
    for mid in motor_ids:
        idx = id_to_idx[mid]
        try:
            s = driver.get_actuator_state(actuator_id=mid)
            joint_pos[idx] = s.position
            joint_vel[idx] = s.velocity
        except Exception as e:
            print(f"[WARN] 馬達 {mid} 讀取失敗: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 模式 1：policy  — ONNX 推論
# ═══════════════════════════════════════════════════════════════════════════════

def home_ramp(motor_ids: list, driver, id_to_idx: dict,
              joint_pos: np.ndarray, joint_vel: np.ndarray,
              ramp_deg_per_step: float = 4.0):
    """從當前位置緩慢移動到 home position（每 20ms 最多 ramp_deg_per_step 度）。
    與 firmware Home state 邏輯完全一致，避免從零位暴衝到初始姿態。
    """
    ctrl_dt = 0.02
    ramp_rad = math.radians(ramp_deg_per_step)
    home_targets = {mid: math.radians(_ZEROS_DEG[MOTOR_CONFIG[mid]["name"]])
                    for mid in motor_ids if MOTOR_CONFIG[mid]["name"] in _ZEROS_DEG}

    _section(f"Home Ramp — 緩移至初始姿態（每步 ≤{ramp_deg_per_step}°，20ms/步）")
    _info("目標: " + "  ".join(
        f"{MOTOR_CONFIG[m]['name'].replace('dof_','')[:10]}={math.degrees(home_targets[m]):+.0f}°"
        for m in motor_ids if m in home_targets))

    step = 0
    while True:
        t0 = time.time()
        read_states(driver, motor_ids, joint_pos, joint_vel, id_to_idx)

        max_err = 0.0
        for mid in motor_ids:
            if mid not in home_targets:
                continue
            idx = id_to_idx[mid]
            err = home_targets[mid] - joint_pos[idx]
            max_err = max(max_err, abs(err))
            step_pos = joint_pos[idx] + math.copysign(min(abs(err), ramp_rad), err)
            send_cmd(driver, mid, step_pos,
                     MOTOR_CONFIG[mid]["kp"], MOTOR_CONFIG[mid]["kd"])

        if step % 50 == 0:
            print(f"  [{step*ctrl_dt:5.1f}s] max_err={math.degrees(max_err):.1f}°  "
                  f"joints=[" + " ".join(f"{math.degrees(joint_pos[id_to_idx[m]]):+.1f}"
                                         for m in motor_ids if m in home_targets) + "]°")

        if max_err < 0.1:   # ~5.7°，與 firmware 相同閾值
            _ok(f"Home Ramp 完成：誤差 {math.degrees(max_err):.2f}° < 5.7°，共 {step} 步 ({step*ctrl_dt:.1f}s)")
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
    _info(f"CAN   : {'DRY RUN（不送指令）' if args.dry_run else f'LIVE — {args.can}'}")
    _info(f"馬達  : {motor_ids}")
    _info(f"Active: {active_ids}")
    _section("腿部關節安全限制")
    _print_joint_limits_table()

    # Home ramp：先緩移到初始姿態，避免從零位暴衝
    if driver and not args.skip_home_ramp:
        home_ramp(motor_ids, driver, id_to_idx, joint_pos, joint_vel)
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
            actions = np.clip(outputs[0], _SAFE_MIN_ARR, _SAFE_MAX_ARR)
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
                print(f"  {'關節':<26}  {'cur(°)':>7}  {'tgt(°)':>7}  {'err(°)':>6}  {'τ(Nm)':>12}")
                for mid in motor_ids:
                    idx      = motor_id_to_policy_idx(mid)
                    cfg      = MOTOR_CONFIG[mid]
                    name     = cfg["name"].replace("dof_", "")
                    cur      = math.degrees(joint_pos[idx])
                    target   = math.degrees(actions[idx])
                    err      = target - cur
                    max_t    = MAX_TORQUE[cfg["type"]]
                    torque   = calc_torque(float(actions[idx]), joint_pos[idx],
                                          joint_vel[idx], cfg["kp"], cfg["kd"], max_t)
                    overload = abs(torque) >= max_t * 0.95
                    flag     = " !OVERLOAD" if overload else ""
                    active_tag = "" if mid in set(active_ids) else " [hold]"
                    if overload:
                        overload_flags.append(name)
                    print(f"  {name:<26}  {cur:>+7.1f}  {target:>+7.1f}  {err:>+6.1f}  "
                          f"{torque:>+7.1f}/{max_t:>3.0f}Nm{flag}{active_tag}")
                if overload_flags:
                    _warn(f"扭矩過載: {overload_flags}")

            sim_t    += ctrl_dt
            step_cnt += 1
            slp = ctrl_dt - (time.time() - t0)
            if slp > 0:
                time.sleep(slp)

    except KeyboardInterrupt:
        print("\n停止")


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
        home_ramp(motor_ids, driver, id_to_idx, joint_pos, joint_vel)

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
        home_ramp(motor_ids, driver, id_to_idx, joint_pos, joint_vel)
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

    # 載入 policy
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

    errors   = []   # per-leg per-frame |policy_output - recorded_target|
    oob_cnt  = 0    # out-of-safe-range count
    nan_cnt  = 0

    _section(f"重播中（{'DRY RUN' if args.dry_run else 'LIVE CAN'}）")
    print(f"  {'幀':>5}  {'時間':>6}  {'均誤差':>7}  {'超界':>4}  {'NaN':>4}  關節(cur→tgt)概覽")

    for row_i, row in enumerate(rows):
        t0 = time.time()

        # 從錄製讀取關節狀態
        joint_pos = np.zeros(20, dtype=np.float32)
        joint_vel = np.zeros(20, dtype=np.float32)
        for rec_i, name in enumerate(RECORDING_JOINT_NAMES):
            pol_i = _REC_TO_POL[rec_i]
            joint_pos[pol_i] = float(row.get(f"pos_{name}", 0))
            joint_vel[pol_i] = float(row.get(f"vel_{name}", 0))

        # 如果有真實馬達，也讀取回饋（overlay 真實值）
        if driver:
            read_states(driver, motor_ids, joint_pos, joint_vel, id_to_idx)

        # policy 推論（假 IMU，replay 時統一用靜止直立）
        feed    = build_policy_feed(step_sess, joint_pos, joint_vel, carry,
                                    num_commands, sim_t, bridge)
        outputs = step_sess.run(None, feed)
        actions = outputs[0]
        carry   = outputs[1]

        # 自動檢查
        if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
            nan_cnt += 1
        clipped = np.clip(actions, _SAFE_MIN_ARR, _SAFE_MAX_ARR)
        if not np.allclose(actions, clipped, atol=1e-6):
            oob_cnt += 1

        # 比對錄製 target
        frame_err = []
        for rec_i, name in enumerate(RECORDING_JOINT_NAMES):
            pol_i      = _REC_TO_POL[rec_i]
            rec_target = float(row.get(f"target_{name}", 0))
            pol_target = float(actions[pol_i])
            frame_err.append(abs(pol_target - rec_target))
        errors.append(frame_err)

        # 送馬達（只有 active_ids 接收錄製 target，其餘保持零位）
        if not args.dry_run and driver:
            for mid in motor_ids:
                idx = motor_id_to_policy_idx(mid)
                pos = float(clipped[idx]) if mid in active_set else 0.0
                send_cmd(driver, mid, pos, MOTOR_CONFIG[mid]["kp"], MOTOR_CONFIG[mid]["kd"])

        if row_i % 50 == 0:
            mean_err = np.mean(frame_err) * 180 / math.pi
            # Show first 3 leg joints as quick overview: cur→tgt
            overview_parts = []
            for mid in motor_ids[:3]:
                idx = motor_id_to_policy_idx(mid)
                cur_d = math.degrees(joint_pos[idx])
                tgt_d = math.degrees(float(clipped[idx]))
                nm = MOTOR_CONFIG[mid]["name"].replace("dof_left_","L.").replace("dof_right_","R.")
                overview_parts.append(f"{nm}:{cur_d:+.0f}→{tgt_d:+.0f}°")
            print(f"  [{row_i:4d}/{len(rows)}]  t={sim_t:5.2f}s  "
                  f"err={mean_err:5.2f}°  oob={oob_cnt:3d}  nan={nan_cnt:3d}  "
                  + "  ".join(overview_parts))

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

    joint_pos = np.zeros(20, dtype=np.float32)
    joint_vel = np.zeros(20, dtype=np.float32)

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
    _leg_mid_order = [31, 32, 33, 34, 35, 41, 42, 43, 44, 45]
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

        actions = outputs[0]
        carry   = outputs[1]
        history.append(actions.copy())

        if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
            results["NaN/Inf"] += 1

        clipped = np.clip(actions, _SAFE_MIN_ARR, _SAFE_MAX_ARR)
        if not np.allclose(actions, clipped, atol=1e-6):
            results["超安全範圍"] += 1

        # 扭矩過載檢查（模擬 PD 扭矩，joint_vel=0 因假設靜止）
        for mid in _leg_mid_order:
            idx    = motor_id_to_policy_idx(mid)
            cfg    = MOTOR_CONFIG[mid]
            torque = calc_torque(float(actions[idx]), joint_pos[idx], 0.0,
                                 cfg["kp"], cfg["kd"], MAX_TORQUE[cfg["type"]])
            if abs(torque) >= MAX_TORQUE[cfg["type"]] * 0.95:
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
        _ok(f"扭矩過載  : {results['扭矩過載']}")
    else:
        _warn(f"扭矩過載  : {results['扭矩過載']}  （有步驟達到 95% 最大扭矩）")
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

    # 錄製
    parser.add_argument("--recording", default=None,
                        help="replay 模式的 CSV 路徑（不指定則自動選最新）")

    # 馬達
    parser.add_argument("--can", default="can0", help="CAN 介面（預設: can0）")
    parser.add_argument("--ids", default="31,32,33,34,35",
                        help="連接的馬達 CAN ID（預設: 31,32,33,34,35）")
    parser.add_argument("--active-ids", default=None,
                        help="實際送指令的馬達 ID，其餘保持零位（不指定=等同 --ids）"
                             "  例: --active-ids 34,44 只動兩個膝關節")
    parser.add_argument("--dry-run", action="store_true",
                        help="不送 CAN 指令，只印推論結果")
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

    # ── 啟動橫幅 ──────────────────────────────────────────────────────────────
    _banner("robot_sim2real — Pi 部署測試腳本")
    _section("啟動設定")
    _info(f"模式    : {args.mode.upper()}")
    _info(f"策略    : {Path(args.policy).name}")
    _info(f"策略路徑: {args.policy}")
    _info(f"CAN     : {args.can}  （{'DRY RUN' if args.dry_run else 'LIVE'}）")
    _info(f"馬達 IDs: {motor_ids}")
    if args.active_ids:
        _info(f"Active  : {active_ids}  （其餘保持零位）")
    _info(f"IMU     : {'H30 USB — ' + args.imu_port if args.imu else '假 IMU（靜止直立）'}")
    _section("馬達配置")
    print(f"  {'ID':>3}  {'關節名稱':<28}  {'型號':>4}  {'kp':>6}  {'kd':>6}  {'最大扭矩':>8}")
    for mid in motor_ids:
        cfg = MOTOR_CONFIG[mid]
        mt = MAX_TORQUE[cfg["type"]]
        tag = " ← active" if mid in active_ids else ""
        print(f"  {mid:>3}  {cfg['name']:<28}  {cfg['type']:>4}  "
              f"{cfg['kp']:>6.0f}  {cfg['kd']:>6.3f}  {mt:>7.1f}Nm{tag}")

    # IMU
    bridge = None
    if args.imu:
        _section("啟動 IMU")
        bridge = start_imu(args.imu_port, args.imu_baud)

    # 馬達驅動器（check 不需要；其他模式 dry_run 時也不建立）
    driver = None
    if not args.dry_run and args.mode not in ("check",):
        _section(f"連接 CAN: {args.can}")
        driver = setup_driver(args.can, motor_ids)
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
