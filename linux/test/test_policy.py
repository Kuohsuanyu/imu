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

# 預設策略檔（可用 --policy 覆蓋）
DEFAULT_POLICY = _KSIM_ROOT / "kbot_robot" / "Policies" / "kbot_zero_position.kinfer"

# 錄製資料夾（來自 train_v1/test_policy.py 的錄製）
RECORDINGS_DIR = _KSIM_ROOT / "recordings"

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
        bridge.run(imu_port, imu_baud, bridge.DEFAULT_VIRT_PORT, bridge.DEFAULT_VIRT_BAUD)

    t = threading.Thread(target=_run, daemon=True, name="h30-bridge")
    t.start()

    print(f"[IMU] 啟動 H30 bridge {imu_port} @ {imu_baud} baud，等待資料...")
    for _ in range(200):
        time.sleep(0.1)
        with bridge._imu_lock:
            if bridge.IMU_STATE["updated"]:
                break
    else:
        print("[WARN] 20 秒內未收到 IMU 資料，使用假值繼續")
    print("[IMU] 就緒")
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
        print(f"  馬達 {mid:2d} ({cfg['name']:<28}) 已啟用")
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

def run_policy(args, motor_ids: list, driver, bridge):
    print(f"\n載入策略: {args.policy}")
    init_sess, step_sess, meta = load_kinfer(args.policy)
    input_names  = [i.name for i in step_sess.get_inputs()]
    num_commands = meta.get("num_commands", 0) or 0
    carry_size   = meta["carry_size"]
    print(f"  輸入: {input_names}  carry_size={carry_size}  commands={num_commands}")

    carry_init = init_sess.run(None, {})
    carry = carry_init[0] if carry_init else np.zeros(carry_size, dtype=np.float32)

    id_to_idx = {mid: motor_id_to_policy_idx(mid) for mid in motor_ids}
    ctrl_dt   = 0.02
    sim_t     = 0.0
    step_cnt  = 0

    joint_pos = np.zeros(20, dtype=np.float32)
    joint_vel = np.zeros(20, dtype=np.float32)

    imu_info = "真實 H30 IMU" if bridge is not None else "假 IMU（直立靜止）"
    print(f"\n=== Policy 模式 | {imu_info} | {'DRY RUN' if args.dry_run else 'LIVE CAN'} ===")
    print("Ctrl+C 停止\n")

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
                    send_cmd(driver, mid, float(actions[idx]),
                             MOTOR_CONFIG[mid]["kp"], MOTOR_CONFIG[mid]["kd"])

            if step_cnt % 50 == 0:
                print(f"[t={sim_t:6.2f}s]", end="")
                for mid in motor_ids:
                    idx    = motor_id_to_policy_idx(mid)
                    name   = MOTOR_CONFIG[mid]["name"].replace("dof_", "")
                    cur    = math.degrees(joint_pos[idx])
                    target = math.degrees(actions[idx])
                    torque = calc_torque(float(actions[idx]), joint_pos[idx],
                                         joint_vel[idx],
                                         MOTOR_CONFIG[mid]["kp"], MOTOR_CONFIG[mid]["kd"],
                                         MAX_TORQUE[MOTOR_CONFIG[mid]["type"]])
                    print(f"  {name}:{cur:5.1f}°→{target:5.1f}° τ={torque:5.1f}Nm", end="")
                print()

            sim_t    += ctrl_dt
            step_cnt += 1
            slp = ctrl_dt - (time.time() - t0)
            if slp > 0:
                time.sleep(slp)

    except KeyboardInterrupt:
        print("\n停止")


# ═══════════════════════════════════════════════════════════════════════════════
# 模式 2：replay  — 從錄製 CSV 重播並驗證
# ═══════════════════════════════════════════════════════════════════════════════

def _latest_recording() -> Path | None:
    """回傳 recordings/ 目錄中最新的 _actions.csv。"""
    if not RECORDINGS_DIR.exists():
        return None
    csvs = sorted(RECORDINGS_DIR.glob("*_actions.csv"))
    return csvs[-1] if csvs else None


def run_replay(args, motor_ids: list, driver, bridge):
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

    print(f"\n=== Replay 模式 ===")
    print(f"錄製: {rec_path}")

    # 載入 CSV
    with open(rec_path, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"  共 {len(rows)} 幀，時長 {float(rows[-1]['time_s']):.2f}s")

    # 載入 policy
    print(f"策略: {args.policy}")
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

    print(f"\n重播中（{'DRY RUN' if args.dry_run else 'LIVE CAN'}）...\n")

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

        # 送馬達
        if not args.dry_run and driver:
            for mid in motor_ids:
                idx = motor_id_to_policy_idx(mid)
                send_cmd(driver, mid, float(clipped[idx]),
                         MOTOR_CONFIG[mid]["kp"], MOTOR_CONFIG[mid]["kd"])

        if row_i % 50 == 0:
            mean_err = np.mean(frame_err) * 180 / math.pi
            print(f"  [{row_i:4d}/{len(rows)}]  t={sim_t:.2f}s  "
                  f"mean_err={mean_err:.2f}°  oob={oob_cnt}  nan={nan_cnt}")

        sim_t += ctrl_dt
        slp = ctrl_dt - (time.time() - t0)
        if slp > 0:
            time.sleep(slp)

    # ── 摘要報告 ────────────────────────────────────────────────────────────
    errors_arr = np.array(errors)   # (N_frames, 10_legs)
    print("\n" + "=" * 60)
    print("=== Replay 摘要報告 ===")
    print(f"  總幀數  : {len(rows)}")
    print(f"  NaN 幀  : {nan_cnt}  {'[FAIL]' if nan_cnt > 0 else '[OK]'}")
    print(f"  超範圍幀: {oob_cnt}  {'[WARN]' if oob_cnt > 0 else '[OK]'}")
    print(f"\n  policy 輸出 vs 錄製 target 偏差（deg）：")
    print(f"  {'關節':<30}  {'均值':>6}  {'最大':>6}  {'P90':>6}")
    for i, name in enumerate(RECORDING_JOINT_NAMES):
        col = errors_arr[:, i] * 180 / math.pi
        print(f"  {name:<30}  {col.mean():6.2f}  {col.max():6.2f}  "
              f"{np.percentile(col, 90):6.2f}")
    overall = errors_arr.flatten() * 180 / math.pi
    print(f"\n  整體均值誤差: {overall.mean():.2f}°  最大: {overall.max():.2f}°")

    passed = nan_cnt == 0 and oob_cnt == 0
    print(f"\n  結果: {'[PASS ✓]' if passed else '[WARN 需確認]'}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# 模式 3：check  — 快速自動驗證
# ═══════════════════════════════════════════════════════════════════════════════

def run_check(args, bridge):
    print(f"\n=== 自動檢查模式（{args.steps} 步） ===")
    print(f"策略: {args.policy}\n")

    init_sess, step_sess, meta = load_kinfer(args.policy)
    num_commands = meta.get("num_commands", 0) or 0
    carry_init   = init_sess.run(None, {})
    carry_size   = meta["carry_size"]
    carry        = carry_init[0] if carry_init else np.zeros(carry_size, dtype=np.float32)

    joint_pos = np.zeros(20, dtype=np.float32)
    joint_vel = np.zeros(20, dtype=np.float32)

    results = {
        "步數": args.steps,
        "NaN/Inf": 0,
        "超安全範圍": 0,
        "錯誤": [],
    }

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

        if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
            results["NaN/Inf"] += 1

        clipped = np.clip(actions, _SAFE_MIN_ARR, _SAFE_MAX_ARR)
        if not np.allclose(actions, clipped, atol=1e-6):
            results["超安全範圍"] += 1

        # 假設靜止：下一步 joint_pos 微移（模擬馬達跟隨）
        for i, name in enumerate(POLICY_JOINT_NAMES):
            if name in SAFE_MIN:
                pid = i
                joint_pos[pid] += (float(actions[pid]) - joint_pos[pid]) * 0.1

        if (step + 1) % 25 == 0:
            range_ok = np.allclose(actions, clipped, atol=1e-6)
            print(f"  step {step+1:4d}  NaN={np.any(np.isnan(actions))}  "
                  f"in_range={range_ok}  "
                  f"leg_targets=[{', '.join(f'{math.degrees(float(actions[_REC_TO_POL[i]])):.1f}°' for i in range(5))}]")

    # 結果
    print("\n" + "=" * 50)
    passed = results["NaN/Inf"] == 0 and not results["錯誤"]
    print(f"  策略檔    : {Path(args.policy).name}")
    print(f"  總步數    : {results['步數']}")
    print(f"  NaN/Inf  : {results['NaN/Inf']}  {'[FAIL]' if results['NaN/Inf'] else '[OK]'}")
    print(f"  超安全範圍: {results['超安全範圍']}  {'[WARN]' if results['超安全範圍'] else '[OK]'}")
    if results["錯誤"]:
        print("  錯誤:")
        for e in results["錯誤"]:
            print(f"    {e}")
    imu_str = "真實 H30 IMU" if bridge else "假 IMU"
    print(f"  IMU 模式  : {imu_str}")
    print(f"\n  結果: {'[PASS ✓]' if passed else '[FAIL ✗]'}")
    print("=" * 50)
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
  check  (預設) — 快速自動驗證 N 步推論
  policy        — 連續推論，可接真實 IMU 和馬達
  replay        — 從錄製 CSV 重播，自動比對 policy 輸出

範例：
  python linux/test/test_policy.py
  python linux/test/test_policy.py --mode check --steps 200
  python linux/test/test_policy.py --mode policy --dry-run
  python linux/test/test_policy.py --mode policy --imu --dry-run
  python linux/test/test_policy.py --mode policy --imu --can can0 --ids 31,32,33,34,35
  python linux/test/test_policy.py --mode replay
  python linux/test/test_policy.py --mode replay --recording {RECORDINGS_DIR}/xxx.csv

可用錄製：
{chr(10).join('  ' + str(p) for p in sorted(RECORDINGS_DIR.glob('*_actions.csv'))[-5:]) if RECORDINGS_DIR.exists() else '  （尚無錄製）'}
        """,
    )

    # 模式
    parser.add_argument("--mode", default="check",
                        choices=["check", "policy", "replay"],
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
                        help="馬達 CAN ID（預設: 31,32,33,34,35）")
    parser.add_argument("--dry-run", action="store_true",
                        help="不送 CAN 指令，只印推論結果")

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
            print(f"[ERROR] 不認識的 motor ID: {mid}（可用：{sorted(MOTOR_CONFIG.keys())}）")
            sys.exit(1)

    # IMU
    bridge = None
    if args.imu:
        bridge = start_imu(args.imu_port, args.imu_baud)

    # 馬達驅動器
    driver = None
    if not args.dry_run and args.mode != "check":
        print(f"連接 CAN: {args.can}")
        driver = setup_driver(args.can, motor_ids)
        print()

    try:
        if args.mode == "check":
            ok = run_check(args, bridge)
            sys.exit(0 if ok else 1)
        elif args.mode == "policy":
            run_policy(args, motor_ids, driver, bridge)
        elif args.mode == "replay":
            run_replay(args, motor_ids, driver, bridge)
    finally:
        if driver:
            print("停用馬達...")
            disable_all(driver, motor_ids)


if __name__ == "__main__":
    main()
