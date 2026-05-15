#!/usr/bin/env python3
"""
馬達測試腳本 — 三種模式

  zero    : 所有馬達歸零位（組裝確認、上電檢查）
  policy  : 載入 .kinfer 模型 + 假 IMU 推論（直立靜止）
  sine    : 正弦波持續運動（確認馬達響應與扭矩輸出）

使用方式：
  # dry run（只印指令，不送 CAN）
  python test_motor_policy.py --mode zero   --dry_run
  python test_motor_policy.py --mode policy --dry_run
  python test_motor_policy.py --mode sine   --dry_run

  # 指定馬達（預設：左腿全部 31~35）
  python test_motor_policy.py --mode zero   --ids 31,32,33,34,35

  # 兩腿全部
  python test_motor_policy.py --mode policy --ids 31,32,33,34,35,41,42,43,44,45

  # 自訂 policy 路徑
  python test_motor_policy.py --mode policy --policy /path/to/my.kinfer
"""

import argparse
import json
import math
import os
import sys
import tarfile
import tempfile
import time

import numpy as np
import onnxruntime as ort

# 真實 IMU 支援（--imu 旗標啟用）
_bridge = None   # 延遲 import，避免沒有 bleak 時報錯

# ── 策略關節順序（20-dim）────────────────────────────────────────────────────
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

# CAN ID → 關節名稱、型號、PD 增益
MOTOR_CONFIG = {
    # 左腿
    31: {"name": "dof_left_hip_pitch_04",  "type": "04", "kp": 150.0, "kd": 24.722},
    32: {"name": "dof_left_hip_roll_03",   "type": "03", "kp": 200.0, "kd": 26.387},
    33: {"name": "dof_left_hip_yaw_03",    "type": "03", "kp": 100.0, "kd":  3.419},
    34: {"name": "dof_left_knee_04",       "type": "04", "kp": 150.0, "kd":  8.654},
    35: {"name": "dof_left_ankle_02",      "type": "02", "kp":  40.0, "kd":  0.990},
    # 右腿
    41: {"name": "dof_right_hip_pitch_04", "type": "04", "kp": 150.0, "kd": 24.722},
    42: {"name": "dof_right_hip_roll_03",  "type": "03", "kp": 200.0, "kd": 26.387},
    43: {"name": "dof_right_hip_yaw_03",   "type": "03", "kp": 100.0, "kd":  3.419},
    44: {"name": "dof_right_knee_04",      "type": "04", "kp": 150.0, "kd":  8.654},
    45: {"name": "dof_right_ankle_02",     "type": "02", "kp":  40.0, "kd":  0.990},
}

ACTUATOR_TYPE_MAP = {"02": "Robstride02", "03": "Robstride03", "04": "Robstride04"}
MAX_TORQUE        = {"04": 84.0, "03": 42.0, "02": 11.9}

# 正弦波參數（各關節）
SINE_AMP_RAD = {
    "04": math.radians(15),   # hip pitch / knee：±15°
    "03": math.radians(10),   # hip roll / yaw：±10°
    "02": math.radians(8),    # ankle：±8°
}
SINE_FREQ_HZ = 0.3   # 0.3 Hz，慢速確認響應


# ── 工具函數 ──────────────────────────────────────────────────────────────────
def motor_id_to_policy_index(mid: int) -> int:
    return POLICY_JOINT_NAMES.index(MOTOR_CONFIG[mid]["name"])

def load_kinfer(path: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(tmpdir)
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            metadata = json.load(f)
        init_sess = ort.InferenceSession(os.path.join(tmpdir, "init_fn.onnx"))
        step_sess = ort.InferenceSession(os.path.join(tmpdir, "step_fn.onnx"))
    return init_sess, step_sess, metadata

def build_policy_feed(step_sess, joint_pos, joint_vel, carry, num_commands, sim_t):
    """IMU 資料 → policy 輸入。有真實 IMU 時讀 bridge.IMU_STATE，否則用假值。"""
    names = {i.name for i in step_sess.get_inputs()}
    feed = {
        "joint_angles":             joint_pos.astype(np.float32),
        "joint_angular_velocities": joint_vel.astype(np.float32),
        "carry":                    carry,
    }

    if _bridge is not None:
        with _bridge._imu_lock:
            acc  = _bridge.IMU_STATE["acc"].copy()
            gyro = _bridge.IMU_STATE["gyro"].copy()
            quat = _bridge.IMU_STATE["quat"].copy()
        proj_grav = _bridge.proj_gravity_from_quat(*quat)
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


# ── 驅動器管理 ────────────────────────────────────────────────────────────────
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
    print("停用馬達...")
    for mid in motor_ids:
        try:
            driver.send_command(
                actuator_id=mid,
                command=PyActuatorCommand(position=0.0, velocity=0.0, torque=0.0, kp=0.0, kd=0.0),
            )
        except Exception:
            pass

def read_states(driver, motor_ids: list, joint_pos, joint_vel, id_to_idx: dict):
    for mid in motor_ids:
        idx = id_to_idx[mid]
        try:
            s = driver.get_actuator_state(actuator_id=mid)
            joint_pos[idx] = s.position
            joint_vel[idx] = s.velocity
        except Exception as e:
            print(f"[WARN] 讀取馬達 {mid} 失敗: {e}")


# ── 模式：零位 ────────────────────────────────────────────────────────────────
def run_zero(args, motor_ids: list, driver):
    """
    將所有指定馬達保持在零位。
    用途：組裝確認關節中位、上電安全檢查。
    """
    ctrl_dt  = 0.02   # 50 Hz
    step_cnt = 0

    print("\n=== 零位模式（Ctrl+C 停止）===")
    if args.dry_run:
        print("  [DRY RUN：只印資訊，不送 CAN]\n")

    joint_pos = np.zeros(20, dtype=np.float32)
    joint_vel = np.zeros(20, dtype=np.float32)
    id_to_idx = {mid: motor_id_to_policy_index(mid) for mid in motor_ids}

    try:
        while True:
            t0 = time.time()

            if driver:
                read_states(driver, motor_ids, joint_pos, joint_vel, id_to_idx)

            for mid in motor_ids:
                cfg = MOTOR_CONFIG[mid]
                if not args.dry_run:
                    send_cmd(driver, mid, 0.0, cfg["kp"], cfg["kd"])

            if step_cnt % 50 == 0:
                print(f"[t={step_cnt*ctrl_dt:6.1f}s]", end="")
                for mid in motor_ids:
                    idx  = id_to_idx[mid]
                    name = MOTOR_CONFIG[mid]["name"].replace("dof_", "")
                    cur  = math.degrees(joint_pos[idx])
                    if args.dry_run:
                        t = calc_torque(0.0, joint_pos[idx], joint_vel[idx],
                                        MOTOR_CONFIG[mid]["kp"], MOTOR_CONFIG[mid]["kd"],
                                        MAX_TORQUE[MOTOR_CONFIG[mid]["type"]])
                        print(f"  {name}: {cur:5.1f}° τ={t:6.1f}Nm", end="")
                    else:
                        print(f"  {name}: {cur:5.1f}°", end="")
                print()

            step_cnt += 1
            sleep = ctrl_dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\n停止")


# ── 模式：policy ──────────────────────────────────────────────────────────────
def run_policy(args, motor_ids: list, driver):
    """
    載入 .kinfer 模型，假 IMU（直立），送指令到馬達。
    """
    print(f"\n載入策略: {args.policy}")
    init_sess, step_sess, metadata = load_kinfer(args.policy)
    input_names  = [i.name for i in step_sess.get_inputs()]
    num_commands = metadata.get("num_commands", 0) or 0
    carry_size   = metadata["carry_size"]
    print(f"  輸入: {input_names}")
    print(f"  carry_size={carry_size}  num_commands={num_commands}")

    carry_init = init_sess.run(None, {})
    carry = carry_init[0] if carry_init else np.zeros(carry_size, dtype=np.float32)

    id_to_idx = {mid: motor_id_to_policy_index(mid) for mid in motor_ids}
    ctrl_dt   = 0.02
    sim_t     = 0.0
    step_cnt  = 0

    joint_pos = np.zeros(20, dtype=np.float32)
    joint_vel = np.zeros(20, dtype=np.float32)

    print("\n=== Policy 模式（假 IMU，Ctrl+C 停止）===")
    if args.dry_run:
        print("  [DRY RUN：只印資訊，不送 CAN]\n")

    try:
        while True:
            t0 = time.time()

            if driver:
                read_states(driver, motor_ids, joint_pos, joint_vel, id_to_idx)

            feed    = build_policy_feed(step_sess, joint_pos, joint_vel, carry, num_commands, sim_t)
            outputs = step_sess.run(None, feed)
            actions = outputs[0]
            carry   = outputs[1]

            for mid in motor_ids:
                idx    = id_to_idx[mid]
                target = float(actions[idx])
                cfg    = MOTOR_CONFIG[mid]
                if args.dry_run:
                    pass   # 只在印出時計算
                else:
                    send_cmd(driver, mid, target, cfg["kp"], cfg["kd"])

            if step_cnt % 50 == 0:
                print(f"[t={sim_t:6.2f}s]", end="")
                for mid in motor_ids:
                    idx    = id_to_idx[mid]
                    target = float(actions[idx])
                    cur    = joint_pos[idx]
                    cfg    = MOTOR_CONFIG[mid]
                    name   = cfg["name"].replace("dof_", "")
                    torque = calc_torque(target, cur, joint_vel[idx],
                                         cfg["kp"], cfg["kd"], MAX_TORQUE[cfg["type"]])
                    print(f"  {name}: {math.degrees(cur):5.1f}°→{math.degrees(target):5.1f}° τ={torque:5.1f}Nm", end="")
                print()

            sim_t    += ctrl_dt
            step_cnt += 1
            sleep = ctrl_dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\n停止")


# ── 模式：sine ────────────────────────────────────────────────────────────────
def run_sine(args, motor_ids: list, driver):
    """
    正弦波運動：確認每顆馬達的響應與扭矩輸出。
    不需要 policy 檔案。
    """
    ctrl_dt   = 0.02
    step_cnt  = 0
    omega     = 2 * math.pi * SINE_FREQ_HZ

    id_to_idx = {mid: motor_id_to_policy_index(mid) for mid in motor_ids}
    joint_pos = np.zeros(20, dtype=np.float32)
    joint_vel = np.zeros(20, dtype=np.float32)

    print(f"\n=== 正弦波模式 ±{SINE_FREQ_HZ} Hz（Ctrl+C 停止）===")
    if args.dry_run:
        print("  [DRY RUN：只印資訊，不送 CAN]\n")

    try:
        while True:
            t0  = time.time()
            sim_t = step_cnt * ctrl_dt

            if driver:
                read_states(driver, motor_ids, joint_pos, joint_vel, id_to_idx)

            for mid in motor_ids:
                idx  = id_to_idx[mid]
                cfg  = MOTOR_CONFIG[mid]
                amp  = SINE_AMP_RAD[cfg["type"]]
                target = amp * math.sin(omega * sim_t)
                if not args.dry_run:
                    send_cmd(driver, mid, target, cfg["kp"], cfg["kd"])

            if step_cnt % 50 == 0:
                print(f"[t={sim_t:6.1f}s]", end="")
                for mid in motor_ids:
                    idx  = id_to_idx[mid]
                    cfg  = MOTOR_CONFIG[mid]
                    amp  = SINE_AMP_RAD[cfg["type"]]
                    tgt  = amp * math.sin(omega * sim_t)
                    cur  = joint_pos[idx]
                    name = cfg["name"].replace("dof_", "")
                    torque = calc_torque(tgt, cur, joint_vel[idx],
                                         cfg["kp"], cfg["kd"], MAX_TORQUE[cfg["type"]])
                    print(f"  {name}: tgt={math.degrees(tgt):5.1f}° cur={math.degrees(cur):5.1f}° τ={torque:5.1f}Nm", end="")
                print()

            step_cnt += 1
            sleep = ctrl_dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\n停止")


# ── 主程式 ────────────────────────────────────────────────────────────────────
def main():
    global _bridge

    default_policy = "/home/andykuo/ksim-gym/kbot_robot/Policies/kbot_zero_position.kinfer"

    parser = argparse.ArgumentParser(
        description="馬達測試腳本（支援真實 IMU）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
模式說明：
  zero    — 所有馬達歸零位（組裝確認、上電安全測試）
  policy  — 載入 .kinfer 模型推論（預設假 IMU，--imu 啟用真實）
  sine    — 正弦波運動測試（不需要 policy）

範例：
  python test_motor_policy.py --mode zero   --dry_run
  python test_motor_policy.py --mode policy --dry_run --ids 31,32,33,34,35
  python test_motor_policy.py --mode policy --imu --dry_run          # 真實 IMU + 假馬達
  python test_motor_policy.py --mode policy --imu --can can0 --ids 31,32,33,34,35,41,42,43,44,45
        """,
    )
    parser.add_argument("--mode", default="policy",
                        choices=["zero", "policy", "sine"],
                        help="測試模式（預設: policy）")
    parser.add_argument("--policy", default=default_policy,
                        help=f"kinfer 路徑（policy 模式用，預設: {default_policy}）")
    parser.add_argument("--can", default="can0",
                        help="CAN 介面（預設: can0）")
    parser.add_argument("--ids", default="31,32,33,34,35",
                        help="馬達 CAN ID，逗號分隔（預設: 31,32,33,34,35）")
    parser.add_argument("--dry_run", action="store_true",
                        help="不接馬達，只印推論結果")
    parser.add_argument("--imu", action="store_true",
                        help="啟用真實 IMU（WT901BLE67 via BLE）")
    parser.add_argument("--imu-name", default="WT901BLE67",
                        help="BLE 裝置名稱（預設: WT901BLE67）")
    args = parser.parse_args()

    # 啟動 BLE IMU（背景 thread）
    if args.imu:
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location(
            "bridge",
            pathlib.Path(__file__).parent / "bridge.py"
        )
        _bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_bridge)
        print(f"[IMU] 啟動 BLE bridge，搜尋 '{args.imu_name}'...")
        _bridge.start_ble_thread(target_name=args.imu_name, use_serial=False)
        print("[IMU] 等待第一筆 IMU 資料（最多 20 秒）...")
        for _ in range(200):
            time.sleep(0.1)
            with _bridge._imu_lock:
                if _bridge.IMU_STATE["updated"]:
                    break
        else:
            print("[WARN] 未收到 IMU 資料，使用假值繼續")
        print("[IMU] 就緒")

    # 解析馬達 ID
    motor_ids = [int(x) for x in args.ids.split(",")]
    for mid in motor_ids:
        if mid not in MOTOR_CONFIG:
            print(f"[ERROR] 不認識的 motor ID: {mid}（可用：{sorted(MOTOR_CONFIG.keys())}）")
            sys.exit(1)

    # 建立驅動器
    driver = None
    if not args.dry_run:
        print(f"連接 CAN: {args.can}")
        driver = setup_driver(args.can, motor_ids)
        print()

    # 執行模式
    try:
        if args.mode == "zero":
            run_zero(args, motor_ids, driver)
        elif args.mode == "policy":
            run_policy(args, motor_ids, driver)
        elif args.mode == "sine":
            run_sine(args, motor_ids, driver)
    finally:
        if driver:
            disable_all(driver, motor_ids)
            print("完成")


if __name__ == "__main__":
    main()
