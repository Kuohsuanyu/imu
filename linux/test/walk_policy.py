#!/usr/bin/env python3
"""
fr01 走路策略 — 真機 IMU 即時推論
=====================================
載入 .bin（JAX/equinox）格式的 fr01 走路 checkpoint，
直接對真機做 50Hz 推論，透過 CAN 送位置指令。

使用方式：
  # 安裝 JAX（CPU 推論，不需 GPU）
  conda activate ksim

  # Dry run：不送馬達，驗證觀測 + 輸出是否正常
  python imu/linux/test/walk_policy.py --dry-run

  # 真實 IMU（dry run）
  python imu/linux/test/walk_policy.py --imu --dry-run

  # 真實 IMU + 雙腿馬達（全速上線）
  python imu/linux/test/walk_policy.py --imu --can can0 --left-can can1

  # 指定 checkpoint（預設：imu/policy/walk_500step.bin）
  python imu/linux/test/walk_policy.py --imu --can can0 \\
      --ckpt /home/andykuo/ksim-gym/imu/policy/walk_500step.bin

觀測向量（31維，與訓練完全一致）：
  [0]     sin(sim_t)
  [1]     cos(sim_t)
  [2:12]  joint_pos × 10（右 hip_pitch/roll/yaw/knee/ankle，左 hip_pitch/roll/yaw/knee/ankle）
  [12:22] joint_vel × 10（同上順序）
  [22:25] projected_gravity × 3（機身座標系，由 IMU 四元數計算）
  [25:28] imu_acc × 3
  [28:31] imu_gyro × 3

關節與馬達 ID 對應：
  obs[0]  = right_hip_pitch_04  → motor 41
  obs[1]  = right_hip_roll_03   → motor 42
  obs[2]  = right_hip_yaw_03    → motor 43
  obs[3]  = right_knee_04       → motor 44
  obs[4]  = right_ankle_02      → motor 45
  obs[5]  = left_hip_pitch_04   → motor 31
  obs[6]  = left_hip_roll_03    → motor 32
  obs[7]  = left_hip_yaw_03     → motor 33
  obs[8]  = left_knee_04        → motor 34
  obs[9]  = left_ankle_02       → motor 35
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import math
import os
import socket as _socket
import struct as _struct
import sys
import tarfile
import threading as _threading
import time
from pathlib import Path

import numpy as np

# ── 路徑設定 ─────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent   # imu/linux/test/
_LINUX_IMU = _HERE.parent / "imu"              # imu/linux/imu/
_REPO_ROOT = _HERE.parent.parent               # imu/

# ── 推論用 CPU JAX（上線前不需 GPU）──────────────────────────────────────────
os.environ["JAX_PLATFORMS"] = "cpu"

import equinox as eqx
import jax
import jax.numpy as jnp
import ksim

# ── 預設 checkpoint ──────────────────────────────────────────────────────────
DEFAULT_CKPT = _REPO_ROOT / "policy" / "walk_500step.bin"

# ── 網路超參數（必須與 train_3.py 完全一致）──────────────────────────────────
HIDDEN_SIZE  = 128
DEPTH        = 5
NUM_MIXTURES = 5
VAR_SCALE    = 0.5
NUM_JOINTS   = 10
NUM_ACTOR_INPUTS = 31

# ── 控制頻率 ─────────────────────────────────────────────────────────────────
CTRL_DT = 0.02   # 50 Hz

# ── 關節零點（訓練 ZEROS，與 train_3.py 完全一致）────────────────────────────
# 順序固定：右腿 → 左腿（與訓練觀測向量中 joint_pos 的順序一致）
JOINT_NAMES = [
    "dof_right_hip_pitch_04",
    "dof_right_hip_roll_03",
    "dof_right_hip_yaw_03",
    "dof_right_knee_04",
    "dof_right_ankle_02",
    "dof_left_hip_pitch_04",
    "dof_left_hip_roll_03",
    "dof_left_hip_yaw_03",
    "dof_left_knee_04",
    "dof_left_ankle_02",
]
ZEROS = {
    "dof_right_hip_pitch_04":  math.radians(-20.0),
    "dof_right_hip_roll_03":   0.0,
    "dof_right_hip_yaw_03":    0.0,
    "dof_right_knee_04":       math.radians(-50.0),
    "dof_right_ankle_02":      math.radians(30.0),
    "dof_left_hip_pitch_04":   math.radians(20.0),
    "dof_left_hip_roll_03":    0.0,
    "dof_left_hip_yaw_03":     0.0,
    "dof_left_knee_04":        math.radians(50.0),
    "dof_left_ankle_02":       math.radians(-30.0),
}
ZEROS_ARR = np.array([ZEROS[n] for n in JOINT_NAMES], dtype=np.float32)

# ── 馬達 CAN ID 配置（KP/KD 與 kscale metadata 一致）────────────────────────
# motor_id → joint_obs_idx（上方 JOINT_NAMES 的索引）
MOTOR_CONFIG = {
    41: {"name": "dof_right_hip_pitch_04", "type": "04", "obs_idx": 0, "kp": 150.0, "kd": 24.722},
    42: {"name": "dof_right_hip_roll_03",  "type": "03", "obs_idx": 1, "kp": 200.0, "kd": 26.387},
    43: {"name": "dof_right_hip_yaw_03",   "type": "03", "obs_idx": 2, "kp": 100.0, "kd":  3.419},
    44: {"name": "dof_right_knee_04",      "type": "04", "obs_idx": 3, "kp": 150.0, "kd":  8.654},
    45: {"name": "dof_right_ankle_02",     "type": "02", "obs_idx": 4, "kp":  40.0, "kd":  0.990},
    31: {"name": "dof_left_hip_pitch_04",  "type": "04", "obs_idx": 5, "kp": 150.0, "kd": 24.722},
    32: {"name": "dof_left_hip_roll_03",   "type": "03", "obs_idx": 6, "kp": 200.0, "kd": 26.387},
    33: {"name": "dof_left_hip_yaw_03",    "type": "03", "obs_idx": 7, "kp": 100.0, "kd":  3.419},
    34: {"name": "dof_left_knee_04",       "type": "04", "obs_idx": 8, "kp": 150.0, "kd":  8.654},
    35: {"name": "dof_left_ankle_02",      "type": "02", "obs_idx": 9, "kp":  40.0, "kd":  0.990},
}
MOTOR_IDS = [41, 42, 43, 44, 45, 31, 32, 33, 34, 35]

# 最大力矩：soft_torque_limit（與訓練 PositionActuators 一致，非 MJCF actuatorfrcrange）
MAX_TORQUE = {"04": 84.0, "03": 42.0, "02": 11.9}

# ── 安全位置限制（ZEROS ± margin）────────────────────────────────────────────
_MARGIN_DEG = 55.0
SAFE_MIN_ARR = np.array([
    math.radians(math.degrees(ZEROS[n]) - _MARGIN_DEG) for n in JOINT_NAMES
], dtype=np.float32)
SAFE_MAX_ARR = np.array([
    math.radians(math.degrees(ZEROS[n]) + _MARGIN_DEG) for n in JOINT_NAMES
], dtype=np.float32)

# ── 各型號物理量程（CAN 解碼用）──────────────────────────────────────────────
_MOTOR_PHYS_RANGES = {
    "04": {"angle": (-4 * math.pi, 4 * math.pi), "vel": (-15.0, 15.0)},
    "03": {"angle": (-4 * math.pi, 4 * math.pi), "vel": (-20.0, 20.0)},
    "02": {"angle": (-4 * math.pi, 4 * math.pi), "vel": (-44.0, 44.0)},
}
_CAN_FRAME_SIZE = 16

# ── Home ramp 時間 ────────────────────────────────────────────────────────────
RAMP_DURATION_S = 3.0


# ══════════════════════════════════════════════════════════════════════════════
# Actor 網路（必須與 train_3.py 完全一致）
# ══════════════════════════════════════════════════════════════════════════════

class Actor(eqx.Module):
    input_proj: eqx.nn.Linear
    rnns: tuple
    output_proj: eqx.nn.Linear
    num_inputs:   int   = eqx.static_field()
    num_outputs:  int   = eqx.static_field()
    num_mixtures: int   = eqx.static_field()
    min_std:      float = eqx.static_field()
    max_std:      float = eqx.static_field()
    var_scale:    float = eqx.static_field()

    def __init__(self, key, *, num_inputs, num_outputs, min_std, max_std,
                 var_scale, hidden_size, num_mixtures, depth):
        key, k1 = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(num_inputs, hidden_size, key=k1)
        key, k2 = jax.random.split(key)
        self.rnns = tuple([eqx.nn.GRUCell(hidden_size, hidden_size, key=rk)
                           for rk in jax.random.split(k2, depth)])
        self.output_proj = eqx.nn.Linear(hidden_size, num_outputs * 3 * num_mixtures, key=key)
        self.num_inputs   = num_inputs
        self.num_outputs  = num_outputs
        self.num_mixtures = num_mixtures
        self.min_std  = min_std
        self.max_std  = max_std
        self.var_scale = var_scale

    def forward(self, obs_n, carry):
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)
        sl = self.num_outputs * self.num_mixtures
        mean_nm   = out_n[:sl].reshape(self.num_outputs, self.num_mixtures)
        std_nm    = out_n[sl:sl*2].reshape(self.num_outputs, self.num_mixtures)
        logits_nm = out_n[sl*2:].reshape(self.num_outputs, self.num_mixtures)
        std_nm = jnp.clip(
            (jax.nn.softplus(std_nm) + self.min_std) * self.var_scale,
            max=self.max_std,
        )
        # Actor 輸出以 ZEROS 為偏移（mean 加上站姿零點）
        zeros_jax = jnp.array(ZEROS_ARR)
        mean_nm = mean_nm + zeros_jax[:, None]
        return (
            ksim.MixtureOfGaussians(means_nm=mean_nm, stds_nm=std_nm, logits_nm=logits_nm),
            jnp.stack(out_carries, axis=0),
        )


class Critic(eqx.Module):
    input_proj: eqx.nn.Linear
    rnns: tuple
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()

    def __init__(self, key, *, num_inputs, hidden_size, depth):
        key, k1 = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(num_inputs, hidden_size, key=k1)
        key, k2 = jax.random.split(key)
        self.rnns = tuple([eqx.nn.GRUCell(hidden_size, hidden_size, key=rk)
                           for rk in jax.random.split(k2, depth)])
        self.output_proj = eqx.nn.Linear(hidden_size, 1, key=key)
        self.num_inputs = num_inputs


class Model(eqx.Module):
    actor:  Actor
    critic: Critic

    def __init__(self, key, *, num_actor_inputs, num_actor_outputs, num_critic_inputs,
                 min_std, max_std, var_scale, hidden_size, num_mixtures, depth):
        ak, ck = jax.random.split(key)
        self.actor = Actor(ak, num_inputs=num_actor_inputs, num_outputs=num_actor_outputs,
                           min_std=min_std, max_std=max_std, var_scale=var_scale,
                           hidden_size=hidden_size, num_mixtures=num_mixtures, depth=depth)
        self.critic = Critic(ck, num_inputs=num_critic_inputs,
                             hidden_size=hidden_size, depth=depth)


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint 載入
# ══════════════════════════════════════════════════════════════════════════════

def load_actor(ckpt_path: str | Path) -> Actor:
    """從 xax .bin（tar.gz）checkpoint 載入 Actor。"""
    key = jax.random.PRNGKey(0)
    template = Model(
        key,
        num_actor_inputs=NUM_ACTOR_INPUTS,
        num_actor_outputs=NUM_JOINTS,
        num_critic_inputs=288,   # critic 不用於推論，但需要佔位
        min_std=0.001, max_std=1.0, var_scale=VAR_SCALE,
        hidden_size=HIDDEN_SIZE, num_mixtures=NUM_MIXTURES, depth=DEPTH,
    )
    with tarfile.open(str(ckpt_path), "r:gz") as tar:
        model_file = tar.extractfile("model_0")
        if model_file is None:
            raise ValueError(f"Checkpoint {ckpt_path} 缺少 model_0")
        model = eqx.tree_deserialise_leaves(io.BytesIO(model_file.read()), template)
    print(f"  [OK] 載入 checkpoint: {ckpt_path}")
    return model.actor


# ══════════════════════════════════════════════════════════════════════════════
# JIT 推論
# ══════════════════════════════════════════════════════════════════════════════

@jax.jit
def run_actor_jit(actor: Actor, obs: jax.Array, carry: jax.Array):
    """確定性推論（mode = 最高權重 Gaussian 的均值）。"""
    dist, new_carry = actor.forward(obs, carry)
    action = dist.mode()   # ksim.MixtureOfGaussians.mode()
    return action, new_carry


# ══════════════════════════════════════════════════════════════════════════════
# Raw CAN 讀取器（仿 test_policy.py，無 mismatch 問題）
# ══════════════════════════════════════════════════════════════════════════════

def _can_scale(raw_u16: int, phys_min: float, phys_max: float) -> float:
    return phys_min + (raw_u16 / 65535.0) * (phys_max - phys_min)


class CanStateReader:
    def __init__(self, interfaces: list[str]):
        self._lock  = _threading.Lock()
        self._state: dict = {}
        self._stop  = _threading.Event()
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
                mux = frame[3] & 0x1F
                if mux != 0x02:
                    continue
                motor_id = frame[1]
                if motor_id not in MOTOR_CONFIG:
                    continue
                mtype  = MOTOR_CONFIG[motor_id]["type"]
                ranges = _MOTOR_PHYS_RANGES[mtype]
                raw_angle = _struct.unpack_from(">H", frame, 8)[0]
                raw_vel   = _struct.unpack_from(">H", frame, 10)[0]
                pos = _can_scale(raw_angle, *ranges["angle"])
                vel = _can_scale(raw_vel,   *ranges["vel"])
                with self._lock:
                    self._state[motor_id] = {"pos": pos, "vel": vel}
        finally:
            sock.close()

    def get(self, motor_id: int):
        with self._lock:
            s = self._state.get(motor_id)
        if s is None:
            return None, None
        return s["pos"], s["vel"]

    def stop(self):
        self._stop.set()


_can_reader: "CanStateReader | None" = None


# ══════════════════════════════════════════════════════════════════════════════
# 馬達驅動器
# ══════════════════════════════════════════════════════════════════════════════

def setup_motors(right_can: str, left_can: str) -> dict:
    """初始化馬達：ping → enable → 啟動 CAN reader → 等待位置回報。
    right_can: 右腿 CAN 介面（41-45）
    left_can:  左腿 CAN 介面（31-35）
    """
    from robstride_driver import PyRobstrideDriver, PyRobstrideActuatorType, PyActuatorCommand

    ACTUATOR_TYPE_MAP = {"02": "Robstride02", "03": "Robstride03", "04": "Robstride04"}

    # 左右腿各建一個 driver（可能是同一個 iface）
    iface_drivers: dict = {}
    for iface in sorted(set([right_can, left_can])):
        drv = PyRobstrideDriver(iface)
        drv.connect(iface)
        iface_drivers[iface] = drv

    can_assignment = {}
    for mid in MOTOR_IDS:
        can_assignment[mid] = left_can if 31 <= mid <= 39 else right_can

    driver_map = {}

    print(f"\n  [Phase 1] Ping 馬達（右腿:{right_can}  左腿:{left_can}）")
    for mid in MOTOR_IDS:
        iface = can_assignment[mid]
        drv   = iface_drivers[iface]
        cfg   = MOTOR_CONFIG[mid]
        atype = getattr(PyRobstrideActuatorType, ACTUATOR_TYPE_MAP[cfg["type"]])
        drv.add_actuator(can_id=mid, actuator_type=atype)
        time.sleep(0.05)
        driver_map[mid] = drv
        print(f"    ping {mid:2d} ({cfg['name'].replace('dof_','')[:22]}) [{iface}]")
    time.sleep(0.15)

    print(f"\n  [Phase 2] Enable 馬達")
    for mid in MOTOR_IDS:
        cfg    = MOTOR_CONFIG[mid]
        drv    = driver_map[mid]
        repeat = 3 if cfg["type"] == "04" else 1
        for _ in range(repeat):
            try:
                drv.enable_actuator(actuator_id=mid)
            except Exception:
                pass
            time.sleep(0.1 if cfg["type"] == "04" else 0.05)
        print(f"    enable {mid:2d} ({cfg['name'].replace('dof_','')})")
    time.sleep(0.3)

    # 啟動 raw CAN reader（兩個介面）
    global _can_reader
    ifaces = sorted(set([right_can, left_can]))
    _can_reader = CanStateReader(ifaces)
    print(f"\n  [CAN Reader] 已啟動 raw 幀讀取: {ifaces}")

    def _prime_once():
        for mid in MOTOR_IDS:
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

    # 等最多 2 秒；無回應的馬達重新 enable 再 prime（最多 3 輪）
    for attempt in range(3):
        t_end = time.time() + 2.0
        while time.time() < t_end:
            if all(_can_reader.get(mid)[0] is not None for mid in MOTOR_IDS):
                break
            time.sleep(0.05)
        missing = [mid for mid in MOTOR_IDS if _can_reader.get(mid)[0] is None]
        if not missing:
            break
        print(f"  [嘗試 {attempt+1}/3] 未回應馬達重新 enable: {missing}")
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

    ready   = [mid for mid in MOTOR_IDS if _can_reader.get(mid)[0] is not None]
    missing = [mid for mid in MOTOR_IDS if _can_reader.get(mid)[0] is None]
    print(f"  位置回報：{len(ready)}/{len(MOTOR_IDS)} 顆")
    if missing:
        print(f"  [WARN] 以下馬達未回報（可能在保護模式）: {missing}")

    return driver_map


def send_cmd(driver_map, mid: int, position: float, kp: float, kd: float):
    from robstride_driver import PyActuatorCommand
    driver_map[mid].send_command(
        actuator_id=mid,
        command=PyActuatorCommand(position=position, velocity=0.0, torque=0.0, kp=kp, kd=kd),
    )


def disable_all(driver_map):
    from robstride_driver import PyActuatorCommand
    for mid in MOTOR_IDS:
        try:
            driver_map[mid].send_command(
                actuator_id=mid,
                command=PyActuatorCommand(position=0.0, velocity=0.0, torque=0.0, kp=0.0, kd=0.0),
            )
        except Exception:
            pass


def read_states(joint_pos: np.ndarray, joint_vel: np.ndarray):
    """從 CanStateReader 更新 joint_pos / joint_vel（10D）。"""
    if _can_reader is None:
        return
    for mid, cfg in MOTOR_CONFIG.items():
        pos, vel = _can_reader.get(mid)
        if pos is not None:
            joint_pos[cfg["obs_idx"]] = pos
            joint_vel[cfg["obs_idx"]] = vel


def home_ramp(driver_map, joint_pos: np.ndarray, joint_vel: np.ndarray):
    """從當前位置緩慢插值到 ZEROS 站姿（RAMP_DURATION_S 秒）。"""
    # 讀取起始位置
    print("  等待馬達位置資料...", end="", flush=True)
    t_wait = time.time() + 3.0
    while time.time() < t_wait:
        if _can_reader is not None and all(
                _can_reader.get(mid)[0] is not None for mid in MOTOR_IDS):
            break
        time.sleep(0.05)
    print(" 完成")

    read_states(joint_pos, joint_vel)
    start_pos = joint_pos.copy()

    print(f"\n  [Home Ramp] {RAMP_DURATION_S:.0f}秒緩移至站姿...")
    print("  起點: " + "  ".join(
        f"{MOTOR_CONFIG[m]['name'].replace('dof_','')[:10]}={math.degrees(start_pos[MOTOR_CONFIG[m]['obs_idx']]):+.1f}°"
        for m in MOTOR_IDS))
    print("  目標: " + "  ".join(
        f"{MOTOR_CONFIG[m]['name'].replace('dof_','')[:10]}={math.degrees(ZEROS_ARR[MOTOR_CONFIG[m]['obs_idx']]):+.0f}°"
        for m in MOTOR_IDS))

    t_start = time.time()
    step    = 0
    while True:
        t0      = time.time()
        elapsed = t0 - t_start
        alpha   = min(elapsed / RAMP_DURATION_S, 1.0)

        for mid in MOTOR_IDS:
            idx    = MOTOR_CONFIG[mid]["obs_idx"]
            interp = start_pos[idx] + alpha * (ZEROS_ARR[idx] - start_pos[idx])
            send_cmd(driver_map, mid, interp, MOTOR_CONFIG[mid]["kp"], MOTOR_CONFIG[mid]["kd"])

        # 更新實際位置（只用於顯示）
        read_states(joint_pos, joint_vel)

        if step % 25 == 0:
            print(f"  [{elapsed:4.1f}/{RAMP_DURATION_S:.0f}s]  " +
                  "  ".join(f"{MOTOR_CONFIG[m]['name'].replace('dof_','')[:8]}"
                             f"={math.degrees(joint_pos[MOTOR_CONFIG[m]['obs_idx']]):+.1f}°"
                             for m in MOTOR_IDS))

        if alpha >= 1.0:
            print(f"  [OK] Home Ramp 完成（{elapsed:.1f}s）")
            break

        step += 1
        slp = CTRL_DT - (time.time() - t0)
        if slp > 0:
            time.sleep(slp)


# ══════════════════════════════════════════════════════════════════════════════
# IMU bridge
# ══════════════════════════════════════════════════════════════════════════════

def start_imu(imu_port: str = "/dev/ttyACM0", imu_baud: int = 460800):
    """啟動 H30 bridge thread，等待第一筆 IMU 資料。"""
    bridge_path = _LINUX_IMU / "bridge_h30.py"
    if not bridge_path.exists():
        raise FileNotFoundError(f"找不到 IMU bridge: {bridge_path}")

    spec   = importlib.util.spec_from_file_location("bridge", bridge_path)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)

    def _run():
        bridge.run_imu_only(imu_port, imu_baud)

    t = _threading.Thread(target=_run, daemon=True, name="h30-bridge")
    t.start()

    print(f"  [IMU] 啟動 H30 bridge {imu_port} @ {imu_baud} baud，等待資料...")
    for _ in range(200):
        time.sleep(0.1)
        with bridge._imu_lock:
            if bridge.IMU_STATE["updated"]:
                print(f"  [OK] IMU 就緒（{imu_port}）")
                return bridge
    print("  [WARN] 20 秒內未收到 IMU 資料，使用假值繼續")
    return bridge


def get_imu_obs(bridge) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """從 bridge 讀取 IMU 狀態，回傳 (proj_grav, imu_acc, imu_gyro)。"""
    with bridge._imu_lock:
        acc  = bridge.IMU_STATE["acc"].copy().astype(np.float32)
        gyro = bridge.IMU_STATE["gyro"].copy().astype(np.float32)
        quat = bridge.IMU_STATE["quat"].copy()   # [qw, qx, qy, qz]
    proj_grav = np.array(bridge.proj_gravity_from_quat(*quat), dtype=np.float32)
    return proj_grav, acc, gyro


def fake_imu_obs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """假 IMU：直立靜止狀態。"""
    proj_grav = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    imu_acc   = np.array([0.0, 0.0, -9.81], dtype=np.float32)
    imu_gyro  = np.zeros(3, dtype=np.float32)
    return proj_grav, imu_acc, imu_gyro


# ══════════════════════════════════════════════════════════════════════════════
# 主策略迴圈
# ══════════════════════════════════════════════════════════════════════════════

def run_walk_policy(args):
    print("\n" + "═" * 60)
    print("  fr01 走路策略 — 真機推論")
    print("═" * 60)

    # ── 載入 Actor ──────────────────────────────────────────────────
    print(f"\n  載入 checkpoint: {args.ckpt}")
    actor = load_actor(args.ckpt)

    # JIT 編譯（避免第一步卡頓）
    print("  JIT 編譯 Actor...", end="", flush=True)
    dummy_obs   = jnp.zeros(NUM_ACTOR_INPUTS)
    dummy_carry = jnp.zeros((DEPTH, HIDDEN_SIZE))
    _, _ = run_actor_jit(actor, dummy_obs, dummy_carry)
    print(" 完成")

    carry = jnp.zeros((DEPTH, HIDDEN_SIZE))

    # ── IMU ──────────────────────────────────────────────────────────
    if args.imu:
        print(f"\n  啟動 IMU bridge...")
        bridge = start_imu(args.imu_port, args.imu_baud)
    else:
        bridge = None
        print("  [IMU] 使用假值（直立靜止）")

    # ── 馬達 ─────────────────────────────────────────────────────────
    driver_map = None
    if not args.dry_run and args.can:
        left_can = args.left_can or args.can
        print(f"\n  設定馬達（右腿:{args.can}  左腿:{left_can}）...")
        driver_map = setup_motors(args.can, left_can)
    elif args.dry_run:
        print("  [DRY RUN] 不送馬達指令")
    else:
        print("  [WARN] 未指定 --can，不送馬達指令（等同 --dry-run）")

    # ── 觀測狀態 ─────────────────────────────────────────────────────
    joint_pos = ZEROS_ARR.copy()
    joint_vel = np.zeros(NUM_JOINTS, dtype=np.float32)

    # ── Home Ramp ────────────────────────────────────────────────────
    if driver_map and not args.skip_home_ramp:
        home_ramp(driver_map, joint_pos, joint_vel)
        input("\n  [確認] 已到站姿，按 Enter 開始 Policy 推論...")
    elif args.skip_home_ramp and driver_map:
        print("  [WARN] --skip-home-ramp：確認機器人已手動擺到站姿")

    # ── 推論迴圈 ─────────────────────────────────────────────────────
    print(f"\n  {'═'*50}")
    print(f"  開始 Policy 推論（50 Hz）  Ctrl+C 停止")
    print(f"  {'═'*50}\n")

    sim_t     = 0.0
    step_cnt  = 0
    loop_lags = []

    try:
        while True:
            t0 = time.time()

            # 1. 讀取馬達狀態
            if driver_map:
                read_states(joint_pos, joint_vel)

            # 2. 讀取 IMU
            if bridge is not None:
                proj_grav, imu_acc, imu_gyro = get_imu_obs(bridge)
            else:
                proj_grav, imu_acc, imu_gyro = fake_imu_obs()

            # 3. 組裝 31D 觀測向量
            obs_np = np.concatenate([
                np.array([math.sin(sim_t)], dtype=np.float32),
                np.array([math.cos(sim_t)], dtype=np.float32),
                joint_pos,
                joint_vel,
                proj_grav,
                imu_acc,
                imu_gyro,
            ])  # shape (31,)

            # 4. Actor 推論
            obs_jax  = jnp.array(obs_np)
            action_jax, carry = run_actor_jit(actor, obs_jax, carry)
            target_pos = np.clip(np.array(action_jax, dtype=np.float32),
                                 SAFE_MIN_ARR, SAFE_MAX_ARR)

            # 5. 送馬達指令
            if driver_map and not args.dry_run:
                for mid in MOTOR_IDS:
                    idx = MOTOR_CONFIG[mid]["obs_idx"]
                    send_cmd(driver_map, mid, float(target_pos[idx]),
                             MOTOR_CONFIG[mid]["kp"], MOTOR_CONFIG[mid]["kd"])

            # 6. 顯示（每 50 步 = 1 秒）
            if step_cnt % 50 == 0:
                lag_mean = np.mean(loop_lags[-50:]) * 1000 if loop_lags else 0.0
                print(f"\n[t={sim_t:6.2f}s  step={step_cnt}  loop_lag={lag_mean:.1f}ms]")
                print(f"  proj_grav=[{proj_grav[0]:+.3f} {proj_grav[1]:+.3f} {proj_grav[2]:+.3f}]  "
                      f"gyro=[{imu_gyro[0]:+.3f} {imu_gyro[1]:+.3f} {imu_gyro[2]:+.3f}]")
                for mid in MOTOR_IDS:
                    cfg    = MOTOR_CONFIG[mid]
                    idx    = cfg["obs_idx"]
                    cur    = joint_pos[idx]
                    vel    = joint_vel[idx]
                    tgt    = float(target_pos[idx])
                    max_t  = MAX_TORQUE[cfg["type"]]
                    tau    = cfg["kp"] * (tgt - cur) + cfg["kd"] * (-vel)
                    tau    = max(-max_t, min(max_t, tau))
                    pct    = abs(tau) / max_t * 100
                    flag   = " !OVER" if pct >= args.torque_limit * 100 else ""
                    print(f"  {mid:2d} {cfg['name'].replace('dof_',''):<26}"
                          f"  pos={math.degrees(cur):+6.1f}°"
                          f"  tgt={math.degrees(tgt):+6.1f}°"
                          f"  τ={tau:+7.2f}Nm ({pct:4.1f}%){flag}")

            # 7. 計時 & 等待
            elapsed = time.time() - t0
            loop_lags.append(elapsed)
            sim_t    += CTRL_DT
            step_cnt += 1
            slp = CTRL_DT - elapsed
            if slp > 0:
                time.sleep(slp)

    except KeyboardInterrupt:
        print("\n\n  停止")
    finally:
        if driver_map:
            print("  [安全] disable 所有馬達...")
            disable_all(driver_map)
        if _can_reader:
            _can_reader.stop()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="fr01 走路策略 — JAX/equinox .bin checkpoint → 真機推論",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt", default=str(DEFAULT_CKPT),
        help=f"checkpoint 路徑（預設: {DEFAULT_CKPT}）",
    )
    parser.add_argument(
        "--can", default=None, metavar="IFACE",
        help="右腿 CAN 介面（41-45），例如 can0（不指定則 dry-run）",
    )
    parser.add_argument(
        "--left-can", default=None, metavar="IFACE",
        help="左腿 CAN 介面（31-35），例如 can1（預設與 --can 相同）",
    )
    parser.add_argument(
        "--imu", action="store_true",
        help="啟動真實 H30 IMU（否則使用假 IMU 直立值）",
    )
    parser.add_argument(
        "--imu-port", default="/dev/ttyACM0",
        help="H30 IMU serial port（預設: /dev/ttyACM0）",
    )
    parser.add_argument(
        "--imu-baud", type=int, default=460800,
        help="H30 IMU baud rate（預設: 460800）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="不送任何馬達指令，只顯示推論輸出",
    )
    parser.add_argument(
        "--skip-home-ramp", action="store_true",
        help="跳過 home ramp（假設機器人已手動到站姿）",
    )
    parser.add_argument(
        "--torque-limit", type=float, default=0.8,
        help="扭矩警告閾值（比例 0~1，預設 0.8 = 80%%）",
    )
    args = parser.parse_args()

    # 驗證 checkpoint
    if not Path(args.ckpt).exists():
        print(f"[ERROR] checkpoint 不存在: {args.ckpt}")
        print(f"  請先確認檔案位置，或用 --ckpt 指定路徑")
        sys.exit(1)

    run_walk_policy(args)


if __name__ == "__main__":
    main()
