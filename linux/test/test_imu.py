#!/usr/bin/env python3
"""
WHEELTEC H30 Mini IMU 純測試腳本
==================================
不需要 ONNX / policy / CAN / 馬達 / tty0tty。
只要把 H30 Mini 插上 USB，就可以立即執行。

功能：
  - 自動偵測 H30 串口
  - 即時顯示 acc / gyro / quat / projected_gravity
  - 自動健康檢查（|acc| ≈ 9.81, |quat| ≈ 1.0, 資料率 > 50 Hz）
  - Ctrl+C 結束並印出統計摘要

使用方式：
  python3 linux/test/test_imu.py
  python3 linux/test/test_imu.py --port /dev/ttyACM0
  python3 linux/test/test_imu.py --duration 30   # 跑 30 秒後自動結束

依賴：pyserial, numpy（sudo pip3 install pyserial numpy）
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from pathlib import Path

# ── 自動加入 linux/imu/ 到路徑，重用 bridge_h30 的解析函數 ────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "imu"))

import bridge_h30 as _bridge   # 只借用 _parse_yis_frame, 常數，不啟動虛擬串口

import numpy as np

try:
    import serial
except ImportError:
    print("[ERROR] 缺少 pyserial → sudo pip3 install pyserial")
    sys.exit(1)

# ── 候選串口 ──────────────────────────────────────────────────────────────────
CANDIDATE_PORTS = [
    "/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2", "/dev/ttyACM3",
    "/dev/ttyACM4", "/dev/ttyACM5", "/dev/ttyACM6", "/dev/ttyACM7",
    "/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2",
]
IMU_BAUD        = 460800

# ── 健康閾值 ──────────────────────────────────────────────────────────────────
ACC_NORM_MIN  = 7.0    # m/s²  靜止時 |acc| 應在此範圍
ACC_NORM_MAX  = 12.0
QUAT_NORM_MIN = 0.99   # 四元數正規化
QUAT_NORM_MAX = 1.01
MIN_RATE_HZ   = 50.0   # 最低可接受資料率


# ── 終端工具 ──────────────────────────────────────────────────────────────────
_use_color = sys.stdout.isatty()
def _g(s): return f"\033[32m{s}\033[0m" if _use_color else s
def _y(s): return f"\033[33m{s}\033[0m" if _use_color else s
def _r(s): return f"\033[31m{s}\033[0m" if _use_color else s
def _b(s): return f"\033[1m{s}\033[0m"  if _use_color else s


def auto_detect_port() -> str | None:
    for p in CANDIDATE_PORTS:
        if Path(p).exists():
            return p
    return None


def proj_gravity(qw, qx, qy, qz):
    """重力方向在 body frame（H30 NED 四元數慣例）。"""
    r02 = 2 * (qx * qz - qw * qy)
    r12 = 2 * (qy * qz + qw * qx)
    r22 = qw**2 - qx**2 - qy**2 + qz**2
    return r02, r12, r22


def health_str(val, lo, hi):
    if lo <= val <= hi:
        return _g("OK")
    elif lo * 0.8 <= val <= hi * 1.2:
        return _y("WARN")
    else:
        return _r("FAIL")


# ═══════════════════════════════════════════════════════════════════════════════

def run(port: str, duration: float | None):
    print(_b(f"\n=== H30 Mini IMU 測試 ==="))
    print(f"串口: {port}  波特率: {IMU_BAUD}\n")

    try:
        ser = serial.Serial(port, IMU_BAUD, timeout=0.01)
    except serial.SerialException as e:
        print(_r(f"[ERROR] 無法開啟 {port}: {e}"))
        print("  → 確認 H30 Mini 已插入，並執行 sudo chmod 666 " + port)
        sys.exit(1)

    buf         = bytearray()
    frame_count = 0
    last_data: dict = {}
    t_start     = time.time()
    t_last_stat = t_start
    t_last_frame= t_start
    rate_history: list[float] = []

    # 統計
    acc_norms:  list[float] = []
    quat_norms: list[float] = []
    intervals:  list[float] = []
    last_frame_t = None

    print(f"{'時間':>6}  {'|acc|(m/s²)':>12}  "
          f"{'gyro(rad/s)':>30}  {'|quat|':>8}  {'proj_grav':>22}  {'Hz':>6}")
    print("-" * 100)

    try:
        while True:
            now = time.time()
            elapsed = now - t_start
            if duration and elapsed >= duration:
                break

            chunk = ser.read_all() or ser.read(512)
            if not chunk:
                time.sleep(0.001)
                continue

            buf.extend(chunk)
            pos = 0

            while pos < len(buf) - _bridge.PROTOCOL_MIN_LEN:
                if buf[pos] != _bridge.YIS_H1 or buf[pos + 1] != _bridge.YIS_H2:
                    pos += 1
                    continue
                payload_len = buf[pos + _bridge.PROTOCOL_LEN_POS]
                if pos + _bridge.PROTOCOL_MIN_LEN + payload_len > len(buf):
                    break
                result, next_pos = _bridge._parse_yis_frame(buf, pos)
                if result is None:
                    pos += 1
                    continue

                last_data.update(result)
                frame_count += 1

                # 幀間隔統計
                if last_frame_t is not None:
                    intervals.append(now - last_frame_t)
                last_frame_t = now

                pos = next_pos

            buf = buf[pos:]

            # 每 0.5 秒刷新一次顯示
            if now - t_last_stat >= 0.5 and last_data:
                acc  = last_data.get('acc',  (0.0, 0.0, -9.81))
                gyro = last_data.get('gyro', (0.0, 0.0, 0.0))
                quat = last_data.get('quat', (1.0, 0.0, 0.0, 0.0))

                acc_n  = math.sqrt(sum(a**2 for a in acc))
                quat_n = math.sqrt(sum(q**2 for q in quat))
                pg     = proj_gravity(*quat)

                acc_norms.append(acc_n)
                quat_norms.append(quat_n)

                # 資料率
                dt = now - t_last_stat
                recent_frames = sum(1 for _ in range(1))   # 用 interval 計算
                if intervals:
                    rate = 1.0 / (sum(intervals[-20:]) / len(intervals[-20:]))
                else:
                    rate = 0.0

                rate_history.append(rate)

                acc_h  = health_str(acc_n,  ACC_NORM_MIN,  ACC_NORM_MAX)
                quat_h = health_str(quat_n, QUAT_NORM_MIN, QUAT_NORM_MAX)
                rate_h = health_str(rate,   MIN_RATE_HZ,   999)

                print(
                    f"{elapsed:6.1f}s  "
                    f"acc=[{acc[0]:+6.2f} {acc[1]:+6.2f} {acc[2]:+6.2f}] |{acc_n:.2f}|{acc_h}  "
                    f"gyro=[{gyro[0]:+5.3f} {gyro[1]:+5.3f} {gyro[2]:+5.3f}]  "
                    f"|q|={quat_n:.4f}{quat_h}  "
                    f"pg=[{pg[0]:+5.2f} {pg[1]:+5.2f} {pg[2]:+5.2f}]  "
                    f"{rate:5.0f}Hz{rate_h}"
                )
                t_last_stat = now

    except KeyboardInterrupt:
        print("\n")

    finally:
        ser.close()

    # ── 摘要報告 ───────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print(_b("=== IMU 測試摘要 ==="))
    print(f"  總時間  : {total_time:.1f} s")
    print(f"  總幀數  : {frame_count}")

    if frame_count == 0:
        print(_r("  [FAIL] 未收到任何 IMU 資料"))
        print("  → 確認串口正確、H30 Mini 已供電、波特率 460800")
        return False

    avg_rate = frame_count / total_time if total_time > 0 else 0
    print(f"  平均頻率: {avg_rate:.1f} Hz  "
          f"{health_str(avg_rate, MIN_RATE_HZ, 9999)}")

    if acc_norms:
        avg_acc = sum(acc_norms) / len(acc_norms)
        print(f"  |acc| 均值: {avg_acc:.3f} m/s²  "
              f"（靜止應約 9.81）  {health_str(avg_acc, ACC_NORM_MIN, ACC_NORM_MAX)}")

    if quat_norms:
        avg_qn = sum(quat_norms) / len(quat_norms)
        print(f"  |quat| 均值: {avg_qn:.5f}  "
              f"（應約 1.0）  {health_str(avg_qn, QUAT_NORM_MIN, QUAT_NORM_MAX)}")

    if last_data:
        acc  = last_data.get('acc',  (0, 0, -9.81))
        gyro = last_data.get('gyro', (0, 0, 0))
        quat = last_data.get('quat', (1, 0, 0, 0))
        pg   = proj_gravity(*quat)
        print(f"\n  最後一幀：")
        print(f"    acc  = [{acc[0]:+.3f}, {acc[1]:+.3f}, {acc[2]:+.3f}] m/s²")
        print(f"    gyro = [{gyro[0]:+.4f}, {gyro[1]:+.4f}, {gyro[2]:+.4f}] rad/s")
        print(f"    quat = [w={quat[0]:+.4f} x={quat[1]:+.4f} y={quat[2]:+.4f} z={quat[3]:+.4f}]")
        print(f"    proj_gravity = [{pg[0]:+.3f}, {pg[1]:+.3f}, {pg[2]:+.3f}]")

    # 最終判斷
    passed = (
        frame_count > 0
        and avg_rate >= MIN_RATE_HZ
        and (not acc_norms or ACC_NORM_MIN <= sum(acc_norms)/len(acc_norms) <= ACC_NORM_MAX)
        and (not quat_norms or QUAT_NORM_MIN <= sum(quat_norms)/len(quat_norms) <= QUAT_NORM_MAX)
    )
    print()
    print(f"  結果: {_g('[PASS ✓]') if passed else _r('[FAIL ✗]')}")
    print("=" * 60)
    print("""
下一步：
  若 PASS → 執行前置設定後空跑模型
    sudo bash linux/imu/start_h30.sh
    ./run_test.sh --mode policy --imu --dry-run
    """)
    return passed


def main():
    parser = argparse.ArgumentParser(
        description="H30 Mini IMU 純測試（不需要任何其他硬體）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python3 linux/test/test_imu.py
  python3 linux/test/test_imu.py --port /dev/ttyACM1
  python3 linux/test/test_imu.py --duration 30
        """
    )
    parser.add_argument("--port", default=None,
                        help="H30 串口（不指定則自動偵測）")
    parser.add_argument("--baud", type=int, default=IMU_BAUD,
                        help=f"波特率（預設: {IMU_BAUD}）")
    parser.add_argument("--duration", type=float, default=None,
                        help="測試秒數（不指定則 Ctrl+C 結束）")
    args = parser.parse_args()

    port = args.port
    if port is None:
        port = auto_detect_port()
        if port is None:
            print(_r("[ERROR] 找不到 H30 Mini 串口"))
            print(f"  已嘗試: {CANDIDATE_PORTS}")
            print("  → 確認 H30 Mini USB 已插入，或用 --port 指定")
            sys.exit(1)
        print(f"[自動偵測] 使用串口: {port}")

    ok = run(port, args.duration)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
