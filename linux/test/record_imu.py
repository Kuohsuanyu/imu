#!/usr/bin/env python3
"""
H30 Mini IMU 資料錄製腳本
==========================
在樹梅派上錄製真實 IMU 資料，存成 CSV。
錄製的 CSV 可以拿回電腦與模擬的 IMU 資料對比，確認推論輸入一致。

使用方式：
  python3 linux/test/record_imu.py                      # Ctrl+C 停止
  python3 linux/test/record_imu.py --duration 30        # 錄 30 秒
  python3 linux/test/record_imu.py --port /dev/ttyACM7  # 指定串口

輸出 CSV 欄位：
  time_s, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z,
  quat_w, quat_x, quat_y, quat_z, pg_x, pg_y, pg_z

依賴：pyserial, numpy
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "imu"))

import bridge_h30 as _bridge

try:
    import serial
except ImportError:
    print("[ERROR] 缺少 pyserial → sudo apt install python3-serial")
    sys.exit(1)

import numpy as np

CANDIDATE_PORTS = [
    "/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2", "/dev/ttyACM3",
    "/dev/ttyACM4", "/dev/ttyACM5", "/dev/ttyACM6", "/dev/ttyACM7",
    "/dev/ttyUSB1", "/dev/ttyUSB2",
]
IMU_BAUD    = 460800
OUTPUT_DIR  = Path(__file__).resolve().parent.parent.parent / "recordings"


def auto_detect_port() -> str | None:
    for p in CANDIDATE_PORTS:
        if Path(p).exists():
            return p
    return None


def proj_gravity(qw, qx, qy, qz):
    r02 = 2 * (qx * qz - qw * qy)
    r12 = 2 * (qy * qz + qw * qx)
    r22 = qw**2 - qx**2 - qy**2 + qz**2
    return r02, r12, r22


def run(port: str, duration: float | None, output_dir: Path):
    print(f"\n=== IMU 錄製 ===")
    print(f"串口: {port}  |  輸出: {output_dir}")
    if duration:
        print(f"錄製時長: {duration} 秒")
    else:
        print("按 Ctrl+C 停止錄製")
    print()

    try:
        ser = serial.Serial(port, IMU_BAUD, timeout=0.01)
    except serial.SerialException as e:
        print(f"[ERROR] 無法開啟 {port}: {e}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    tag      = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{tag}_imu_record.csv"

    fieldnames = [
        "time_s",
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
        "quat_w", "quat_x", "quat_y", "quat_z",
        "pg_x", "pg_y", "pg_z",
    ]

    buf         = bytearray()
    last_data:  dict = {}
    t_start     = time.time()
    frame_count = 0
    rows: list[dict] = []

    print(f"{'時間':>6}  acc(m/s²)                      gyro(rad/s)             pg")
    print("-" * 80)

    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

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

                    if ('acc' in last_data and 'gyro' in last_data and 'quat' in last_data):
                        acc  = last_data['acc']
                        gyro = last_data['gyro']
                        # NED → ENU quaternion conversion
                        qw_r, qx_r, qy_r, qz_r = last_data['quat']
                        qw, qx, qy, qz = _bridge._ned_to_enu(qw_r, qx_r, qy_r, qz_r)
                        pg = proj_gravity(qw, qx, qy, qz)

                        row = {
                            "time_s":  round(elapsed, 5),
                            "acc_x":   round(acc[0],  6),
                            "acc_y":   round(acc[1],  6),
                            "acc_z":   round(acc[2],  6),
                            "gyro_x":  round(gyro[0], 6),
                            "gyro_y":  round(gyro[1], 6),
                            "gyro_z":  round(gyro[2], 6),
                            "quat_w":  round(qw,  6),
                            "quat_x":  round(qx,  6),
                            "quat_y":  round(qy,  6),
                            "quat_z":  round(qz,  6),
                            "pg_x":    round(pg[0],   6),
                            "pg_y":    round(pg[1],   6),
                            "pg_z":    round(pg[2],   6),
                        }
                        writer.writerow(row)
                        f.flush()

                    pos = next_pos

                buf = buf[pos:]

                # 每秒印一行
                if frame_count % 200 == 1 and last_data:
                    acc  = last_data.get('acc',  (0, 0, -9.81))
                    gyro = last_data.get('gyro', (0, 0, 0))
                    qw_r, qx_r, qy_r, qz_r = last_data.get('quat', (0, 1, 0, 0))
                    qw, qx, qy, qz = _bridge._ned_to_enu(qw_r, qx_r, qy_r, qz_r)
                    pg = proj_gravity(qw, qx, qy, qz)
                    print(
                        f"{elapsed:6.1f}s  "
                        f"[{acc[0]:+6.2f} {acc[1]:+6.2f} {acc[2]:+6.2f}]  "
                        f"[{gyro[0]:+5.3f} {gyro[1]:+5.3f} {gyro[2]:+5.3f}]  "
                        f"pg=[{pg[0]:+5.2f} {pg[1]:+5.2f} {pg[2]:+5.2f}]"
                    )

    except KeyboardInterrupt:
        print("\n[停止]")
    finally:
        ser.close()

    total_time = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"  錄製完成: {csv_path}")
    print(f"  時長: {total_time:.1f}s  幀數: {frame_count}  "
          f"平均頻率: {frame_count/total_time:.0f} Hz")
    print(f"{'='*50}")
    print(f"\n將此檔案傳回電腦，放入 recordings/ 目錄即可用 replay 模式比對。")
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="H30 Mini IMU 錄製（輸出 CSV 供電腦端推論比對）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python3 linux/test/record_imu.py
  python3 linux/test/record_imu.py --port /dev/ttyACM7 --duration 30
  python3 linux/test/record_imu.py --out /tmp/imu_test.csv
        """
    )
    parser.add_argument("--port", default=None, help="H30 串口（不指定則自動偵測）")
    parser.add_argument("--baud", type=int, default=IMU_BAUD)
    parser.add_argument("--duration", type=float, default=None, help="錄製秒數")
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR),
                        help=f"輸出目錄（預設: {OUTPUT_DIR}）")
    args = parser.parse_args()

    port = args.port or auto_detect_port()
    if port is None:
        print(f"[ERROR] 找不到 H30 串口，已嘗試: {CANDIDATE_PORTS}")
        print("  → 用 --port 指定")
        sys.exit(1)
    print(f"[串口] {port}")

    run(port, args.duration, Path(args.out_dir))


if __name__ == "__main__":
    main()
