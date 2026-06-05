#!/usr/bin/env python3
"""
WHEELTEC H30 Mini (YESENSE 協定) → Hiwonder 協定橋接器

WHEELTEC H30 Mini 透過 USB (CH9102 虛擬串口, 460800 baud) 輸出 YESENSE 二進位資料
本腳本解析後轉成 Hiwonder 協定，寫入 /dev/tnt0（透過 tty0tty 核心模組）
讓 deploye_robot firmware 透過 /dev/ttyUSB0 讀取，完全不需修改 Rust 程式碼

輸出的 Hiwonder 封包：
  0x51  加速度（m/s²，含重力）
  0x52  角速度（rad/s）
  0x59  四元數（[qw, qx, qy, qz]，正規化）

前置準備（Pi 上）：
  sudo insmod tty0tty/module/tty0tty.ko
  sudo chmod 666 /dev/tnt0 /dev/tnt1
  sudo ln -sf /dev/tnt1 /dev/ttyUSB0

使用方式：
  python3 bridge_h30.py
  python3 bridge_h30.py --port /dev/ttyACM0 --baud 460800
"""

import argparse
import math
import struct
import time

import serial

# ── YESENSE 協定常數 ───────────────────────────────────────────────────────────
YIS_H1 = 0x59   # 'Y'
YIS_H2 = 0x53   # 'S'
PROTOCOL_MIN_LEN   = 7
PROTOCOL_LEN_POS   = 4
PAYLOAD_POS        = 5
CRC_CALC_START_POS = 2
TLV_HEADER_LEN     = 2

ID_ACC  = 0x10
ID_GYRO = 0x20
ID_QUAT = 0x41

FACTOR = 1e-6   # 所有感測器整數值的換算因子

# ── Hiwonder 編碼用的物理量程 ─────────────────────────────────────────────────
ACC_FULLSCALE  = 16.0 * 9.80665              # m/s²（±16g）
GYRO_FULLSCALE = 2000.0 * math.pi / 180.0   # rad/s（±2000 dps）

# ── 虛擬串口（deploye_robot firmware 讀這端）─────────────────────────────────
DEFAULT_VIRT_PORT = "/dev/tnt0"
DEFAULT_VIRT_BAUD = 230400

# ── 跨執行緒共享狀態（test_policy.py --imu 讀取用）──────────────────────────
import threading
import numpy as np

_imu_lock = threading.Lock()
IMU_STATE: dict = {
    "acc":     np.array([0.0, 0.0, -9.81], dtype=np.float64),
    "gyro":    np.zeros(3, dtype=np.float64),
    "quat":    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),  # [qw,qx,qy,qz]
    "updated": False,
}


def _ned_to_enu(qw, qx, qy, qz):
    """H30 NED 四元數 → ENU 四元數（firmware/policy 所需慣例）。
    NED 平躺 q≈[0,1,0,0] → ENU identity [1,0,0,0]
    """
    return float(qx), float(-qw), float(qz), float(-qy)


def proj_gravity_from_quat(qw: float, qx: float, qy: float, qz: float):
    """從 H30 NED 四元數計算投影重力向量（body frame）。
    先轉 ENU 再做 R.T @ [0,0,-1]。
    H30 Z 軸朝下安裝，NED→ENU 後 pg Z 符號相反，翻轉對齊訓練慣例（直立=-1）。
    """
    import numpy as np
    ew, ex, ey, ez = _ned_to_enu(qw, qx, qy, qz)
    r02 = 2 * (ex * ez - ew * ey)
    r12 = 2 * (ey * ez + ew * ex)
    r22 = ew * ew - ex * ex - ey * ey + ez * ez
    return np.array([r02, r12, r22], dtype=np.float32)  # 翻轉：[-r02,-r12,-r22] → [r02,r12,r22]


# ── 工具函數 ──────────────────────────────────────────────────────────────────

def _get_int32(b) -> int:
    v = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)
    if v & 0x8000_0000:
        v -= 0x1_0000_0000
    return v


def _yis_checksum(data, length: int) -> int:
    a = b = 0
    for i in range(length):
        a = (a + data[i]) & 0xFF
        b = (b + a) & 0xFF
    return (b << 8) | a


def _clamp_i16(v: float) -> int:
    return max(-32768, min(32767, int(round(v))))


def _hiwonder_packet(ptype: int, v0: float, v1: float, v2: float, v3: float = 0.0) -> bytes:
    """封裝 Hiwonder 11-byte 封包：[0x55][ptype][4×int16 LE][checksum]"""
    body = bytes([0x55, ptype]) + struct.pack(
        '<hhhh', _clamp_i16(v0), _clamp_i16(v1), _clamp_i16(v2), _clamp_i16(v3))
    return body + bytes([sum(body) & 0xFF])


# ── YESENSE 幀解析 ─────────────────────────────────────────────────────────────

def _parse_yis_frame(buf: bytearray, pos: int):
    """解析 buf[pos] 開始的一個 YESENSE 幀。回傳 (dict, next_pos)；失敗回 (None, pos+1)。"""
    payload_len = buf[pos + PROTOCOL_LEN_POS]
    frame_total = PROTOCOL_MIN_LEN + payload_len

    crc_pos = pos + CRC_CALC_START_POS + payload_len + 3
    crc_rx  = buf[crc_pos] | (buf[crc_pos + 1] << 8)
    crc_len = payload_len + 3   # tid(2) + len(1) + payload
    crc_calc = _yis_checksum(buf[pos + CRC_CALC_START_POS:], crc_len)
    if crc_rx != crc_calc:
        return None, pos + 1

    result = {}
    p   = pos + PAYLOAD_POS
    end = pos + PAYLOAD_POS + payload_len

    while p + TLV_HEADER_LEN <= end:
        tid  = buf[p]
        tlen = buf[p + 1]
        data = buf[p + TLV_HEADER_LEN:]
        p   += TLV_HEADER_LEN + tlen

        if tid == ID_ACC and tlen == 12:
            result['acc'] = (
                _get_int32(data)      * FACTOR,
                _get_int32(data[4:])  * FACTOR,
                _get_int32(data[8:])  * FACTOR,
            )
        elif tid == ID_GYRO and tlen == 12:
            result['gyro'] = (
                _get_int32(data)      * FACTOR,
                _get_int32(data[4:])  * FACTOR,
                _get_int32(data[8:])  * FACTOR,
            )
        elif tid == ID_QUAT and tlen == 16:
            result['quat'] = (
                _get_int32(data)       * FACTOR,   # qw
                _get_int32(data[4:])   * FACTOR,   # qx
                _get_int32(data[8:])   * FACTOR,   # qy
                _get_int32(data[12:])  * FACTOR,   # qz
            )

    return result, pos + frame_total


# ── 主迴圈 ────────────────────────────────────────────────────────────────────

def run(imu_port: str, imu_baud: int, virt_port: str, virt_baud: int) -> None:
    print(f"=== WHEELTEC H30 Mini → Hiwonder 橋接器 ===")
    print(f"IMU  : {imu_port} @ {imu_baud} baud")
    print(f"輸出 : {virt_port} @ {virt_baud} baud\n")

    imu_ser = serial.Serial(imu_port, imu_baud, timeout=0.01)
    virt_ser = serial.Serial(virt_port, virt_baud, timeout=0,
                             xonxoff=False, rtscts=False, dsrdtr=False)

    print("橋接中（Ctrl+C 停止）...\n")

    buf         = bytearray()
    frame_count = 0
    last_stat   = time.time()
    last_data   = {}

    try:
        while True:
            chunk = imu_ser.read_all() or imu_ser.read(512)
            if not chunk:
                time.sleep(0.002)
                continue

            buf.extend(chunk)
            pos = 0

            while pos < len(buf) - PROTOCOL_MIN_LEN:
                # 找幀頭
                if buf[pos] != YIS_H1 or buf[pos + 1] != YIS_H2:
                    pos += 1
                    continue

                payload_len = buf[pos + PROTOCOL_LEN_POS]
                if pos + PROTOCOL_MIN_LEN + payload_len > len(buf):
                    break   # 等更多資料

                result, next_pos = _parse_yis_frame(buf, pos)
                if result is None:
                    pos += 1
                    continue

                last_data.update(result)
                frame_count += 1

                # 更新共享狀態（供 test_policy.py --imu 讀取）
                with _imu_lock:
                    if 'acc' in result:
                        IMU_STATE['acc'] = np.array(result['acc'], dtype=np.float64)
                    if 'gyro' in result:
                        IMU_STATE['gyro'] = np.array(result['gyro'], dtype=np.float64)
                    if 'quat' in result:
                        ew, ex, ey, ez = _ned_to_enu(*result['quat'])
                        IMU_STATE['quat'] = np.array([ew, ex, ey, ez], dtype=np.float64)
                    IMU_STATE['updated'] = True

                # acc: m/s² → Hiwonder int16（量程 ±16g * 9.80665 m/s²）
                if 'acc' in result:
                    ax, ay, az = result['acc']
                    virt_ser.write(_hiwonder_packet(0x51,
                        ax / ACC_FULLSCALE * 32768,
                        ay / ACC_FULLSCALE * 32768,
                        az / ACC_FULLSCALE * 32768))

                # gyro: rad/s → Hiwonder int16（量程 ±2000 dps）
                if 'gyro' in result:
                    gx, gy, gz = result['gyro']
                    virt_ser.write(_hiwonder_packet(0x52,
                        gx / GYRO_FULLSCALE * 32768,
                        gy / GYRO_FULLSCALE * 32768,
                        gz / GYRO_FULLSCALE * 32768))

                # quat: H30 NED → ENU → Hiwonder 0x59（int16 × 32768）
                if 'quat' in result:
                    ew, ex, ey, ez = _ned_to_enu(*result['quat'])
                    virt_ser.write(_hiwonder_packet(0x59,
                        ew * 32768, ex * 32768, ey * 32768, ez * 32768))

                virt_ser.flush()
                pos = next_pos

            buf = buf[pos:]

            # 每 2 秒印一次狀態
            now = time.time()
            if now - last_stat >= 2.0 and last_data:
                acc  = last_data.get('acc',  (0, 0, 0))
                gyro = last_data.get('gyro', (0, 0, 0))
                quat = last_data.get('quat', (1, 0, 0, 0))
                g    = math.sqrt(sum(a**2 for a in acc))
                print(f"\r[{frame_count:7d} 幀]  "
                      f"acc=[{acc[0]:+.2f},{acc[1]:+.2f},{acc[2]:+.2f}]m/s²(|{g:.2f}|)  "
                      f"gyro=[{gyro[0]:+.3f},{gyro[1]:+.3f},{gyro[2]:+.3f}]rad/s  "
                      f"quat=[w={quat[0]:+.3f} x={quat[1]:+.3f} y={quat[2]:+.3f} z={quat[3]:+.3f}]",
                      end="", flush=True)
                last_stat = now

    except KeyboardInterrupt:
        print(f"\n已停止，共轉換 {frame_count} 幀")
    finally:
        imu_ser.close()
        virt_ser.close()


def run_imu_only(imu_port: str, imu_baud: int) -> None:
    """只讀取 H30 IMU 並更新 IMU_STATE，不需要 /dev/tnt0 虛擬串口。
    供 test_policy.py --imu 在背景執行緒中使用。
    """
    imu_ser = serial.Serial(imu_port, imu_baud, timeout=0.01)

    buf       = bytearray()
    last_data: dict = {}

    try:
        while True:
            chunk = imu_ser.read_all() or imu_ser.read(512)
            if not chunk:
                time.sleep(0.002)
                continue

            buf.extend(chunk)
            pos = 0

            while pos < len(buf) - PROTOCOL_MIN_LEN:
                if buf[pos] != YIS_H1 or buf[pos + 1] != YIS_H2:
                    pos += 1
                    continue
                payload_len = buf[pos + PROTOCOL_LEN_POS]
                if pos + PROTOCOL_MIN_LEN + payload_len > len(buf):
                    break
                result, next_pos = _parse_yis_frame(buf, pos)
                if result is None:
                    pos += 1
                    continue

                last_data.update(result)

                with _imu_lock:
                    if 'acc' in result:
                        IMU_STATE['acc'] = np.array(result['acc'], dtype=np.float64)
                    if 'gyro' in result:
                        IMU_STATE['gyro'] = np.array(result['gyro'], dtype=np.float64)
                    if 'quat' in result:
                        ew, ex, ey, ez = _ned_to_enu(*result['quat'])
                        IMU_STATE['quat'] = np.array([ew, ex, ey, ez], dtype=np.float64)
                    IMU_STATE['updated'] = True

                pos = next_pos

            buf = buf[pos:]

    except KeyboardInterrupt:
        pass
    finally:
        imu_ser.close()


def main():
    parser = argparse.ArgumentParser(
        description="WHEELTEC H30 Mini (YESENSE) → Hiwonder 橋接器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
前置準備（樹梅派）：
  cd linux/tty0tty/module && make
  sudo insmod tty0tty.ko
  sudo chmod 666 /dev/tnt0 /dev/tnt1
  sudo ln -sf /dev/tnt1 /dev/ttyUSB0

確認 H30 Mini 串口：
  ls /dev/ttyACM* /dev/ttyUSB*

使用：
  python3 bridge_h30.py
  python3 bridge_h30.py --port /dev/ttyACM0
        """
    )
    parser.add_argument("--port", default="/dev/ttyACM0",
                        help="H30 Mini 串口（預設: /dev/ttyACM0）")
    parser.add_argument("--baud", type=int, default=460800,
                        help="H30 Mini 波特率（預設: 460800）")
    parser.add_argument("--virt-port", default=DEFAULT_VIRT_PORT,
                        help=f"tty0tty 輸出端（預設: {DEFAULT_VIRT_PORT}）")
    parser.add_argument("--virt-baud", type=int, default=DEFAULT_VIRT_BAUD,
                        help=f"虛擬串口波特率（預設: {DEFAULT_VIRT_BAUD}）")
    args = parser.parse_args()

    run(args.port, args.baud, args.virt_port, args.virt_baud)


if __name__ == "__main__":
    main()
