#!/usr/bin/env python3
"""
BWT901BLE5.0 → Hiwonder 協定橋接器
BLE 資料 → 虛擬串口 /dev/ttyUSB0（透過內建 socat）

與原始 firmware 架構完全相容：
  firmware (faux-rtos) 讀 /dev/ttyUSB0
  本腳本寫 /dev/ttyVirtual0
  socat 自動橋接兩端

使用方式：
  sudo python3 bridge.py                     # 自動搜尋 WT901BLE67
  sudo python3 bridge.py --name MyDevice     # 指定裝置名稱
  sudo python3 bridge.py --scan              # 掃描附近 BLE 裝置後退出

依賴：
  sudo apt install socat
  pip install bleak pyserial --break-system-packages
"""

import argparse
import asyncio
import os
import struct
import subprocess
import sys
import time

import serial
import bleak

# ── 設定 ──────────────────────────────────────────────────────────────────────
DEFAULT_NAME  = "WT901BLE67"
NOTIFY_UUID   = "0000ffe4-0000-1000-8000-00805f9a34fb"
WRITE_UUID    = "0000ffe9-0000-1000-8000-00805f9a34fb"

FIRMWARE_PORT = "/dev/ttyUSB0"      # firmware 讀這端
BRIDGE_PORT   = "/dev/ttyVirtual0"  # 本腳本寫這端
BAUD_RATE     = 230400

OUTPUT_RATE_HZ = 50
_RATE_MAP = {10: 6, 20: 7, 50: 8, 100: 9, 200: 11}


# ── Hiwonder 封包工具 ─────────────────────────────────────────────────────────
def sign16(n: int) -> int:
    return n - 65536 if n >= 32768 else n

def clamp_i16(v: float) -> int:
    return max(-32768, min(32767, int(round(v))))

def make_packet(ptype: int, v0: float, v1: float, v2: float, v3: float = 0.0) -> bytes:
    """組成 11-byte Hiwonder 封包，checksum = sum(前10bytes) & 0xFF"""
    body = bytes([0x55, ptype]) + struct.pack('<hhhh',
        clamp_i16(v0), clamp_i16(v1), clamp_i16(v2), clamp_i16(v3))
    return body + bytes([sum(body) & 0xFF])


# ── 共享串口 ──────────────────────────────────────────────────────────────────
_ser: serial.Serial = None

def serial_write(data: bytes):
    global _ser
    if _ser and _ser.is_open:
        try:
            _ser.write(data)
            _ser.flush()          # 確保立即送出，不留在緩衝區
        except Exception as e:
            print(f"[WARN] 串口寫入失敗: {e}")


# ── BLE 封包解析 ──────────────────────────────────────────────────────────────
_ble_buf: list = []

def on_notify(sender, data: bytearray):
    for b in data:
        _ble_buf.append(b)
        if len(_ble_buf) == 1 and _ble_buf[0] != 0x55:
            _ble_buf.clear()
            continue
        if len(_ble_buf) == 2 and _ble_buf[1] not in (0x61, 0x71):
            _ble_buf.clear()
            continue
        if len(_ble_buf) == 20:
            _convert_and_forward(_ble_buf[:])
            _ble_buf.clear()

def _convert_and_forward(b: list):
    ptype = b[1]

    if ptype == 0x61:
        ax = sign16(b[3]  << 8 | b[2])  / 32768 * 16     # g
        ay = sign16(b[5]  << 8 | b[4])  / 32768 * 16
        az = sign16(b[7]  << 8 | b[6])  / 32768 * 16
        gx = sign16(b[9]  << 8 | b[8])  / 32768 * 2000   # dps
        gy = sign16(b[11] << 8 | b[10]) / 32768 * 2000
        gz = sign16(b[13] << 8 | b[12]) / 32768 * 2000
        ang_x = sign16(b[15] << 8 | b[14]) / 32768 * 180 # deg
        ang_y = sign16(b[17] << 8 | b[16]) / 32768 * 180
        ang_z = sign16(b[19] << 8 | b[18]) / 32768 * 180

        pkt_acc   = make_packet(0x51, ax * 2048, ay * 2048, az * 2048)
        pkt_gyro  = make_packet(0x52, gx * 32768/2000, gy * 32768/2000, gz * 32768/2000)
        pkt_angle = make_packet(0x53, ang_x * 32768/180, ang_y * 32768/180, ang_z * 32768/180)

        serial_write(pkt_acc + pkt_gyro + pkt_angle)
        print(f"\r[ACC] {ax:5.2f} {ay:5.2f} {az:5.2f}g  "
              f"[GYR] {gx:7.1f} {gy:7.1f} {gz:7.1f}dps  "
              f"[ANG] {ang_x:6.1f} {ang_y:6.1f} {ang_z:6.1f}°",
              end="", flush=True)

    elif ptype == 0x71 and b[2] == 0x51:
        q0 = sign16(b[5]  << 8 | b[4])  / 32768
        q1 = sign16(b[7]  << 8 | b[6])  / 32768
        q2 = sign16(b[9]  << 8 | b[8])  / 32768
        q3 = sign16(b[11] << 8 | b[10]) / 32768
        pkt_quat = make_packet(0x59, q0*32768, q1*32768, q2*32768, q3*32768)
        serial_write(pkt_quat)


# ── socat 管理 ────────────────────────────────────────────────────────────────
def start_socat() -> subprocess.Popen:
    for p in (FIRMWARE_PORT, BRIDGE_PORT):
        if os.path.islink(p) or os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    proc = subprocess.Popen(
        ["socat",
         f"PTY,link={FIRMWARE_PORT},raw,echo=0",
         f"PTY,link={BRIDGE_PORT},raw,echo=0"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(20):
        if os.path.exists(FIRMWARE_PORT) and os.path.exists(BRIDGE_PORT):
            break
        time.sleep(0.1)
    else:
        proc.terminate()
        print("[ERROR] socat 無法建立虛擬串口，請確認已安裝: sudo apt install socat")
        sys.exit(1)

    print(f"虛擬串口已建立：{BRIDGE_PORT} ↔ {FIRMWARE_PORT}")
    return proc


# ── 主流程 ────────────────────────────────────────────────────────────────────
def make_read_cmd(reg: int) -> bytes:
    return bytes([0xff, 0xaa, 0x27, reg, 0x00])

async def scan_devices():
    print("掃描 BLE 裝置（5 秒）...")
    devices = await bleak.BleakScanner.discover(timeout=5)
    if not devices:
        print("  找不到任何裝置")
    for d in devices:
        print(f"  {d.address}  {d.name}")

async def run_bridge(target_name: str):
    global _ser

    socat_proc = start_socat()

    # raw 模式開啟串口，關閉所有流量控制
    _ser = serial.Serial(
        BRIDGE_PORT,
        baudrate=BAUD_RATE,
        timeout=0,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )
    print(f"串口已開啟: {BRIDGE_PORT} @ {BAUD_RATE} baud")

    try:
        # 用名稱搜尋裝置（避免 BLE 隨機 MAC 問題）
        print(f"掃描中，尋找 '{target_name}' ...")
        found_event = asyncio.Event()
        found_device = None

        def on_discovered(device, _):
            nonlocal found_device
            if device.name == target_name and not found_event.is_set():
                found_device = device
                found_event.set()

        async with bleak.BleakScanner(detection_callback=on_discovered):
            await asyncio.wait_for(found_event.wait(), timeout=15.0)

        if found_device is None:
            print(f"[ERROR] 找不到裝置 '{target_name}'，用 --scan 確認名稱")
            return

        print(f"找到：{found_device.name}  ({found_device.address})")
        print(f"連線中...")

        async with bleak.BleakClient(found_device, timeout=30.0) as client:
            print(f"已連線！MTU={client.mtu_size}\n")

            notify_char = write_char = None
            for svc in client.services:
                for ch in svc.characteristics:
                    if ch.uuid == NOTIFY_UUID:
                        notify_char = ch
                    if ch.uuid == WRITE_UUID:
                        write_char = ch

            if notify_char is None:
                print("[ERROR] 找不到 notify characteristic")
                return

            # 設定輸出頻率
            rate_val = _RATE_MAP.get(OUTPUT_RATE_HZ, 8)
            if write_char:
                await client.write_gatt_char(
                    write_char.uuid,
                    bytes([0xff, 0xaa, 0x03, rate_val, 0x00])
                )
                await asyncio.sleep(0.1)
                print(f"輸出率設定：{OUTPUT_RATE_HZ} Hz")

            await client.start_notify(notify_char.uuid, on_notify)
            print("橋接中（按 Ctrl+C 停止）...\n")

            async def quat_request_loop():
                """50Hz 請求四元數"""
                while True:
                    if write_char:
                        try:
                            await client.write_gatt_char(
                                write_char.uuid, make_read_cmd(0x51))
                        except Exception:
                            pass
                    await asyncio.sleep(1.0 / OUTPUT_RATE_HZ)

            async def drain_serial_loop():
                """讀取並丟棄 firmware 送來的設定指令，防止 PTY 緩衝區滿"""
                while True:
                    if _ser and _ser.is_open:
                        try:
                            waiting = _ser.in_waiting
                            if waiting > 0:
                                _ser.read(waiting)
                        except Exception:
                            pass
                    await asyncio.sleep(0.05)

            try:
                await asyncio.gather(
                    quat_request_loop(),
                    drain_serial_loop(),
                    asyncio.sleep(86400),
                )
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass

            await client.stop_notify(notify_char.uuid)

    finally:
        if _ser and _ser.is_open:
            _ser.close()
        socat_proc.terminate()
        print("\n橋接結束，虛擬串口已移除")


def main():
    parser = argparse.ArgumentParser(description="BWT901BLE5.0 → Hiwonder 橋接器")
    parser.add_argument("--name", default=DEFAULT_NAME,
                        help=f"BLE 裝置名稱（預設: {DEFAULT_NAME}）")
    parser.add_argument("--scan", action="store_true",
                        help="掃描附近 BLE 裝置後退出")
    args = parser.parse_args()

    if args.scan:
        asyncio.run(scan_devices())
        return

    if os.geteuid() != 0:
        print("[ERROR] 需要 sudo 權限（建立虛擬串口需要 root）")
        sys.exit(1)

    asyncio.run(run_bridge(args.name))


if __name__ == "__main__":
    main()
