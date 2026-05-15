"""
WHEELTEC H30 Mini IMU reader — YESENSE (YIS) protocol
Baudrate: 460800, Header: 0x59 0x53 ("YS")

Frame format:
  [59][53][TID_L][TID_H][LEN][TLV payload...][CK1][CK2]

Output mapped to humanoid robot observation vector fields:
  imu_acc  [3]  m/s² body frame  (×1e-6 scale, includes gravity ~9.81 on gravity axis)
  imu_gyro [3]  rad/s body frame (×1e-6 scale)
  quat     [4]  [q0=qw, q1=qx, q2=qy, q3=qz] Hamilton convention
  proj_grav[3]  world [0,0,-9.81] rotated into body frame
"""

import serial
import sys
import time
import math

# ─── Protocol constants (from YESENSE SDK) ────────────────────────────────────
PORT     = "COM4"
BAUDRATE = 460800

YIS_H1 = 0x59   # 'Y'
YIS_H2 = 0x53   # 'S'

PROTOCOL_MIN_LEN     = 7
PROTOCOL_TID_POS     = 2
PROTOCOL_LEN_POS     = 4
PAYLOAD_POS          = 5
CRC_CALC_START_POS   = 2
TLV_HEADER_LEN       = 2

# TLV data IDs
ID_TEMP      = 0x01
ID_ACC       = 0x10
ID_GYRO      = 0x20
ID_EULER     = 0x40
ID_QUAT      = 0x41
ID_STATUS    = 0x80

FACTOR = 1e-6   # applied to all sensor ints to get physical units

# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_int32(b):
    v = b[0] | (b[1]<<8) | (b[2]<<16) | (b[3]<<24)
    if v & 0x8000_0000:
        v -= 0x1_0000_0000
    return v

def calc_checksum(data, length):
    a = b = 0
    for i in range(length):
        a = (a + data[i]) & 0xFF
        b = (b + a) & 0xFF
    return (b << 8) | a

def CRC_CALC_LEN(payload_len):
    return payload_len + 3   # tid(2) + len(1)

def CRC_POS(payload_len):
    return CRC_CALC_START_POS + CRC_CALC_LEN(payload_len)

# ─── projected_gravity ────────────────────────────────────────────────────────
def proj_gravity(qw, qx, qy, qz):
    """R.T @ [0,0,-9.81] → gravity in body frame"""
    R02 = 2*(qx*qz + qy*qw)
    R12 = 2*(qy*qz - qx*qw)
    R22 = 1 - 2*(qx*qx + qy*qy)
    return -9.81*R02, -9.81*R12, -9.81*R22

# ─── Display ──────────────────────────────────────────────────────────────────
def display(d):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] TID={d.get('tid',0)} ----------------------------")

    if 'acc' in d:
        ax, ay, az = d['acc']
        g  = math.sqrt(ax**2 + ay**2 + az**2)
        print(f"  imu_acc  (m/s2): x={ax:+8.4f}  y={ay:+8.4f}  z={az:+8.4f}  |g|={g:.3f}")

    if 'gyro' in d:
        gx, gy, gz = d['gyro']
        print(f"  imu_gyro(rad/s): x={gx:+8.4f}  y={gy:+8.4f}  z={gz:+8.4f}")

    if 'quat' in d:
        qw, qx, qy, qz = d['quat']
        qnorm = math.sqrt(qw**2+qx**2+qy**2+qz**2)
        pgx, pgy, pgz = proj_gravity(qw, qx, qy, qz)
        print(f"  quat (w,x,y,z) : {qw:+.4f}  {qx:+.4f}  {qy:+.4f}  {qz:+.4f}  norm={qnorm:.4f}")
        print(f"  proj_gravity   : x={pgx:+7.3f}  y={pgy:+7.3f}  z={pgz:+7.3f}")

    if 'euler' in d:
        pitch, roll, yaw = d['euler']
        print(f"  euler  (deg)   : pitch={pitch:+7.2f}  roll={roll:+7.2f}  yaw={yaw:+7.2f}")

    if 'temp' in d:
        print(f"  temp           : {d['temp']:.1f}C")
    print()

# ─── Frame parser ─────────────────────────────────────────────────────────────
def parse_frame(buf, pos):
    """Parse one YIS frame starting at buf[pos]. Returns (result_dict, frame_end_pos)."""
    payload_len = buf[pos + PROTOCOL_LEN_POS]
    frame_total = PROTOCOL_MIN_LEN + payload_len

    # Verify CRC
    crc_pos = pos + CRC_POS(payload_len)
    crc_rx  = buf[crc_pos] | (buf[crc_pos+1] << 8)
    crc_calc = calc_checksum(buf[pos+CRC_CALC_START_POS:], CRC_CALC_LEN(payload_len))
    if crc_rx != crc_calc:
        return None, pos + 1

    result = {'tid': buf[pos+PROTOCOL_TID_POS] | (buf[pos+PROTOCOL_TID_POS+1] << 8)}

    p    = pos + PAYLOAD_POS
    end  = pos + PAYLOAD_POS + payload_len
    while p + TLV_HEADER_LEN <= end:
        tid = buf[p]
        tlen = buf[p+1]
        data = buf[p+TLV_HEADER_LEN:]
        p   += TLV_HEADER_LEN + tlen

        if tid == ID_ACC and tlen == 12:
            result['acc'] = (get_int32(data)*FACTOR,
                             get_int32(data[4:])*FACTOR,
                             get_int32(data[8:])*FACTOR)
        elif tid == ID_GYRO and tlen == 12:
            result['gyro'] = (get_int32(data)*FACTOR,
                              get_int32(data[4:])*FACTOR,
                              get_int32(data[8:])*FACTOR)
        elif tid == ID_QUAT and tlen == 16:
            q0 = get_int32(data)*FACTOR
            q1 = get_int32(data[4:])*FACTOR
            q2 = get_int32(data[8:])*FACTOR
            q3 = get_int32(data[12:])*FACTOR
            result['quat'] = (q0, q1, q2, q3)  # (qw, qx, qy, qz)
        elif tid == ID_EULER and tlen == 12:
            result['euler'] = (get_int32(data)*FACTOR,   # pitch
                               get_int32(data[4:])*FACTOR, # roll
                               get_int32(data[8:])*FACTOR) # yaw (deg)
        elif tid == ID_TEMP and tlen == 2:
            t = data[0] | (data[1] << 8)
            if t & 0x8000: t -= 0x10000
            result['temp'] = t * 0.01

    return result, pos + frame_total

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Opening {PORT} at {BAUDRATE} baud (YESENSE protocol)...")
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.01)
    except serial.SerialException as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("Reading WHEELTEC H30 Mini... (Ctrl+C to stop)\n")
    buf = bytearray()
    raw_shown = False
    frame_count = 0

    try:
        while True:
            chunk = ser.read_all() or ser.read(256)
            if not chunk:
                time.sleep(0.005)
                continue

            if not raw_shown:
                print(f"[RAW first 20 bytes] {bytes(chunk[:20]).hex()}")
                ok = chunk[0]==0x59 and chunk[1]==0x53
                print(f"  -> Header: {chunk[0]:02x} {chunk[1]:02x} {'[OK YS]' if ok else '[scanning...]'}")
                print()
                raw_shown = True

            buf.extend(chunk)

            pos = 0
            while pos < len(buf) - PROTOCOL_MIN_LEN:
                # Find header
                if buf[pos] != YIS_H1 or buf[pos+1] != YIS_H2:
                    pos += 1
                    continue

                payload_len = buf[pos + PROTOCOL_LEN_POS]
                if pos + PROTOCOL_MIN_LEN + payload_len > len(buf):
                    break  # wait for more data

                result, next_pos = parse_frame(buf, pos)
                if result is not None:
                    frame_count += 1
                    display(result)
                    pos = next_pos
                else:
                    pos += 1

            # Keep only unprocessed bytes
            buf = buf[pos:]

    except KeyboardInterrupt:
        print(f"\nStopped. Total frames parsed: {frame_count}")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
