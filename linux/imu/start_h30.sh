#!/bin/bash
# WHEELTEC H30 Mini → deploye_robot firmware 啟動腳本
# 使用方式：bash start_h30.sh [串口] [波特率]
#   bash start_h30.sh                      # 自動偵測
#   bash start_h30.sh /dev/ttyACM0 460800

IMU_PORT="${1:-}"
IMU_BAUD="${2:-460800}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODULE_DIR="$SCRIPT_DIR/tty0tty/module"

echo "=== H30 Mini → Hiwonder Bridge ==="
echo ""

# ── Python 依賴 ───────────────────────────────────────────────────────────────
python3 -c "import serial" 2>/dev/null || {
    echo "安裝 pyserial..."
    pip install pyserial --break-system-packages
}

# ── tty0tty 核心模組 ──────────────────────────────────────────────────────────
if [ ! -c /dev/tnt0 ]; then
    if [ -f "$MODULE_DIR/tty0tty.ko" ]; then
        echo "載入 tty0tty..."
        sudo insmod "$MODULE_DIR/tty0tty.ko"
    else
        echo "[ERROR] 找不到 tty0tty.ko"
        echo "  cd $MODULE_DIR && make"
        exit 1
    fi
fi
sudo chmod 666 /dev/tnt0 /dev/tnt1 2>/dev/null || true

# /dev/tnt1 → /dev/ttyUSB0（firmware 讀這端）
if [ ! -e /dev/ttyUSB0 ]; then
    echo "建立 /dev/ttyUSB0 → /dev/tnt1 symlink..."
    sudo ln -sf /dev/tnt1 /dev/ttyUSB0
fi

echo "虛擬串口: /dev/tnt0 (bridge 寫) ↔ /dev/tnt1 → /dev/ttyUSB0 (firmware 讀)"
echo ""

# ── 自動偵測 H30 Mini 串口 ────────────────────────────────────────────────────
if [ -z "$IMU_PORT" ]; then
    # CH9102 (H30 Mini USB chip) 或其他常見名稱
    for try_port in /dev/ttyACM0 /dev/ttyACM1 /dev/ttyUSB1 /dev/ttyUSB2; do
        if [ -e "$try_port" ] && [ "$try_port" != "/dev/ttyUSB0" ]; then
            IMU_PORT="$try_port"
            echo "偵測到串口: $IMU_PORT"
            break
        fi
    done
    if [ -z "$IMU_PORT" ]; then
        echo "[ERROR] 找不到 H30 Mini，請手動指定串口："
        echo "  bash start_h30.sh /dev/ttyACM0"
        ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo "  （無可用串口）"
        exit 1
    fi
fi

echo "IMU 串口: $IMU_PORT @ $IMU_BAUD baud"
echo ""
echo "啟動橋接器..."
python3 "$SCRIPT_DIR/bridge_h30.py" --port "$IMU_PORT" --baud "$IMU_BAUD"
