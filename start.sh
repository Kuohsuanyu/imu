#!/bin/bash
# BLE IMU 橋接器啟動腳本
# 使用方式：bash start.sh [MAC位址]

MAC="${1:-F1:60:DA:6D:8E:6C}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== BLE IMU Bridge ==="
echo "MAC: $MAC"
echo ""

# 確認 socat 已安裝
if ! command -v socat &>/dev/null; then
    echo "安裝 socat..."
    sudo apt install -y socat
fi

# 確認 Python 依賴
python3 -c "import bleak, serial" 2>/dev/null || {
    echo "安裝 Python 依賴..."
    pip install bleak pyserial --break-system-packages
}

echo "啟動橋接器..."
sudo python3 "$SCRIPT_DIR/bridge.py" --mac "$MAC"
