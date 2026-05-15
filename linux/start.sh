#!/bin/bash
# BLE IMU 橋接器啟動腳本（tty0tty 版本）
# 使用方式：bash start.sh [裝置名稱]

NAME="${1:-WT901BLE67}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODULE_DIR="$SCRIPT_DIR/tty0tty/module"

echo "=== BLE IMU Bridge (tty0tty) ==="
echo "裝置名稱: $NAME"
echo ""

# ── 確認 Python 依賴 ──────────────────────────────────────────────────────────
python3 -c "import bleak, serial" 2>/dev/null || {
    echo "安裝 Python 依賴..."
    pip install bleak pyserial --break-system-packages
}

# ── 載入 tty0tty 核心模組 ─────────────────────────────────────────────────────
if [ ! -c /dev/tnt0 ]; then
    if [ -f "$MODULE_DIR/tty0tty.ko" ]; then
        echo "載入 tty0tty 模組..."
        sudo insmod "$MODULE_DIR/tty0tty.ko"
    else
        echo "[ERROR] 找不到 tty0tty.ko，請先編譯："
        echo "  cd $MODULE_DIR && make"
        exit 1
    fi
fi

# 確認 /dev/tnt* 權限
sudo chmod 666 /dev/tnt0 /dev/tnt1 2>/dev/null || true

# 建立 /dev/ttyUSB0 → /dev/tnt1 的 symlink（firmware 讀這端）
if [ ! -e /dev/ttyUSB0 ]; then
    echo "建立 /dev/ttyUSB0 → /dev/tnt1 symlink..."
    sudo ln -sf /dev/tnt1 /dev/ttyUSB0
fi

echo "虛擬串口: /dev/tnt0 (bridge) ↔ /dev/tnt1 → /dev/ttyUSB0 (firmware)"
echo ""

# ── 啟動橋接器 ────────────────────────────────────────────────────────────────
echo "啟動橋接器..."
python3 "$SCRIPT_DIR/bridge.py" --name "$NAME"
