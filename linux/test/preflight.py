#!/usr/bin/env python3
"""
robot_sim2real — 樹梅派部署前置作業自動檢查
=============================================
在樹梅派上執行，會依序檢查所有前置條件並回報狀態。

執行方式（從 imu/ repo 根目錄）：
  python3 linux/test/preflight.py

或透過一鍵腳本：
  bash run_preflight.sh
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── 路徑（與 test_policy.py 一致）──────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
_LINUX_IMU = _HERE.parent / "imu"
_REPO_ROOT = _HERE.parent.parent
_KSIM_ROOT = Path("/home/andykuo/ksim-gym")

DEFAULT_POLICY   = _KSIM_ROOT / "kbot_robot" / "Policies" / "kbot_zero_position.kinfer"
RECORDINGS_DIR   = _KSIM_ROOT / "recordings"
BRIDGE_PY        = _LINUX_IMU / "bridge_h30.py"
START_SH         = _LINUX_IMU / "start_h30.sh"
TEST_POLICY_PY   = _HERE / "test_policy.py"

H30_CANDIDATE_PORTS = [
    "/dev/ttyACM0", "/dev/ttyACM1",
    "/dev/ttyUSB1", "/dev/ttyUSB2",
]

# ── 顏色輸出（若 terminal 支援）──────────────────────────────────────────────
_use_color = sys.stdout.isatty()
def _green(s): return f"\033[32m{s}\033[0m" if _use_color else s
def _yellow(s): return f"\033[33m{s}\033[0m" if _use_color else s
def _red(s):   return f"\033[31m{s}\033[0m"   if _use_color else s
def _bold(s):  return f"\033[1m{s}\033[0m"    if _use_color else s

PASS = _green("[OK]   ")
WARN = _yellow("[WARN] ")
FAIL = _red("[FAIL] ")
SKIP = "       "

results: list[tuple[str, str, str]] = []   # (label, status, detail)

def ok(label, detail=""):   results.append((label, "OK",   detail)); print(f"  {PASS}{label}  {detail}")
def warn(label, detail=""): results.append((label, "WARN", detail)); print(f"  {WARN}{label}  {detail}")
def fail(label, detail=""): results.append((label, "FAIL", detail)); print(f"  {FAIL}{label}  {detail}")
def info(label, detail=""): print(f"  {SKIP}{label}  {detail}")


# ═══════════════════════════════════════════════════════════════════════════════
# 各類檢查
# ═══════════════════════════════════════════════════════════════════════════════

def check_python():
    print(_bold("\n[1] Python 環境"))
    v = sys.version_info
    label = f"Python {v.major}.{v.minor}.{v.micro}"
    if v >= (3, 8):
        ok(label)
    else:
        fail(label, "需要 >= 3.8")

    for pkg, import_name in [("numpy", "numpy"), ("onnxruntime", "onnxruntime"),
                               ("pyserial", "serial"), ("struct", "struct")]:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            ok(f"import {pkg}", f"版本 {ver}")
        except ImportError:
            fail(f"import {pkg}", "未安裝 → pip install " + pkg)


def check_files():
    print(_bold("\n[2] 必要檔案"))
    for label, path in [
        ("bridge_h30.py",  BRIDGE_PY),
        ("start_h30.sh",   START_SH),
        ("test_policy.py", TEST_POLICY_PY),
    ]:
        if path.exists():
            ok(label, str(path))
        else:
            fail(label, f"找不到 {path}")

    # run_test.sh
    run_sh = _REPO_ROOT / "run_test.sh"
    if run_sh.exists():
        ok("run_test.sh", str(run_sh))
    else:
        warn("run_test.sh", "不影響功能，但一鍵啟動腳本缺失")


def check_policy():
    print(_bold("\n[3] Policy 檔案 (.kinfer)"))
    if DEFAULT_POLICY.exists():
        size_mb = DEFAULT_POLICY.stat().st_size / 1_000_000
        ok(DEFAULT_POLICY.name, f"{size_mb:.1f} MB  路徑: {DEFAULT_POLICY}")
    else:
        fail(DEFAULT_POLICY.name,
             f"找不到 {DEFAULT_POLICY}\n"
             "        → 確認 kbot_robot/Policies/ 目錄是否有 .kinfer 檔案")
        return

    # 嘗試快速載入驗證
    try:
        import tarfile, json, tempfile
        import onnxruntime as ort
        with tempfile.TemporaryDirectory() as d:
            with tarfile.open(DEFAULT_POLICY, "r:gz") as tar:
                tar.extractall(d)
            with open(os.path.join(d, "metadata.json")) as f:
                meta = json.load(f)
            _ = ort.InferenceSession(os.path.join(d, "step_fn.onnx"))
        inputs_needed = [k for k in ["joint_angles", "joint_angular_velocities",
                                      "projected_gravity", "accelerometer",
                                      "gyroscope", "carry", "command", "time"]]
        ok("kinfer 載入成功",
           f"carry_size={meta.get('carry_size')}  "
           f"commands={meta.get('num_commands', 0)}")
    except Exception as e:
        fail("kinfer 載入失敗", str(e))


def check_recordings():
    print(_bold("\n[4] 錄製資料（replay 用）"))
    if not RECORDINGS_DIR.exists():
        warn("recordings/ 目錄", f"不存在（{RECORDINGS_DIR}）→ replay 模式無法使用，其他模式不影響")
        return
    csvs = sorted(RECORDINGS_DIR.glob("*_actions.csv"))
    if csvs:
        ok(f"找到 {len(csvs)} 筆錄製", f"最新: {csvs[-1].name}")
        for p in csvs[-3:]:
            info("  •", str(p))
    else:
        warn("recordings/ 目錄", f"存在但無 *_actions.csv → 需先在 train_v1/test_policy.py 錄製")


def check_virtual_serial():
    print(_bold("\n[5] 虛擬串口（tty0tty）"))
    # 檢查 lsmod
    try:
        r = subprocess.run(["lsmod"], capture_output=True, text=True, timeout=5)
        if "tty0tty" in r.stdout:
            ok("tty0tty 模組", "已載入")
        else:
            fail("tty0tty 模組", "未載入 → 先執行 sudo bash linux/imu/start_h30.sh")
    except Exception as e:
        warn("tty0tty 模組", f"無法執行 lsmod: {e}")

    for dev in ["/dev/tnt0", "/dev/tnt1"]:
        if Path(dev).exists():
            ok(dev, "存在")
        else:
            fail(dev, f"不存在 → 先執行 sudo bash linux/imu/start_h30.sh")

    # /dev/ttyUSB0 symlink
    usb0 = Path("/dev/ttyUSB0")
    if usb0.is_symlink():
        target = os.readlink("/dev/ttyUSB0")
        if "tnt1" in target:
            ok("/dev/ttyUSB0 → tnt1", f"symlink 正確（→ {target}）")
        else:
            warn("/dev/ttyUSB0", f"symlink 指向 {target}，非 tnt1")
    elif usb0.exists():
        warn("/dev/ttyUSB0", "存在但不是 symlink（可能是真實裝置占用）")
    else:
        fail("/dev/ttyUSB0", "不存在 → 先執行 sudo bash linux/imu/start_h30.sh")


def check_imu_hardware():
    print(_bold("\n[6] H30 Mini USB 裝置"))
    found = [p for p in H30_CANDIDATE_PORTS if Path(p).exists()]
    if found:
        ok("H30 候選串口", f"找到 {found}")
        for p in found:
            # 確認可讀寫
            ok_rw = os.access(p, os.R_OK | os.W_OK)
            if ok_rw:
                ok(f"  {p} 讀寫權限", "OK")
            else:
                warn(f"  {p} 讀寫權限", "需要 sudo 或加入 dialout 群組")
    else:
        warn("H30 串口", f"未在 {H30_CANDIDATE_PORTS} 找到裝置 → IMU 尚未插入或驅動未載入")
        info("", "其他測試模式（check/replay）不需要 IMU，仍可繼續")


def check_can():
    print(_bold("\n[7] CAN 介面（馬達用，可選）"))
    try:
        r = subprocess.run(["ip", "link", "show", "can0"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            if "UP" in r.stdout:
                ok("can0", "介面存在且 UP")
            else:
                warn("can0", "介面存在但未 UP → ip link set can0 up type can bitrate 1000000")
        else:
            warn("can0", "介面不存在 → --dry-run 模式不需要 CAN")
    except FileNotFoundError:
        warn("ip 指令", "找不到 ip 工具")
    except Exception as e:
        warn("can0 檢查", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 摘要報告
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary():
    print("\n" + "=" * 60)
    print(_bold("=== 前置檢查摘要 ==="))
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    n_warn = sum(1 for _, s, _ in results if s == "WARN")
    n_ok   = sum(1 for _, s, _ in results if s == "OK")
    print(f"  通過: {n_ok}  警告: {n_warn}  失敗: {n_fail}")

    if n_fail == 0 and n_warn == 0:
        print(f"\n  {_green('全部通過 ✓')}  可以開始測試")
    elif n_fail == 0:
        print(f"\n  {_yellow('有警告，但基本可用')}  確認警告項目後再繼續")
    else:
        print(f"\n  {_red('有失敗項目，請先修正再測試')}")
        fail_items = [(l, d) for l, s, d in results if s == "FAIL"]
        print("\n  需修正：")
        for label, detail in fail_items:
            print(f"    • {label}: {detail}")

    print("\n" + "=" * 60)
    print(_bold("下一步指引："))
    print("""
  ① 前置設定（每次開機一次）：
       sudo bash linux/imu/start_h30.sh

  ② 快速推論驗證（不需任何硬體）：
       ./run_test.sh --mode check

  ③ 連接 H30 Mini，空跑模型（不接馬達）：
       ./run_test.sh --mode policy --imu --dry-run

  ④ 錄製重播比對（不接馬達）：
       ./run_test.sh --mode replay --dry-run

  ⑤ 接馬達正式啟動（請親自操作）：
       ./run_test.sh --mode policy --imu --can can0
    """)
    print("=" * 60)

    return n_fail == 0


def main():
    print(_bold("robot_sim2real 部署前置作業檢查"))
    print(f"工作目錄: {_REPO_ROOT}")
    print(f"ksim 根目錄: {_KSIM_ROOT}")

    check_python()
    check_files()
    check_policy()
    check_recordings()
    check_virtual_serial()
    check_imu_hardware()
    check_can()

    ok_to_go = print_summary()
    sys.exit(0 if ok_to_go else 1)


if __name__ == "__main__":
    main()
