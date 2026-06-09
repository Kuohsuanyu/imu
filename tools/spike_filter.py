#!/usr/bin/env python3
"""
突波過濾器 — 離線 CSV 預處理 + 即時串流兩用。

離線用法：
  python tools/spike_filter.py input.csv --out input_smooth.csv

即時用法（在 test_policy.py 裡 import）：
  from tools.spike_filter import SpikeFilter
  sf = SpikeFilter(n_joints=10, max_jump_deg=20.0, max_consecutive=3)
  filtered, abort_reason = sf.update(new_target_array)
  if abort_reason:
      trigger_estop(abort_reason)
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


TARGET_COLS = [
    "target_dof_right_hip_pitch_04",
    "target_dof_right_hip_roll_03",
    "target_dof_right_hip_yaw_03",
    "target_dof_right_knee_04",
    "target_dof_right_ankle_02",
    "target_dof_left_hip_pitch_04",
    "target_dof_left_hip_roll_03",
    "target_dof_left_hip_yaw_03",
    "target_dof_left_knee_04",
    "target_dof_left_ankle_02",
]

JOINT_NAMES = [c.replace("target_dof_", "") for c in TARGET_COLS]


class SpikeFilter:
    """
    即時突波過濾器，每幀呼叫一次 update()。

    邏輯：
      - 單幀突波（跳躍超過 max_jump）→ 保持上一幀值，計數 +1
      - 連續突波達 max_consecutive 幀   → 回傳 abort_reason 字串，呼叫方應觸發 E-STOP
      - 正常幀                          → 重設計數，直接通過

    Args:
        n_joints:        關節數量
        max_jump_deg:    單幀最大允許跳躍（度），預設 20°
        max_consecutive: 連續突波中止閾值（幀數），預設 3
        joint_names:     關節名稱列表（用於錯誤訊息）
    """

    def __init__(
        self,
        n_joints: int = 10,
        max_jump_deg: float | list[float] = 35.0,
        max_consecutive: int = 5,
        joint_names: list[str] | None = None,
    ):
        if isinstance(max_jump_deg, (int, float)):
            self.max_jump = np.full(n_joints, math.radians(float(max_jump_deg)))
        else:
            self.max_jump = np.array([math.radians(v) for v in max_jump_deg])
        self.max_consecutive = max_consecutive
        self.joint_names = joint_names or [f"joint_{i}" for i in range(n_joints)]
        self._prev: np.ndarray | None = None
        self._spike_counts = np.zeros(n_joints, dtype=int)
        self.total_fixed = 0

    def reset(self) -> None:
        self._prev = None
        self._spike_counts[:] = 0

    def update(self, target: np.ndarray) -> tuple[np.ndarray, str | None]:
        """
        傳入新一幀的目標關節角度（弧度），回傳 (filtered, abort_reason)。
        abort_reason 為 None 表示正常；非 None 表示應觸發 E-STOP。
        """
        target = np.asarray(target, dtype=float)

        if self._prev is None:
            self._prev = target.copy()
            return target.copy(), None

        diff = target - self._prev
        is_spike = np.abs(diff) > self.max_jump

        # 重設非突波關節的計數
        self._spike_counts[~is_spike] = 0
        self._spike_counts[is_spike] += 1

        # 連續突波 → 中止
        bad = np.where(self._spike_counts >= self.max_consecutive)[0]
        if len(bad) > 0:
            names = ", ".join(self.joint_names[i] for i in bad)
            counts = ", ".join(str(self._spike_counts[i]) for i in bad)
            reason = (
                f"連續突波 {counts} 幀 [{names}]，"
                f"最大跳躍 {math.degrees(float(np.max(np.abs(diff[bad])))):.1f}°"
            )
            return target.copy(), reason

        # 單幀突波 → 保持上一幀值
        output = np.where(is_spike, self._prev, target)
        if np.any(is_spike):
            self.total_fixed += int(np.sum(is_spike))

        self._prev = output.copy()
        return output, None


# ──────────────────────────────────────────────
# 離線 CSV 預處理
# ──────────────────────────────────────────────

def smooth_csv(
    src: Path,
    dst: Path,
    max_jump_deg: float = 35.0,
    max_consecutive: int = 5,
    verbose: bool = True,
) -> bool:
    """
    回傳 True 表示成功，False 表示檔案中存在連續突波（已中止）。
    """
    with open(src, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        print("[ERROR] CSV 為空")
        return False

    # 只處理 target 欄位存在的關節
    active_cols = [c for c in TARGET_COLS if c in fieldnames]
    max_jump_rad = math.radians(max_jump_deg)
    n = len(rows)
    fixed_total = 0

    # 對每個關節各自處理（在原始序列上操作，避免級聯效應）
    data = {col: [float(rows[i][col]) for i in range(n)] for col in active_cols}

    for col in active_cols:
        vals = data[col]
        i = 1
        while i < n - 1:
            jump = abs(vals[i] - vals[i - 1])
            if jump <= max_jump_rad:
                i += 1
                continue

            # 找出突波群的結尾（返回正常的第一幀）
            j = i
            while j < n - 1 and abs(vals[j] - vals[j - 1]) > max_jump_rad:
                j += 1

            cluster_len = j - i + 1
            if cluster_len > max_consecutive:
                name = col.replace("target_dof_", "")
                print(
                    f"[ABORT] frame {i}~{j}: 持續 {cluster_len} 幀突波 [{name}]，"
                    f"最大跳躍 {math.degrees(jump):.1f}°"
                )
                return False

            # 短突波：線性插值覆蓋（用群前一幀和群後一幀）
            prev_val = vals[i - 1]
            next_val = vals[j] if j < n else prev_val
            span = j - i + 1
            for k in range(span):
                vals[i + k] = prev_val + (next_val - prev_val) * (k + 1) / (span + 1)
            fixed_total += span
            i = j + 1

        data[col] = vals

    out_rows = []
    for i, row in enumerate(rows):
        new_row = dict(row)
        for col in active_cols:
            new_row[col] = f"{data[col][i]:.10f}"
        out_rows.append(new_row)

    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    if verbose:
        print(f"[OK]  {n} 幀處理完成，修正突波點 {fixed_total} 個 → {dst}")

    return True


def main():
    parser = argparse.ArgumentParser(description="CSV 突波過濾器")
    parser.add_argument("input", help="輸入 CSV 路徑")
    parser.add_argument("--out", help="輸出路徑（預設：同名加 _smooth）")
    parser.add_argument("--max-jump", type=float, default=35.0, help="單幀最大跳躍度數（預設 35°）")
    parser.add_argument("--max-consecutive", type=int, default=5, help="連續突波中止幀數（預設 5）")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"[ERROR] 找不到檔案: {src}")
        sys.exit(1)

    dst = Path(args.out) if args.out else src.with_name(src.stem + "_smooth.csv")

    ok = smooth_csv(src, dst, max_jump_deg=args.max_jump, max_consecutive=args.max_consecutive)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
