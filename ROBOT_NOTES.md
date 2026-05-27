# KBot 腿部馬達控制 — 操作紀錄

## 硬體配置

### CAN 匯流排分配

| CAN | 腿 | 馬達 ID |
|-----|----|---------|
| can0 | 右腿 | 41, 42, 43, 44, 45 |
| can1 | 左腿 | 31, 32, 33, 34, 35 |

### 馬達對照表

| ID | 關節名稱 | 型號 | kp | kd | 最大扭矩 |
|----|----------|------|----|----|----------|
| 31 | dof_left_hip_pitch_04  | Robstride04 | 150 | 24.722 | 84 Nm |
| 32 | dof_left_hip_roll_03   | Robstride03 | 200 | 26.387 | 42 Nm |
| 33 | dof_left_hip_yaw_03    | Robstride03 | 100 |  3.419 | 42 Nm |
| 34 | dof_left_knee_04       | Robstride04 | 150 |  8.654 | 84 Nm |
| 35 | dof_left_ankle_02      | Robstride02 |  40 |  0.990 | 12 Nm |
| 41 | dof_right_hip_pitch_04 | Robstride04 | 150 | 24.722 | 84 Nm |
| 42 | dof_right_hip_roll_03  | Robstride03 | 200 | 26.387 | 42 Nm |
| 43 | dof_right_hip_yaw_03   | Robstride03 | 100 |  3.419 | 42 Nm |
| 44 | dof_right_knee_04      | Robstride04 | 150 |  8.654 | 84 Nm |
| 45 | dof_right_ankle_02     | Robstride02 |  40 |  0.990 | 12 Nm |

### 站姿零點（ZEROS）

| 關節 | 目標角度 |
|------|----------|
| 左髖俯仰 (31) | +20° |
| 左髖側傾 (32) |   0° |
| 左髖偏擺 (33) |   0° |
| 左膝     (34) | +50° |
| 左踝     (35) | -30° |
| 右髖俯仰 (41) | -20° |
| 右髖側傾 (42) |   0° |
| 右髖偏擺 (43) |   0° |
| 右膝     (44) | -50° |
| 右踝     (45) | +30° |

左右腿的髖俯仰與膝關節角度互為鏡像（正負號相反）。

---

## SSH 連線

```bash
# 任何網路都通（mDNS）
ssh fr01@raspberrypi.local

# 手機熱點固定 IP
ssh fr01@172.20.10.100
```

### 網路優先級
- 手機熱點（Cindy1）優先級 10
- 實驗室網路（F458）優先級 5
- 手機熱點開著 → 自動連手機；手機關掉 → 自動切回實驗室

---

## 測試腳本使用方法

工作目錄：`~/robot_data/imu`

### 常用指令

```bash
# 正弦波測試 — 全腿（左腿 can1，右腿 can0）
python linux/test/test_policy.py --mode sine \
  --ids 31,32,33,34,35,41,42,43,44,45 \
  --can can0 --left-can can1

# 正弦波測試 — 只動髖俯仰+膝，其餘鎖在站姿
python linux/test/test_policy.py --mode sine \
  --ids 31,32,33,34,41,42,43,44 \
  --active-ids 31,34,41,44 \
  --can can0 --left-can can1

# 站姿保持
python linux/test/test_policy.py --mode stand \
  --ids 31,32,33,34,35,41,42,43,44,45 \
  --can can0 --left-can can1

# Policy 推論（假 IMU）
python linux/test/test_policy.py --mode policy --dry-run

# Policy 推論（真 IMU + 馬達）
python linux/test/test_policy.py --mode policy \
  --imu --can can0 --left-can can1 \
  --ids 31,32,33,34,35,41,42,43,44,45
```

### 啟動流程（setup_driver）

程式啟動後自動執行三個階段：

**Phase 1 — Ping**：對所有馬達發送 ping 確認在線

**Phase 2 — Enable**：啟用所有馬達（Robstride04 重送 3 次確保收到）

**Phase 3 — 互動確認**：背景持續對馬達送保持指令，終端顯示待確認清單

```
待確認: [31, 32, 33, 34, 41, 42, 43, 44]
已確認: []
> 
```

| 輸入 | 動作 |
|------|------|
| `31,34` | 確認這兩顆已鎖住，移出待確認清單 |
| `r 31,34` | 對 31、34 重新 enable（馬達沒鎖住時使用） |
| Enter | 全部跳過，直接繼續 |
| `q` | 中止程式 |

全部確認後進入 Home Ramp，緩慢移動到站姿。

---

## CAN ID Mismatch 說明

多顆馬達同時在線時，每顆馬達會持續廣播狀態幀。`get_actuator_state(mid)` 讀到的幀可能來自其他馬達，導致 `Feedback CAN ID mismatch` 警告。

**影響**：只影響狀態讀取（位置回饋），不影響指令發送。馬達仍然接收並執行位置指令。

**正常現象**：讀取成功率約 30–40%，機器人仍可正常行走。

---

## 安全關機

```bash
sudo shutdown -h now
```

等綠燈停止閃爍後再拔電源。

---

## Git 操作

```bash
# 更新到最新版
git pull

# 推送更改（token 已設定在 remote URL）
git add linux/test/test_policy.py
git commit -m "說明"
git push
```
