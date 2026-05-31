# KBot 問題紀錄

## P1 — CAN ID Mismatch 警告（已根治）

**現象**
```
[WARN] 馬達 41 讀取失敗（3次）: Protocol error: Feedback CAN ID mismatch
```
控制正常，但每個控制週期大量出現警告，干擾輸出。

**根本原因**
`PyRobstrideDriver.get_actuator_state(mid)` 的設計是：送一個讀取 request，然後從 CAN socket 讀下一個 frame。但 8 顆馬達同時廣播狀態，下一個 frame 不一定是 `mid` 的回應，可能是其他馬達的廣播幀 → driver 比對 ID 不符就拋出 exception。

成功率約 30–40%，但指令送出完全不受影響。

**解法（最終）**
仿 Rust firmware（`deploye_robot/firmware/src/actuator.rs: read_responses_update`）的做法：
- 開 raw socketcan socket 讀**所有**幀
- 解析 mux（byte[3] & 0x1F）：只處理 mux=0x02（feedback）
- 從 byte[1] 取 motor_id，路由到對應的狀態字典
- 背景執行緒持續更新，`read_states` 直接讀字典，完全不依賴 `get_actuator_state()`

實作：`test_policy.py` 的 `CanStateReader` 類別。

---

## P2 — Unknown mux value: 21（Python driver panic）

**現象**
```
thread panicked at src/protocol.rs:421:18:
Unknown mux value: 21
PanicException: Unknown mux value: 21
```
程式直接崩潰，發生在 `enable_actuator()` 時。

**根本原因**
mux=0x15（十進位 21）是 Robstride 的 unsolicited fault frame（非請求型錯誤幀）。
Rust firmware 在 `handle_response()` 中有特殊處理（靜默忽略），Python driver 沒有，收到就 panic。
馬達 33 在 can0 上廣播此幀，導致任何操作 can0 的 Python driver 都會崩潰。

**解法**
`CanStateReader` 讀到 mux≠0x02 時一律靜默忽略（含 0x15），完全繞過 Python driver 的讀取路徑。Enable 指令送出仍使用 PyRobstrideDriver，只是讀取不再走 `get_actuator_state()`。

---

## P3 — Robstride04 馬達間歇不響應（31, 34, 41, 44）

**現象**
馬達一開始可以動，運作一段時間後不再響應任何指令。`robstride state` 顯示可找到馬達（faults=0），但位置不動。

**疑似原因**
過熱保護或過載保護進入 fault state，需要重新 enable 才能恢復。

**解法**
`setup_driver` Phase 3 互動確認：
- 背景持續對所有馬達送 hold 指令（50 Hz）
- 使用者摸馬達感受是否有 kp 阻力，確認後輸入 ID
- 輸入 `r ID` 對特定馬達重新 enable 直到鎖住

---

## P4 — CAN 匯流排緩衝區滿（ENOBUFS）

**現象**
```
RuntimeError: IO error: No buffer space available (os error 105)
```

**根本原因**
`get_actuator_state()` 每次呼叫都會送出一個 CAN request frame。在讀取重試迴圈中短時間大量發送，超過 kernel CAN socket 緩衝區。

**解法**
廢棄重試迴圈，改為 `CanStateReader` 被動讀取，不主動送 request。

---

## P6 — CanStateReader 無法取得初始位置（0/N 顆）

**現象**
```
等待所有馬達回傳位置資料... 完成（0/1 顆）
[WARN] 未收到資料的馬達: [44]
```
等待 3 秒後仍然讀不到任何馬達位置。

**根本原因**
Robstride 馬達**不主動廣播**狀態幀。只有收到控制指令後，才會回傳 mux=0x02 的 feedback 幀。
啟動 CanStateReader 後若沒有任何指令送出，reader 永遠不會收到資料。

**解法**
啟動 reader 後，立刻對每顆馬達送一次 `kp=0 kd=0` 的無力指令（不會移動），觸發馬達回傳第一幀位置資料，reader 捕捉後即可正常運作。

---

## P7 — 啟動時馬達瞬間扯動（進保護）

**現象**
程式啟動後馬達會突然抖動一下，有時直接觸發過載保護進入 fault state。

**根本原因（三層）**

1. **Phase 3 send_loop 送 position=0°**：CanStateReader 剛啟動時還沒有資料，fallback 到 `hold_pos=0.0`，帶全力 kp 強制移到 0° → 瞬間大力。

2. **home_ramp 從 joint_pos=0 出發**：`joint_pos` 初始化為全零，不是馬達實際位置。第一步計算的 step_pos 是以 0° 為基準，但馬達可能在 -50°，PD 誤差 50° → 大力。

3. **home_ramp 用誤差判斷提前結束**：初始位置錯誤（0°）→ 第一步送出 → reader 讀到實際位置（-49.8°）→ 誤差 < 1° → 立刻結束，整個 ramp 只花 0.0s。

**解法**
- Phase 3 send_loop：等所有馬達都有資料（來自 P6 的 prime 指令）再開始，沒有資料就 `continue`（完全不送指令），不送 0°。
- home_ramp：等所有馬達有資料 → 讀實際起點位置 → 固定 3 秒線性插值到目標，不以誤差判斷提前結束。

---

## P5 — SSH 斷線（切換網路時）

**現象**
切換到手機熱點後 SSH 斷線，因為 IP 位址改變。

**解法**
使用 `raspberrypi.local`（mDNS / Avahi），不管 IP 是什麼都能連：
```bash
ssh fr01@raspberrypi.local
```
手機熱點優先級設為 10（高於實驗室網路 5），手機開著自動切熱點，關掉自動切回實驗室網路。

---

## CAN 幀格式參考（Robstride Feedback，mux=0x02）

```
byte[0]   host_id
byte[1]   actuator_can_id   ← motor ID（用於分類）
byte[2]   fault_flags
byte[3]   mux（低 5 bits）  0x02=feedback, 0x15=fault
byte[4-7] socketcan header（len/pad/res0/dlc）
byte[8-9]   angle_be    big-endian u16 → position
byte[10-11] vel_be      big-endian u16 → velocity
byte[12-13] torque_be   big-endian u16 → torque
byte[14-15] temp_be     big-endian u16 → temperature
```

**量程換算**（CAN u16 [0,65535] → 物理量）：
| 型號 | 角度 | 角速度 |
|------|------|--------|
| Robstride04 | ±4π rad | ±15 rad/s |
| Robstride03 | ±4π rad | ±20 rad/s |
| Robstride02 | ±4π rad | ±44 rad/s |
