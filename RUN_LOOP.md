# 长期循环运行

## 已具备的保护

- 每完成一局立即追加一行 JSONL，并调用 `flush` 与 `fsync`。
- 崩溃留下的最后一条不完整 JSON 会在下次 `--resume` 时自动截掉。
- 每个输出文件有独立进程锁，禁止两个进程同时写入。
- `meta.json` 保存 seed、策略、NPC 等级、模拟次数和时间上限。
- 使用不同配置续跑会直接报错，避免混入不同实验。
- 每局更新汇总状态，退出后写入 `summary.json`。
- 汇总按策略对阵、NPC、NPC 等级、场地和 NPC 等级加场地分组。
- `--max-minutes` 到达后正常退出，下一次可使用 `--resume` 继续。

## 第一次运行

先做单局验证：

```powershell
cd E:\Splatoon\tableturf\AutoTableTurf-rebuild

.\.venv\Scripts\tableturf.exe selfplay `
  --games 1 `
  --seed 20261022 `
  --level 3 `
  --policy-a mcts `
  --policy-b greedy `
  --simulations 32 `
  --time-limit 1 `
  --output artifacts\selfplay\production.jsonl
```

确认生成以下文件：

```text
artifacts/selfplay/production.jsonl
artifacts/selfplay/production.jsonl.meta.json
artifacts/selfplay/production.jsonl.summary.json
artifacts/selfplay/production.jsonl.lock
```

## 手工续跑

```powershell
.\.venv\Scripts\tableturf.exe selfplay `
  --games 1000000 `
  --seed 20261022 `
  --level 3 `
  --policy-a mcts `
  --policy-b greedy `
  --simulations 32 `
  --time-limit 1 `
  --max-minutes 60 `
  --resume `
  --output artifacts\selfplay\production.jsonl
```

每次最多运行 60 分钟。到时间后查看：

```powershell
Get-Content artifacts\selfplay\production.jsonl.summary.json
```

再次执行相同命令即可继续。

## 自动分批续跑

```powershell
.\scripts\run_loop.ps1 `
  -OutputFile artifacts\selfplay\production.jsonl `
  -PolicyA mcts `
  -PolicyB greedy `
  -Seed 20261022 `
  -Level 3 `
  -Simulations 32 `
  -TimeLimitSeconds 1 `
  -MinutesPerBatch 60
```

脚本会每批运行 60 分钟，然后自动以同一配置续跑。按 `Ctrl+C` 停止；最后完整记录的局会保留。

## 新实验

修改 seed、策略、等级、模拟次数或时间上限时，必须使用新的输出路径。直接对旧文件更换配置会被拒绝。

## 注意

当前循环产生的是自对弈和评估数据。MCTS 搜索树每局结束后仍会丢弃，不会跨局持久化学习。

