# 牌组搜索

## 当前模板

每套候选牌组固定 15 张且不重复：

```text
2 张面积 15+
4 张面积 12-13 的普通牌
1 张面积 12、3 SP 的大招牌
3 张面积 8，其中固定包含 MiniGame_KoJudgekun
4 张面积 4-5
1 张面积 3
```

牌组列表由固定 seed 确定性生成并去重。改变候选数量不会改变前面已生成的
牌组，但会改变本次实验的 meta 配置，因此正式实验应固定参数。

## 对手与种子

- NPC 等级：3。
- 固定 20 个场景：覆盖三类 AI 标签。
- 每套牌在同一组场景、同一组 seed 下运行 20 局。
- 我方策略：MCTS，默认 32 次模拟、每步软上限 1 秒。
- NPC 策略：Greedy 代理。

搜索结果只表示在这些代理条件下表现最好的候选牌组，不代表真实游戏全局
最优。

## 启动并自动显示监控窗口

```powershell
cd E:\Splatoon\tableturf\AutoTableTurf-rebuild

.\scripts\run_deck_search.ps1 `
  -OutputDir artifacts\deck_search\main `
  -Decks 120 `
  -GamesPerDeck 20 `
  -Seed 20261022 `
  -Simulations 32 `
  -TimeLimitSeconds 1 `
  -MinutesPerBatch 240
```

监控窗口显示：

- 本轮运行时间。
- 已完成牌组和总局数。
- 当前牌组、NPC 场景和对局序号。
- 实时前 20 名排行榜。
- 胜、负、平、胜率和平均净胜墨迹。

选择排行榜中的牌组可查看 15 张卡牌 ID。

## 输出文件

```text
artifacts/deck_search/main/
├─ meta.json
├─ decks.json
├─ games.jsonl
├─ progress.json
└─ leaderboard.json
```

每局完成立即追加到 `games.jsonl`。进程停止后，再次运行相同参数并添加
`--resume` 可继续。

