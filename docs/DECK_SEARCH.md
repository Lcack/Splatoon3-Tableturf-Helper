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

以下命令在项目根目录执行，脚本使用项目内 `.venv`：

```powershell
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

## 后续搜索扩展

以下是工程方案，当前命令仍采用上述随机模板，不代表这些阶段已经实现：

1. 保留随机模板作为基准，固定场景和 seed，记录每张候选卡的曝光与完整牌组表现。
2. 从优胜牌组生成局部替换候选，同时加入未充分覆盖的卡牌并去重。
3. 分析单卡和双卡组合时控制场景、共现与样本量，避免直接把所在牌组的平均胜率当作单卡能力。
4. 精英邻域与加权候选一起复测，最终使用未参与搜索的 seed 和场景，再进入实机验证。

模板之外的卡牌组合不在当前搜索覆盖范围内，不能声称穷尽卡池或找到全局最优。玩家经验模板见 [STRATEGY_NOTES](STRATEGY_NOTES.md)，评价目标见 [AI_ROADMAP](AI_ROADMAP.md)。

## 多机执行设想

当前没有集群调度。后续可按完整候选牌组和场景拆分任务，为配置与 seed 生成固定任务 ID；各 worker 写独立分片，由主进程合并、去重和排名。不要让多个 worker 同时写同一个 JSONL。

