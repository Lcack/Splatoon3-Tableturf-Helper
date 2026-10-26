# 自定义牌组配置

牌组配置是一个 JSON 文件，必须包含正好 15 张不重复的卡牌。

```json
{
  "name": "我的牌组",
  "cards": [
    "MiniGame_TakoPod",
    "MiniGame_BombCurling",
    "MiniGame_SaberLight00",
    "MiniGame_BombRobot",
    "MiniGame_Tower",
    "MiniGame_ShooterFlash00",
    "MiniGame_StringerShort00",
    "MiniGame_BrushNormal00",
    "MiniGame_StringerNormal00",
    "MiniGame_SpGreatBarrier",
    "MiniGame_TakoCommander",
    "MiniGame_ShockSonar",
    "MiniGame_Tamaire",
    "MiniGame_TakoStrike",
    "MiniGame_TakoDozer"
  ]
}
```

`cards` 优先使用 `card_id`。如果中英文名称在卡牌数据中唯一，也可以直接写名称。
在采集窗口或视频分析窗口点击“加载牌组配置”即可启用。
点击“自动牌组”会清除指定牌组，恢复从全部 266 张卡的逐槽识别。
