# 听评研究方案（感知验证，非临床）

目的（只验证两件事，不主张疗效）：
1. 验证器分数与人类感知唤醒度的一致性（跨域效度的感知锚点）。
2. ISO 规划器是否比 direct-jump / endpoint-only 产生更平滑、更符合
   目标轨迹的感知体验（独立于验证器的评价通道）。

## 设计
- N ≈ 24，被试内设计，每人约 15 分钟。
- 每人听 8–12 条轨迹，每条约 24–30 秒（K 段各 4–6 秒拼接）。
- 条件：direct-to-calm / rule interpolation / full ISO-BoN
  （时间允许加 endpoint-only BoN）。
- 相同起点/终点条件配对；条件与顺序随机化（拉丁方或完全随机）；
  不显示方法名称；建议耳机。

## 每条轨迹四个 7 点 Likert 问题
1. Endpoint calmness — "How calm did the music feel by the end?"
2. Transition smoothness — "How smooth and gradual was the emotional
   transition?"
3. Start match — "How well did the beginning match the stated initial
   arousal level?"
4. Overall trajectory — "How well did the sequence move from the stated
   starting state toward calmness?"

另：随机抽取的单段音频若干，让被试直接评 perceived arousal（1-9），
用于计算 Spearman ρ(verifier score, perceived arousal) —— 一个系数
同时补验证器感知效度与规划器独立评价。

## 禁问（超出本研究范围与伦理边界）
- Did this reduce your anxiety?
- Is this therapeutically effective? / Would this treat insomnia?

## 分析（scripts/analyze_listening.py，数据到手即跑）
- 混合效应模型：rating ~ planner + (1|participant) + (1|trajectory)
- 报告：各 planner 边际均值、成对差、95% CI、Holm 校正、
  被试层面配对效应量。
- ρ(proxy arousal, perceived arousal)。

## 论文措辞
"a perceptual validation study with N listeners" ——
不称 clinical / therapeutic efficacy study。

## Stone 需要做的
- 确认所在机构对这类无干预听评的伦理豁免/轻量流程。
- 招 ~24 人（同学/实验室即可，记录年龄段与音乐背景两项人口学）。
- 刺激文件与在线问卷（Google Form 或本地网页）由 Claude 生成，
  ISO 实验跑完当天交付。
