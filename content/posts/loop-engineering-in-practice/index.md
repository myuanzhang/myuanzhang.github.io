---
title: "Loop Engineering 实战：搭你的第一个自治循环"
date: 2026-06-22
draft: false
tags: ["Loop Engineering", "AI", "Claude Code", "Codex", "Agent"]
---

> 读完本文你能先判断**自己到底要不要建 Loop**，再亲手搭出第一个「每日 CI 故障分类」自治循环——只读、只写一个 `TODO.md`、零写代码风险——并理解什么时候该让它承担更多。

> **时效性说明（2026-06-22 核实）**：本文涉及的 `/loop`、`/goal` 命令语法已对照 [Claude Code 官方文档](https://code.claude.com/docs/en/scheduled-tasks) 核实；`/loop` 需 Claude Code **v2.1.72+**，`/goal` 需 **v2.1.139+**，用 `claude --version` 查看。Codex 部分对照 [OpenAI Codex Automations 文档](https://developers.openai.com/codex/app/automations)。这些是快速演化的功能，照做前请以官方文档为准。

## Why — 为什么需要 Loop

过去两年，你从编程 Agent 里拿到结果的方式是这样的：写一个好提示词，给足上下文，读它的回复，再敲下一条。Agent 是工具，你全程握着它，一轮接一轮。**瓶颈是你**——你打字、你判断、你推进每一步。

[Addy Osmani 在 2026-06-07 的原创博客](https://addyosmani.com/blog/loop-engineering/) 里把这件事讲透了：当编程 Agent 进入「自主循环」这一代（Claude Code、OpenAI Codex），让人逐字推进反而成了系统里最慢的环节。Anthropic Claude Code 负责人 Boris Cherny 的说法很直白——[「我不再直接提示 Claude，而是让循环替我提示它、决定下一步；我的工作是写循环」](https://x.com/rohanpaul_ai/status/2063289804708835412)；OpenClaw 作者 Peter Steinberger 则说 [「你不该再手动提示编程 Agent，而该设计让 Agent 自我提示的循环」](https://x.com/steipete/status/2063697162748260627)。

这就是 Loop Engineering（循环工程）：**把「提示 Agent 的那个人」从你身上替换掉，转而设计替你做这件事的系统**。注意它不否定提示词工程——一个 Loop 由许多提示词组成，写得差的提示词放进 Loop，只会让糟糕的结果产出得更快。Loop 工程是叠在提示词工程之上的一层，不是替代。

### Loop Engineering 的演化位置

把它放进上下文里：

```
Prompt Engineering（写好一条指令）
    ↓
Context Engineering（组装上下文窗口）
    ↓
Loop Engineering（设计让 Agent 自驱动的外循环）
```

每一层叠在下一层之上。写不好提示词，Loop 只会更快地产出垃圾。组装不好上下文，Agent 每轮都在重新发现已知信息。Loop 工程解决的是第三层问题：**谁来驱动下一步**。

本文不停在「是什么」，而是把重点全部压在一件事上：**让你亲手跑通第一个 Loop**。

## 先别建：4 条件测试

诚实的版本是：**Loop Engineering 是真的，但多数开发者还不需要它。** Loop 有成本——它重读上下文、重试、探索，无论这一轮有没有产出都在烧 token。建任何 Loop 之前，先跑这个测试。**四条缺一，Loop 的成本就高于回报：**

1. **任务重复**（至少每周一次）。Loop 把搭建成本摊到很多次运行上。一次性的活，一个好提示词更快更省——不每周复发，那不是 Loop，是「你跑过一次的脚本」。
2. **验证自动化**。Loop 需要一个能在你不在场时判工作失败的东西：测试套件、类型检查、linter、build。没有自动关卡，你就得回到椅子上读每个 diff——那正是 Loop 本该替你去掉的活。
3. **token 预算扛得住浪费**。这项技术随预算缩放：token 近乎免费的人觉得它显然，按量计费的人觉得它鲁莽。
4. **Agent 有资深工程师的工具**。日志、可复现环境、能跑自己写的代码看哪里崩。否则 Loop 在盲目迭代。

### 适合与不适合的场景

| 适合交给 Loop | 需要人坐在椅子上 |
|--------------|-----------------|
| CI 故障分类 | 架构重写 |
| 依赖升级 PR | Auth / 支付代码 |
| lint 修复 | 生产部署 |
| flaky test 复现 | 模糊的产品工作 |
| 强测试覆盖下的 issue→PR | 「完成」靠主观判断的任务 |

本文选的「每日 CI 故障分类」恰好四条全过：每天发生、CI 本身就是自动关卡、只读因此几乎不烧 token、Agent 能读到 CI 日志。下面就拿它动手。

## What — 一个 Loop 由什么构成

### 内循环 vs 外循环

每个 Agent 内部已经有一个「内循环」：读文件 → 改代码 → 跑测试 → 读错误 → 再改。这部分 Agent 自己会做。Loop Engineering 工作在它的**上一层**——你设计的「外循环」：

| 层级 | 谁在驱动 | 做什么 |
|------|---------|--------|
| 内循环（Agent 内置） | Agent 自身 | 感知 → 推理 → 行动 → 观察 |
| 外循环（你来设计） | 你设计的系统 | 发现任务 → 分派 Agent → 验证结果 → 记录状态 → 开下一轮 |

你不再坐在 Agent 旁边为每一步打指令，而是设计一套外部系统替你驾驶内循环。

### 五阶段核心循环

一个 Loop 的骨架是五个首尾相连的阶段。它的力量不在任何单步，而在**闭环**——测试失败不是终点而是新上下文，类型错误不是阻断而是「假设错了」的信号。

```
┌─────────┐
│  Intent  │ ← 定义成功长什么样
└────┬─────┘
     ▼
┌─────────┐
│ Context  │ ← 收集代码、文档、报错
└────┬─────┘
     ▼
┌─────────┐
│  Action  │ ← 编辑文件、运行命令
└────┬─────┘
     ▼
┌──────────────┐
│ Observation  │ ← 取测试结果、Diff
└────┬─────────┘
     ▼
┌────────────┐
│ Adjustment │ ← 更新计划，决定是否重复
└────┬───────┘
     │
     └──── 未完成？回到 Intent ────┘
```

| 阶段 | 做什么 |
|------|--------|
| 意图 Intent | 定义成功长什么样、约束是什么 |
| 上下文 Context | 收集代码、文档、报错、规范 |
| 行动 Action | 编辑文件、运行命令、调用工具 |
| 观察 Observation | 取测试结果、编译错误、运行输出、Diff |
| 调整 Adjustment | 据观察更新计划，重复直到完成或受阻 |

## How — 六大要素如何协作

一个能真正独立运行的 Loop 需要五个核心组件，加一处贯穿始终的记忆。理解它们如何拼在一起，比单独记住每一个更重要。

1. **自动触发器（Automations）** —— Loop 的心跳。没有它，Loop 只是「你手动跑了一次的操作」。它定义「**什么时候**、**做什么**」。
2. **并行隔离（Worktrees）** —— 多个 Agent 同时改同一个文件就像两个人往同一行代码提交，结果是灾难。Git Worktree 给每个 Agent 独立的工作目录和分支，共享同一段 Git 历史但改动互不干扰。
3. **技能文件（Skills）** —— 一个含 `SKILL.md` 的文件夹，写明项目规范、构建命令、历史教训。Agent 每次会话加载它，而不是每次从零猜测你的项目长什么样。
4. **连接器（Plugins / Connectors，基于 MCP）** —— 让 Agent 不只看见本地文件，还能读 Issue、查库、调 API、发 Slack。这是「Agent 说『这是修复方案』」与「Loop 自动开 PR、关联工单、CI 绿了通知频道」之间的差别。
5. **子 Agent（Sub-Agents）** —— 把**写代码的**和**检查代码的**分开（制作者-检查者模式）。写代码的模型给自己打分太宽容；一个指令不同、有时模型也更强的检查者，能抓住第一个 Agent 自圆其说放过的问题。
6. **持久记忆（Memory）** —— 模型每次对话之间会彻底遗忘。解药简单到像作弊：**把状态写进仓库里的文件**（如 `TODO.md`）。Agent 会忘，仓库不会。

### 六要素的协作流程

把它们串起来看一个完整的运行周期：

```
触发器（Cron/事件）
    │ 唤醒 Loop
    ▼
技能文件 ← Agent 加载项目规范
    │
    ▼
连接器 ← 读 CI 日志 / GitHub Issue / Slack
    │
    ▼
子 Agent A（制作者）← 在隔离 Worktree 中执行
    │ 产出 diff / 状态更新
    ▼
子 Agent B（检查者）← 对照规范和测试审查
    │ 通过 / 打回
    ▼
记忆文件 ← 更新 TODO.md / 开 PR / 通知
```

一个**触发器**按计划唤醒 Loop，调用一个**技能**去读昨天的 CI 失败与开放 Issue，把发现写进**记忆文件**；对每条值得做的，开一个隔离的 **worktree** 派**子 Agent** 起草，另一个子 Agent 对照技能规范和测试做审查；**连接器**负责开 PR、更新工单。你只设计了一次，没有提示其中任何一步。

下面我们从最小、最安全的版本开始落地——一个技能 + 一个触发器 + 一个状态文件，再借 CI 当现成的 gate。

## Demo — 搭一个「每日 CI 故障分类」Loop

我们要搭的是 Addy Osmani 推荐的**第一个 Loop**：只读操作（读 CI 日志、读 Issue），只写一个 `TODO.md`，**不开 PR、不改任何源码**。风险几乎为零，却已经很有用——每天早上你打开 `TODO.md` 就知道今天该做什么，而不用花半小时翻 CI 日志。

通过了上面的 4 条件测试后，别上来就搞多 Agent swarm。**最小可用 Loop 只有四件套**——下面五个步骤就是在拼这四件：

- **一个 Automation**（步骤 3）——按节奏触发、按明确条件停止。
- **一个 Skill**（步骤 1）——一个 `SKILL.md`，存项目上下文，免每轮从零重推。
- **一个 State File**（步骤 2）——记录已完成/待办，让明天的运行续上而非重启。
- **一个 Gate**——能自动判坏的测试/build。本 Demo 里 CI 本身就是 gate；**这一件决定 Loop 是帮你还是只烧钱**（详见 Reflection 的失败模式）。

**顺序很重要**：先让一次手动运行可靠 → 把它写成 Skill → 包成 Loop → 再上调度。跳步是 Loop 在生产里翻车的方式。

我们用 Claude Code 实现主线，并在最后给出 OpenAI Codex 的等价做法。

### 步骤 0 — 环境准备

确认 Claude Code 版本满足要求（`/loop` 需 v2.1.72+）：

```bash
claude --version
```

在你的代码仓库根目录启动一个会话：

```bash
cd your-project
claude
```

> 本 Demo 的 Loop 在会话保持打开期间运行（`/loop` 是 session 内的轮询）。要在关掉终端后仍持续运行，见步骤 4 的「云端 / Desktop 计划任务」。

### 步骤 1 — 用 Skill 固化「分类」这件事

与其每天把一大段指令粘进调度里（没人会去维护它），不如把「分类」写成一个技能文件，让触发器只 `$skill-name` 调用它。在仓库里创建：

```markdown
<!-- 文件路径：.claude/skills/ci-triage/SKILL.md -->
---
name: ci-triage
description: 读取昨日 CI 失败与开放 Issue，分类后写入 TODO.md。只读分析，不修改源码、不开 PR。
---

## 任务
1. 读取昨天的 CI 失败日志和标记为 'bug' 的开放 Issue。
2. 按可能的根因给发现分类（如：flaky test / 依赖问题 / 真实回归）。
3. 把带优先级的行动清单写进仓库根目录的 TODO.md。

## 硬约束
- 只读：不得编辑任何源文件。
- 不得开 PR、不得 push、不得合并。
- TODO.md 用「进行中 / 待处理 / 已完成」三段式，每条注明 CI Run 编号或 Issue 链接。

## 分类规则
- flaky test：同一测试在最近 3 次运行中时过时不过
- 依赖问题：错误信息包含版本冲突、下载失败、lock file 不一致
- 真实回归：新提交引入的确定性失败，且在该提交前 CI 是绿的
- 环境问题：CI runner 资源耗尽、超时、网络不通

## 优先级
- P0：阻塞主干合并的失败
- P1：影响多个开发者的 flaky test
- P2：可以下个迭代处理的非阻塞问题
```

`description` 里那句「只读分析，不修改源码」很关键——它既是给人看的边界，也让 Agent 在匹配任务时知道该加载它。分类规则和优先级的明确定义让 Agent 不用猜，输出质量更稳定。

### 步骤 2 — 用一个明确的状态文件作为记忆

Loop 的「记忆」就是仓库里一个随代码提交的 Markdown 文件。先手动建一个骨架，让 Agent 知道往哪写：

```markdown
<!-- 文件路径：TODO.md（Loop 的状态文件，随代码一起提交） -->
# Loop 任务状态

最后更新：（由自动 Loop 填写）

## 进行中

## 待处理

## 已完成
```

为什么是文件而不是靠对话记忆？因为模型在两次运行之间会遗忘。把状态放在仓库里，明天的运行才能接上今天的进度。

#### Loop 运行后的实际产出

第一次运行完成后，`TODO.md` 大约长这样：

```markdown
# Loop 任务状态

最后更新：2026-06-22 09:03 (by ci-triage loop)

## 进行中

## 待处理

### P0 - 阻塞主干
- [ ] **真实回归**：`test_auth_flow` 在 commit `a3f21c` 后 100% 失败
  - CI Run: [#4521](https://github.com/your-org/repo/actions/runs/4521)
  - 错误：`AssertionError: expected 200, got 401`
  - 可能原因：该 commit 改动了 `middleware/auth.py` 的 token 校验逻辑

### P1 - 影响多人
- [ ] **flaky test**：`test_payment_webhook` 最近 5 次运行 3 次通过 2 次超时
  - CI Runs: #4519, #4520, #4521
  - 模式：只在并行执行时超时，疑似资源竞争
- [ ] **依赖问题**：`npm install` 在 CI 上间歇性报 `ERESOLVE` 冲突
  - 涉及：`@types/react@19.2.1` vs `react@18.3.0`
  - Issue: [#287](https://github.com/your-org/repo/issues/287)

### P2 - 下个迭代
- [ ] **环境问题**：macOS runner 磁盘空间告警 (剩余 < 5GB)
  - CI Run: #4518 (warning, 未阻塞)

## 已完成
- [x] ~~flaky test `test_db_connection`~~：已由 @zhangsan 在 PR #312 修复 (2026-06-21)
```

你早上打开它，3 秒就知道今天最该做什么——而不是花 30 分钟逐条翻 CI 日志。

### 步骤 3 — 用 /loop 定时触发（最小安全版）

现在装上心跳。在 Claude Code 会话里直接输入 `/loop`，**间隔在前、提示在后**（这是官方语法，注意不是 `--schedule` 旗标）：

```text
/loop 1d 调用 $ci-triage：读取昨天的 CI 失败和标记 bug 的 Issue，分类后写入 TODO.md。不要编辑任何源文件，不要开 PR。
```

- `1d` 是间隔，支持的单位是 `s`（秒）、`m`（分）、`h`（时）、`d`（天）。Claude 会把它转成 cron 表达式、确认节奏并返回一个任务 ID。
- 间隔也可以放句尾，比如 `... every 2 hours`。
- 想先观察行为，可以把间隔缩短到 `/loop 5m ...` 跑几轮看效果。

如果你想让 Claude 自己决定节奏（PR 活跃时查得勤、安静时查得疏），省掉间隔即可：

```text
/loop 检查 CI 是否通过并把失败项分类写进 TODO.md，只读不改源码
```

**管理你的 Loop：**

```text
/loop status          # 查看当前活跃的 Loop
/loop stop <task-id>  # 按 ID 停止特定 Loop
Esc                   # 在 Loop 等待下一轮时按 Esc 清除
```

固定间隔的 Loop 会一直跑到你停止它，或七天后自动过期。

> ⚠️ **Token 成本**：带验证子 Agent 的定时 Loop 每次触发都消耗 Token，且随任务复杂度剧烈波动。先用较慢节奏（每天一次）观察几天成本，再考虑加快。Addy Osmani 本人也反复强调这一点。

### 步骤 4 —（可选）让它在你关掉终端后也能跑

`/loop` 只在会话开着时轮询。如果要无人值守，Claude Code 提供另外两种计划任务（按官方文档对比）：

| 方式 | 运行在 | 最小间隔 | 适合 |
|------|--------|---------|------|
| `/loop` | 你的机器（会话内） | 1 分钟 | 会话期间的快速轮询 |
| Desktop 计划任务 | 你的机器 | 1 分钟 | 需要访问本地文件和工具 |
| 云端任务（Cloud） | Anthropic 云 | 1 小时 | 关机也要可靠运行 |

对「每日 CI 分类」这种轻量只读任务，Desktop 计划任务或云端任务都合适。你也可以直接用自然语言让 Claude 管理：

```text
帮我每个工作日早上 9 点运行 $ci-triage，结果写进 TODO.md
```

底层 Claude 会用 `CronCreate` 建任务（5 字段 cron 表达式），用 `CronList` 列出、`CronDelete` 按 ID 取消。注意 cron 时间按你的**本地时区**解释——`0 9 * * 1-5` 就是当地工作日 9 点。

### 步骤 5 — 在 OpenAI Codex 里的等价做法

如果你用的是 Codex，能力是一一对应的，只是入口不同。在 Codex app 的 **Automations** 面板新建一个自动化：填项目、提示（同样用 `$ci-triage` 触发技能）、schedule（需要自定义节奏时填 cron），以及在本地检出还是后台 worktree 中运行。

关键的安全开关在沙箱设置：把自动化设为**只读模式（read-only）**，任何尝试改文件、联网的工具调用都会失败——这正好匹配我们「只读 + 只写 TODO.md」的目标（写 `TODO.md` 属项目内写入，按需在 rules 里 allowlist）。有发现的运行会进 **Triage 收件箱**，没发现的自动归档。

## 进阶 — 从阶段 1 到阶段 2

当你的只读 Loop 稳定跑了一两周、你信任它的分类质量后，自然会想让它做更多——从"告诉我问题在哪"升级为"顺便起草一个修复"。这是阶段 2：**草稿模式**。

### 阶段 2 的 Skill 文件

```markdown
<!-- 文件路径：.claude/skills/ci-fix-draft/SKILL.md -->
---
name: ci-fix-draft
description: 对 TODO.md 中 P0 级别的真实回归，在隔离分支起草修复并跑测试。不 push、不开 PR。
---

## 任务
1. 读取 TODO.md 中标记为「P0 - 真实回归」的条目。
2. 对每条 P0，创建一个隔离的 git worktree 分支。
3. 在该分支中尝试修复，运行相关测试。
4. 测试通过 → 在 TODO.md 中标注「草稿就绪：branch-name」。
5. 测试失败 → 在 TODO.md 中标注「修复尝试失败：原因」，不再重试。

## 硬约束
- 只在 worktree 分支中操作，不动 main。
- 不 push、不开 PR、不合并。
- 每条 P0 最多尝试修复 3 次，超过就标注失败并停止。
- 修改范围仅限报错堆栈直接涉及的文件，不做「顺手重构」。
```

### 阶段 2 的触发方式

```text
/loop 1d 先运行 $ci-triage 更新 TODO.md，然后对 P0 项运行 $ci-fix-draft
```

### 关键变化

| 阶段 1 | 阶段 2 新增 |
|--------|-----------|
| 只读 CI 日志 | 在 worktree 中改代码 |
| 只写 TODO.md | 写 TODO.md + 创建分支 |
| 零风险 | 风险受限于隔离分支（不 push） |
| 无需 gate | **测试套件**是 gate |
| 你手动修 | 你审查 diff 后决定是否采纳 |

注意阶段 2 的**新前提**：你的测试套件必须足够好，能在 Agent 修错的时候把它挡住。如果测试覆盖率低于 60%，先补测试再上阶段 2。

## Reflection — 何时让 Loop 承担更多，何时不

我们刚搭的是阶段 1。Addy Osmani 给出的自主度阶梯是这样的，**逐级提升、别跳级**：

| 阶段 | Loop 能做 | 你做什么 |
|------|----------|---------|
| 1 只读 | 发现问题、分类、写状态文件 | 看 `TODO.md`，手动决定顺序 |
| 2 草稿 | 起草修复、跑测试、写入分支 | 审查 diff，手动 push |
| 3 半自动 | 开 Draft PR、跑 CI、通知 Slack | 审查 PR，手动 Merge |
| 4 全自动 | 制作者 + 检查者双 Agent，CI 绿后自动合并 | 异常介入，定期审计合并历史 |

往上爬时，前面讲的六要素会逐个登场：阶段 2 起需要 **Worktrees** 隔离并行改动；阶段 3 起需要 **连接器** 开 PR、发通知；阶段 4 必须上 **子 Agent** 做制作者-检查者分离——尤其是用 Claude Code 的 `/goal` 让 Loop 跑到一个**可验证条件**成立，比如：

```text
/goal test/auth 下所有测试通过且 lint 干净
```

`/goal` 设好后立即开跑，每一轮结束由一个**独立的小模型**判断条件是否成立——写代码的 Agent 不是给它打分的那个。`/goal clear` 提前停止；`claude -p "/goal ..."` 可在非交互模式下一次跑到底。

### 失败模式：Ralph Wiggum Loop

这里藏着一个最常见的翻车点：**Ralph Wiggum loop**（工程师 Geoffrey Huntley 命名）——Agent 在半成品上过早宣布「完成」，Loop 没有硬关卡就一直空转烧钱。

根因三件套：

1. **没有真实验证器** — 只让第二个 Agent「review」却无客观信号，等于两个乐观主义者互相点头
2. **软完成条件** — 「done」由 Agent 判断而非测试/build
3. **没有硬停止** — 超时、重试上限都没设

修复就是那个 gate——**一个客观的、能判工作失败的信号**：测试过/不过、build 编译/不编译、linter 返回 0/非 0，而不是「有意见」的验证器。这也是为什么我们的第一个 Loop 直接复用 CI 当 gate。

### Loop 变强后更尖锐的三个问题

Loop 变好不等于你的责任变轻。有三个问题会随 Loop 变强而**更尖锐**，不是更轻松：

- **验证仍是你的责任**。无人值守运行的 Loop，也是无人值守地制造错误的 Loop。检查者子 Agent 能降低风险，但「通过了验证」是一个声明，不是证明——合并代码的人工审查不能消失。
- **理解债（Comprehension Debt）积累更快**。Loop 产出代码越快，你真正理解的比例越低。唯一解药是：读 Loop 产出的代码。
- **认知投降（Cognitive Surrender）**。Loop 自动运转时，接受它返回的任何东西最舒适——这是最隐性的危险。Addy Osmani 那句话值得贴在屏幕上：「两个人可以构建完全相同的 Loop，却得到截然相反的结果。一个用它在深度理解的基础上更快推进，另一个用它来回避理解本身。Loop 不知道区别。你知道。」

## 故障排查

初次搭建 Loop 时最常遇到的问题：

| 症状 | 原因 | 修复 |
|------|------|------|
| Loop 没有触发 | Claude Code 版本低于 v2.1.72 | `claude update` 升级 |
| Loop 触发了但 TODO.md 没更新 | Skill 文件路径错误或 `$skill-name` 拼写不匹配 | 检查 `.claude/skills/` 目录结构 |
| 分类结果全是 P2，明显有 P0 被漏掉 | Skill 中的分类规则太模糊 | 在 SKILL.md 中给出具体的判断条件和示例 |
| Token 消耗异常高 | Agent 在读大量 CI 日志时上下文爆炸 | 在 Skill 中限制「只读最近 24h 内失败的 run，最多 10 条」 |
| Loop 写了源码（违反只读约束） | Skill 中的硬约束不够显眼 | 把「不得编辑源文件」放在 Skill 最顶部；在 `/loop` 提示中再重复一次 |
| 云端任务没有按时运行 | Cron 时区问题 | 确认是你的本地时区还是 UTC；用 `CronList` 检查实际 schedule |

### 调试技巧

1. **先手动跑一次**。在 Claude Code 会话中直接说"运行 $ci-triage"，看它做了什么、写了什么。确认输出合理后再加 `/loop`。
2. **缩短间隔观察**。用 `/loop 5m ...` 观察 3-4 轮，确认行为稳定后再改回 `1d`。
3. **看 Loop 的内部日志**。Claude Code 在每轮运行后会输出摘要，留意它读了哪些文件、调用了哪些工具。
4. **设定硬停止**。在 Skill 中写明「如果连续 3 轮 TODO.md 无变化，停止并通知」。

## 快速参考

### Claude Code 命令速查

```bash
# 版本检查
claude --version

# 启动 Loop（会话内轮询）
/loop 1d <提示>              # 固定间隔
/loop <提示>                  # Agent 自决节奏
/loop status                  # 查看活跃 Loop
/loop stop <id>               # 停止 Loop

# Goal（跑到条件成立）
/goal <可验证条件>            # 启动
/goal clear                   # 取消

# 非交互模式
claude -p "/goal 所有测试通过"  # 一次跑到底

# 计划任务管理（底层命令）
CronCreate                    # 创建 cron 任务
CronList                      # 列出所有任务
CronDelete <id>               # 删除任务
```

### Codex 等价操作

| Claude Code | OpenAI Codex |
|-------------|-------------|
| `/loop` | Automations 面板 → New |
| Skill 文件 (`.claude/skills/`) | Project Instructions / AGENTS.md |
| `/goal` | Automations + 自定义 exit condition |
| Worktree（自动） | 沙箱环境（自动隔离） |
| MCP 连接器 | 内置 GitHub/Linear 集成 |

### 搭建检查清单

- [ ] 跑过 4 条件测试，确认任务适合 Loop
- [ ] 创建 `.claude/skills/ci-triage/SKILL.md`，含分类规则和硬约束
- [ ] 创建 `TODO.md` 骨架文件
- [ ] 手动运行一次 `$ci-triage`，确认输出合理
- [ ] 用 `/loop 5m` 观察 3-4 轮行为
- [ ] 确认稳定后改为 `/loop 1d` 或设置云端 cron
- [ ] 跑一周后回顾：分类准确率、token 消耗、遗漏率

---

**所以：先搭好这个零风险的只读 Loop，让它每天替你整理战场；理解了它的行为，再一阶一阶往上交权。** 设计 Loop，但要像一个打算继续当工程师的人那样设计它——而不是只负责按下启动键的人。
