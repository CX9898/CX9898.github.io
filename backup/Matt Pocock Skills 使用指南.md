# Matt Pocock Skills 使用指南

本文档整理 `https://github.com/mattpocock/skills` 项目下的 Skill，分类按“用途”组织，说明它们分别解决什么问题，以及什么时候应该使用。

更新时间：2026-06-08；当前统计：29 个 Skill。

## 如何选择 Skill

- 有 bug、异常行为或性能回归：优先用 [`diagnose`](#diagnose)。
- 用户要求测试先行：用 [`tdd`](#tdd)。
- 设计、状态机、数据模型或 UI 方案还没定：用 [`prototype`](#prototype)。
- 想追问并打磨计划：用 [`grill-me`](#grill-me)。
- 想把计划对齐项目术语和 ADR：用 [`grill-with-docs`](#grill-with-docs)。
- 想理解一片陌生代码：用 [`zoom-out`](#zoom-out)。
- 想改善架构或寻找重构机会：用 [`improve-codebase-architecture`](#improve-codebase-architecture)。
- 想把需求沉淀成 PRD：用 [`to-prd`](#to-prd)。
- 想把 PRD 或计划拆成任务：用 [`to-issues`](#to-issues)。
- 想处理 issue 队列：用 [`triage`](#triage)。
- 想审查分支、PR 或 WIP diff：用 [`review`](#review)。
- 想写文章：先用 [`writing-fragments`](#writing-fragments) 收集素材，再用 [`writing-shape`](#writing-shape) 或 [`writing-beats`](#writing-beats) 成文；已有草稿则用 [`edit-article`](#edit-article)。
- 想创建新的 Skill：用 [`write-a-skill`](#write-a-skill)。

## 分类速查

### 工程类

| Skill | 作用 | 适合场景 |
| --- | --- | --- |
| [`diagnose`](#diagnose) | 系统化排查 bug 和性能回归 | 报错、行为异常、性能变慢、非确定性问题 |
| [`tdd`](#tdd) | 按红-绿-重构循环开发 | 用户要求测试先行，或变更风险较高 |
| [`prototype`](#prototype) | 构建可丢弃原型验证设计 | 状态机、数据模型、业务规则或 UI 方案未定 |
| [`improve-codebase-architecture`](#improve-codebase-architecture) | 发现架构深化和重构机会 | 代码难测、耦合重、模块边界混乱 |
| [`grill-with-docs`](#grill-with-docs) | 结合领域文档追问设计并沉淀术语/决策 | 计划需要对齐 `CONTEXT.md` 或 ADR |
| [`zoom-out`](#zoom-out) | 从更高层解释代码结构 | 不熟悉某块代码，需要模块地图和调用关系 |
| [`to-prd`](#to-prd) | 把当前上下文整理成 PRD | 需求已清楚，需要沉淀产品需求文档 |
| [`to-issues`](#to-issues) | 把计划或 PRD 拆成垂直切片 issue | 大任务要拆给人或 agent 并行实现 |
| [`triage`](#triage) | 按状态机处理 issue | 分流 bug/需求，标记 ready-for-agent 等状态 |
| [`setup-matt-pocock-skills`](#setup-matt-pocock-skills) | 配置工程 Skill 所需项目上下文 | 首次在 repo 中使用工程类 Skill |

### 生产力类

| Skill | 作用 | 适合场景 |
| --- | --- | --- |
| [`caveman`](#caveman) | 极简压缩沟通 | 用户要求简短、少 token、caveman mode |
| [`grill-me`](#grill-me) | 追问并打磨计划或设计 | 想 stress-test 方案，但不需要更新项目文档 |
| [`handoff`](#handoff) | 生成会话交接文档 | 换 agent、换会话、暂停后继续 |
| [`write-a-skill`](#write-a-skill) | 创建新的 agent Skill | 想把重复工作流沉淀成 `SKILL.md` |

### 写作与知识管理类

| Skill | 状态 | 作用 | 适合场景 |
| --- | --- | --- | --- |
| [`edit-article`](#edit-article) | 稳定 | 编辑和改进文章草稿 | 已有文章，需要重排结构和润色 |
| [`writing-fragments`](#writing-fragments) | 进行中 | 追问并收集写作碎片 | 想先积累素材，不急着定结构 |
| [`writing-shape`](#writing-shape) | 进行中 | 把原始材料塑造成完整文章 | 有 notes/fragments，想形成论证型文章 |
| [`writing-beats`](#writing-beats) | 进行中 | 按 beat 逐段组装叙事文章 | 想写成 journey，而不是传统大纲 |
| [`obsidian-vault`](#obsidian-vault) | 稳定 | 管理 Obsidian 笔记和 wikilinks | 查找、创建、整理个人知识库笔记 |

### 杂项工具类

| Skill | 作用 | 适合场景 |
| --- | --- | --- |
| [`setup-pre-commit`](#setup-pre-commit) | 配置 Husky、lint-staged、Prettier、typecheck/test | JS/TS 项目需要提交前质量门禁 |
| [`scaffold-exercises`](#scaffold-exercises) | 创建课程练习目录结构 | 搭建 section、exercise、problem、solution、explainer |
| [`migrate-to-shoehorn`](#migrate-to-shoehorn) | 将测试中的 `as` 断言迁移到 `@total-typescript/shoehorn` | 测试数据需要 partial object 或故意错误类型 |
| [`git-guardrails-claude-code`](#git-guardrails-claude-code) | 阻止 Claude Code 执行危险 git 命令 | 防止误 push、reset、clean、删除分支 |

### 教学与审查类

| Skill | 状态 | 作用 | 适合场景 |
| --- | --- | --- | --- |
| [`teach`](#teach) | 进行中 | 使用 lesson、reference、learning records 做长期教学 | 用户想持续学习一个主题 |
| [`review`](#review) | 进行中 | 双轴 review：Standards 与 Spec | 审查分支、PR 或 WIP diff |

### 废弃 Skill

| Skill | 替代方案 | 说明 |
| --- | --- | --- |
| [`ubiquitous-language`](#ubiquitous-language) | [`grill-with-docs`](#grill-with-docs) | 旧版 DDD 术语提取流程 |
| [`request-refactor-plan`](#request-refactor-plan) | [`improve-codebase-architecture`](#improve-codebase-architecture) + [`to-issues`](#to-issues) | 旧版重构计划访谈流程 |
| [`qa`](#qa) | [`triage`](#triage) 或 [`diagnose`](#diagnose) | 旧版 QA 到 issue 流程 |
| [`design-an-interface`](#design-an-interface) | [`prototype`](#prototype) 或 [`grill-with-docs`](#grill-with-docs) | 旧版并行接口设计流程 |

## 工程类详解

### diagnose

用于系统化排查 bug、异常行为和性能回归。核心是先建立可重复运行的反馈环，再复现、提出假设、加仪器验证、修复并补回归测试。

适合在用户说“debug this”“diagnose this”，或报告功能报错、输出错误、性能变慢、行为不稳定时使用。

不要凭直觉直接改代码。这个 Skill 要求先得到一个可信的 pass/fail 信号，再围绕信号定位根因。

### tdd

用于按红-绿-重构循环开发。测试应验证公开接口的行为，而不是内部实现细节。

适合在用户明确要求 TDD、test-first、red-green-refactor，或变更风险较高、需要先锁定外部行为时使用。

它强调用一个垂直切片推进：先写失败测试，再写最小实现，最后重构。

### prototype

用于构建可丢弃原型，帮助验证状态机、业务规则、数据模型或 UI 方案。

如果问题偏业务逻辑，可以做可运行的终端原型。如果问题偏 UI，可以做多个可切换的界面变体。

适合在设计还没定、用户说“prototype this”“let me play with it”或“try a few designs”时使用。

### improve-codebase-architecture

用于从领域语言和 ADR 出发，寻找代码库中的架构深化机会。

它重点关注模块是否太浅、耦合是否泄漏、测试边界是否困难，以及复杂度是否可以收进更深的模块接口。

适合在用户想改善架构、找重构机会，或觉得代码难测、难理解、边界混乱时使用。

### grill-with-docs

用于结合现有领域文档和 ADR，逐步追问用户的计划或设计。

当术语、边界或决策变清楚时，它会同步更新 `CONTEXT.md` 或 ADR，让讨论结果沉淀进项目文档。

适合在计划需要对齐项目已有语言、存在模糊术语，或需要确认设计是否违反已有 ADR 时使用。

### zoom-out

用于让 agent 跳到更高层抽象，解释某片代码在整体系统中的位置、相关模块、调用方和职责边界。

适合在用户不熟悉某个代码区域，或需要先理解大图而不是直接钻进某个函数时使用。

它通常不负责改代码，而是帮助建立模块地图。

### to-prd

用于把当前对话和代码库理解整理成 PRD，并发布到项目 issue tracker。

它不重新访谈用户，而是综合已有上下文，输出 problem statement、solution、user stories、implementation decisions 和 testing decisions。

适合在需求已经聊得比较清楚，需要沉淀成产品需求文档时使用。

### to-issues

用于把计划、spec 或 PRD 拆成可独立领取的 issue。

它强调端到端垂直切片，而不是按前端、后端、数据库这类层级水平拆分。

适合在大功能需要多人或多个 agent 并行实现，或需要区分 HITL 与 AFK 切片时使用。

### triage

用于按 issue triage 状态机处理问题。

它会区分 `bug` / `enhancement`，以及 `needs-triage`、`needs-info`、`ready-for-agent`、`ready-for-human`、`wontfix` 等状态。

适合查看哪些 issue 需要关注、分流具体 issue，或为 AFK agent 准备清晰的问题说明。

### setup-matt-pocock-skills

用于为工程类 Skill 搭建项目级配置。

它会确认 issue tracker、triage 标签映射、领域文档布局，并写入 `AGENTS.md` 或 `CLAUDE.md` 及 `docs/agents/`。

适合在首次使用 `to-issues`、`to-prd`、`triage`、`diagnose`、`tdd`、`improve-codebase-architecture` 或 `zoom-out` 前运行。

## 生产力类详解

### caveman

用于进入极简压缩沟通模式。它会删掉寒暄、填充词和不必要解释，同时保留技术准确性。

适合在用户说“caveman mode”“be brief”“less tokens”，或希望减少 token 消耗时使用。

### grill-me

用于猛烈追问用户的计划或设计，逐个拆开决策分支，直到双方达成清晰共识。

适合在用户想 stress-test 一个设计，或计划还没有完全想清楚时使用。

如果需要同步更新领域文档，优先用 `grill-with-docs`。

### handoff

用于把当前会话压缩成一份交接文档，方便另一个 agent 或下一次会话继续工作。

适合在上下文很长、需要换会话，或工作要暂停但希望保留关键状态时使用。

交接内容通常包括进展、决策、下一步和推荐使用的 Skill。

### write-a-skill

用于创建新的 agent Skill，包括 `SKILL.md` 结构、触发条件、引用文件和可选脚本。

适合在用户想创建、编写或改进一个 Skill，或希望把重复任务沉淀成可复用工作流时使用。

## 写作与知识管理类详解

### edit-article

用于编辑和改进已有文章草稿，包括重排结构、增强清晰度、压缩段落和改善行文。

流程是先按 heading 划分文章，确认信息依赖顺序，再逐节改写。

适合在用户已有文章，希望改善结构、表达、段落顺序或可读性时使用。

### writing-fragments

用于通过追问挖掘写作碎片，并持续追加到一个 Markdown 文件中。

碎片可以是观点、句子、例子、抱怨、半成品想法、对话或一组相关观察。

适合在用户还不想定结构，只想先收集素材，或提到 fragments、ideate、raw material 时使用。

### writing-shape

用于把一堆原始材料塑造成文章。

它会先提出多个开头方向，再逐段增长文章，并和用户讨论段落、列表、引用、表格等形式。

适合在用户已有 notes、fragments 或 rough draft，并想把它们变成论证型文章时使用。

### writing-beats

用于把文章当作一段 journey 来写。

用户先从候选起始 beat 中选择一个，然后一段一段决定下一步走向，直到文章自然结束。

适合在用户有原始材料，但想写成叙事型文章，而不是传统大纲或论证结构时使用。

### obsidian-vault

用于搜索、创建和管理 Obsidian vault 中的笔记。

它使用 `[[wikilinks]]` 和 index notes 组织知识，也可以查找反向链接。

适合在用户想查找、创建、整理个人知识库笔记，或维护 Obsidian 索引时使用。

## 杂项工具类详解

### setup-pre-commit

用于为当前 repo 设置 Husky pre-commit hook、lint-staged、Prettier，并可接入 typecheck 和 test。

适合在 JS/TS 项目需要提交前自动格式化、类型检查、测试，或 repo 还没有统一提交前质量门禁时使用。

### scaffold-exercises

用于创建课程练习目录结构，包括 section、exercise、problem、solution、explainer。

适合在用户想 scaffold exercises，或项目中有 `exercises/` 结构和 `pnpm ai-hero-cli internal lint` 校验时使用。

### migrate-to-shoehorn

用于把测试文件中的 TypeScript `as` 类型断言迁移到 `@total-typescript/shoehorn`。

它使用 `fromPartial()`、`fromAny()` 等方法表达测试数据，只适用于测试代码，不用于生产代码。

适合在测试里有大量 `as Type` 或 `as unknown as Type`，并需要构造 partial test data 时使用。

### git-guardrails-claude-code

用于为 Claude Code 配置 hook，阻止危险 git 命令执行。

典型拦截对象包括 `git push`、`git reset --hard`、`git clean -f`、`git branch -D`。

适合在用户想防止 agent 误推送、误删除、误重置，或团队希望增加 git 安全护栏时使用。

## 教学与审查类详解

### teach

用于把当前目录当成长期教学工作区。

它通过 `MISSION.md`、`GLOSSARY.md`、`RESOURCES.md` 和 learning records 追踪学习状态。

适合在用户想持续学习一个主题，或学习需要跨多次会话推进时使用。

### review

用于从指定固定点对当前 diff 做双轴 review：Standards 和 Spec。

Standards 检查代码是否符合项目文档化标准。Spec 检查实现是否符合原始 issue、PRD 或 spec。

适合在用户想 review 当前分支、PR、WIP，或说“review since main / since commit X”时使用。

## 废弃 Skill 详解

这些 Skill 位于 `deprecated` 目录。它们仍可读，但通常不建议优先使用，因为已有更新的 Skill 或组合流程可以覆盖。

### ubiquitous-language

旧版用途是从当前对话提取 DDD 风格的通用语言术语表，并写入 `UBIQUITOUS_LANGUAGE.md`。

现在更推荐使用 `grill-with-docs` 和 `CONTEXT.md` 流程来沉淀领域语言。

### request-refactor-plan

旧版用途是通过访谈生成详细重构计划，并创建 GitHub issue。

现在更推荐先用 `improve-codebase-architecture` 发现候选，再用 `to-prd` 或 `to-issues` 沉淀执行计划。

### qa

旧版用途是把交互式 QA 反馈变成 GitHub issue。

现在可根据情况使用 `triage`、`diagnose` 或 `to-issues` 替代。

### design-an-interface

旧版用途是让多个 sub-agent 并行生成不同接口设计，再比较权衡。

现在可根据问题使用 `prototype`、`grill-with-docs` 或 `improve-codebase-architecture` 承接。
