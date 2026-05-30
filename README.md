# Sonata Tutorial

提问驱动的 Sonata 架构教程。适合初次接触项目的开发者和 agent 快速建立全局认知。

---

## 目录

### 第一层：入门（00~07）

快速建立全局认知。适合初次接触项目的开发者和 agent。

| 文件 | 主题 | 前置阅读 |
|------|------|----------|
| `00-overview.html` | Sonata 是什么？要解决什么问题？ | — |
| `01-core-model.html` | Score / Task / Dependency / ShapeAssumption | 00 |
| `02-eligibility.html` | check_static_eligibility 如何判断谁能进入 Sonata | 01 |
| `03-dependencies.html` | Sequential vs Dataflow 两种依赖策略 | 01 |
| `04-serialization.html` | JSON 序列化、fingerprint、schema version | 01 |
| `05-storage-audit.html` | Storage key 内存别名追踪、审计元数据 | 01 |
| `06-fallback.html` | FallbackCode 枚举、消息到 code 的映射、解耦架构、工业界经验 | 02 |
| `07-running-tests.html` | 测试结构、mock IR、运行命令、当前限制、v0.1 进度 | — |

### 第二层：深入（deep-*）

逐模块、逐函数的完整解析。覆盖每个边界条件、每个设计决策的取舍理由和切换条件。
适合需要修改代码或评审设计的开发者和 agent。

| 文件 | 主题 | 对应源文件 | 前置阅读 |
|------|------|-----------|----------|
| `deep-01-score-model.html` | score.py 完整解析 | `src/sonata/score.py` | 01 |
| `deep-02-eligibility.html` | eligibility.py 完整解析 | `src/sonata/eligibility.py` | 02, deep-01 |
| `deep-03-dependencies.html` | dependencies.py 完整解析 | `src/sonata/dependencies.py` | 03, deep-01 |
| `deep-04-serialization.html` | serialization.py 完整解析 | `src/sonata/serialization.py` | 04, deep-01 |
| `deep-05-storage.html` | storage.py 完整解析 | `src/sonata/storage.py` | 05, deep-01 |
| `deep-06-audit.html` | audit.py 完整解析 | `src/sonata/audit.py` | 05, deep-05 |
| `deep-07-fallback.html` | fallback.py 完整解析 | `src/sonata/fallback.py` | 06, deep-01 |
| `deep-08-design-decisions.html` | 设计决策全景 | `reports/roadmap_history.md`, `roadmap_undecided.md` | deep-01~07 |

第二层文件使用 `deep-` 前缀命名，与第一层区分。

---

## 子目录

```
tutorial/
├── README.md              ← 本文件（维护规范，供 human 和 agent 阅读）
├── 00~07 *.html           ← 第一层：入门（提问驱动，快速建立全局认知）
├── deep-*  *.html         ← 第二层：深入（逐模块、逐函数完整解析）
└── temp/                  ← 临时草稿区，用于信息收集和整理
```

**`temp/` 的用途**：
- 在更新 tutorial 过程中存放临时笔记、草稿、待归档的信息
- 内容稳定后整理进对应的主题文件，temp 中的原始草稿可以删除
- 不代表最终结论，随时可以被覆盖或清理

---

## Agent 注意事项：格式规范

**教程文件必须使用 HTML，不允许使用 Markdown。**

理由：HTML 在浏览器里有更好的排版控制（表格、代码块、导航链接），
且能独立于任何 Markdown 渲染器运行。用浏览器直接打开就能阅读。

创建或更新教程文件时遵循以下规范：

1. **文件扩展名**：`.html`，不是 `.md`
2. **编码**：UTF-8，`<html lang="zh-CN">`
3. **样式**：内嵌 `<style>` 块，使用以下统一基础样式（复制现有文件即可）：
   - `body`：`max-width: 760px`，`margin: 2.5em auto`，系统字体
   - 代码块：浅灰背景 `<pre><code>`，行内 `<code>`
   - 表格：`border-collapse`，灰白表头
   - 导航：底部 `<div class="nav">`，链接前后篇
4. **导航链接**：每个文件底部放上一篇/下一篇链接，形成串联阅读路径
5. **内容结构**：第一层以"提问："为二级标题驱动，每节回答一个问题；第二层以模块/函数为单位，逐行解析代码和设计决策
6. **代码块**：用 `<pre><code>` 包裹，不要用 Markdown 反引号围栏

---

## 维护责任

**本文件随版本更新而更新，不是一次性产物。**

### Human 维护义务

- 每次 milestone（v0.1 → v0.2 → ...）完成后，检查 tutorial 内容是否与当前代码一致
- 新增重要模块（如 v0.2 的 PlanHandle、v0.3 的 alias 分析）时，及时增补对应主题文件
- 删除已废弃的内容，避免教程与代码出现分歧

### Agent 维护义务

- 开发过程中如果发现 tutorial 描述与实际代码不一致，**主动提出**或直接修正
- 修正时必须保持 HTML 格式，使用 `temp/` 暂存草稿再归档
- 新建教程文件时，从现有文件复制样式块和导航结构，保持风格统一
- 在 `temp/` 中记录版本更新期间的变更笔记，供后续整理
- PR 描述中注明是否需要同步更新 tutorial

### 共同原则

- Tutorial 描述的是**当前版本的行为**，不是历史变迁记录
- 不要在教程正文里记录"从 vX 到 vY 的变化"——那是 changelog 的工作
- `temp/` 中可以存放版本迁移期间的过渡笔记，但最终应归档或删除

---

## 版本同步记录

| 版本 | tutorial 最后同步时间 | 备注 |
|------|----------------------|------|
| v0.0 | 2026-05-30 | 初始创建，基于 v0.0 基线代码 |
