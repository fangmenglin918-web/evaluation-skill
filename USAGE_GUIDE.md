# 评估Skill使用指南

本文档说明如何在 **Cursor** 和 **Claude Code** 中安装和使用评估skill。

---

## 一、Skill文件结构

```
evaluation-skill/
├── SKILL.md                           # 核心指令文档（元数据层+指令层）
├── references/                        # 参考资源层
│   ├── dimension_design.md            # 维度设计详细指南
│   └── consistency_metrics.md         # 一致性指标详解
├── assets/                            # 模板资源层
│   ├── rubric_template.md             # 评分细则模板
│   ├── prompt_template.md             # 评分Prompt模板
│   └── redline_template.md            # 红线检测Prompt模板
└── scripts/                           # 工具脚本层
    ├── calculate_consistency.py       # 一致性计算脚本
    └── analyze_discrepancy.py         # 差异分析脚本
```

---

## 二、在Cursor中使用

### 方法1：通过 .cursorrules 引用

1. **将skill文件夹放入项目目录**
   ```bash
   cp -r evaluation-skill /your/project/path/skills/
   ```

2. **在项目根目录创建 `.cursorrules` 文件**
   ```markdown
   # 项目规则
   
   ## 评估相关任务
   
   当需要进行评估体系设计、评分标准制定、一致性分析时，请参考以下skill：
   
   - 核心指南：`skills/evaluation-skill/SKILL.md`
   - 维度设计：`skills/evaluation-skill/references/dimension_design.md`
   - 一致性指标：`skills/evaluation-skill/references/consistency_metrics.md`
   
   模板文件位于 `skills/evaluation-skill/assets/` 目录。
   工具脚本位于 `skills/evaluation-skill/scripts/` 目录。
   ```

3. **使用时直接对话**
   - Cursor会自动读取 `.cursorrules` 并了解skill的存在
   - 提问时可以引用具体文件：`@skills/evaluation-skill/SKILL.md`

### 方法2：通过 Docs 功能添加

1. **打开Cursor设置** → **Features** → **Docs**
2. **点击 "Add new doc"**
3. **选择本地文件夹**，选中 `evaluation-skill` 目录
4. **使用时**，在对话中输入 `@Docs` 选择评估skill

### 方法3：直接在对话中引用

在Cursor Chat中使用 `@` 符号直接引用文件：
```
@evaluation-skill/SKILL.md 帮我设计一个对话质量评估体系
```

---

## 三、在Claude Code中使用

### 方法1：将skill放入项目目录

1. **复制skill到项目中**
   ```bash
   cp -r evaluation-skill /your/project/path/.claude/skills/
   ```

2. **在 `.claude/settings.json` 中配置（可选）**
   ```json
   {
     "skills": {
       "evaluation": {
         "path": ".claude/skills/evaluation-skill",
         "autoLoad": true
       }
     }
   }
   ```

3. **使用时**
   ```bash
   claude "请阅读 .claude/skills/evaluation-skill/SKILL.md 帮我设计评估标准"
   ```

### 方法2：通过CLAUDE.md引用

1. **在项目根目录创建或编辑 `CLAUDE.md`**
   ```markdown
   # 项目说明
   
   ## 可用的Skills
   
   ### 评估方法论
   当需要设计评估体系、创建评分标准、分析评测数据时，请参考：
   - `.claude/skills/evaluation-skill/SKILL.md`
   
   相关资源：
   - 维度设计指南：`.claude/skills/evaluation-skill/references/dimension_design.md`
   - 一致性指标：`.claude/skills/evaluation-skill/references/consistency_metrics.md`
   - 评分模板：`.claude/skills/evaluation-skill/assets/`
   ```

2. **使用Claude Code命令**
   ```bash
   claude "帮我创建一个产品功能的评分Prompt"
   # Claude会自动读取CLAUDE.md了解skill的位置
   ```

### 方法3：命令行直接引用

```bash
# 查看skill内容
claude "cat .claude/skills/evaluation-skill/SKILL.md"

# 请求帮助
claude "基于 .claude/skills/evaluation-skill/assets/rubric_template.md 模板，帮我创建一个AI问答功能的评分细则"
```

---

## 四、使用示例

### 示例1：设计新功能的评估体系

**对话**：
```
请帮我设计一个"智能客服"功能的评估体系。
参考 @evaluation-skill/SKILL.md 的方法论。
```

**预期输出**：
- 功能定义
- 评估维度（基础效果、响应质量、情感连接等）
- 0-4分评分细则
- 典型案例

### 示例2：创建评分Prompt

**对话**：
```
基于 @evaluation-skill/assets/prompt_template.md 模板，
帮我创建一个评估AI翻译质量的Prompt。

评估维度：准确性、流畅性、专业术语、格式规范
```

### 示例3：分析评分一致性

**操作**：
```bash
# 准备数据文件 (CSV格式，包含 human_score 和 model_score 列)

# 运行一致性计算
python scripts/calculate_consistency.py evaluation_data.csv --output results.json

# 运行差异分析
python scripts/analyze_discrepancy.py evaluation_data.csv --output analysis/
```

### 示例4：优化现有Prompt

**对话**：
```
我有一个评分Prompt，模型和人工的一致率只有55%。
请帮我分析可能的问题，并参考 @evaluation-skill/references/consistency_metrics.md 给出优化建议。

[粘贴你的Prompt和一些不一致的案例]
```

---

## 五、最佳实践

### 1. 渐进式使用

- **入门**：先阅读 `SKILL.md` 了解整体框架
- **实践**：使用 `assets/` 中的模板创建评估标准
- **优化**：用 `scripts/` 分析数据，迭代改进

### 2. 小步迭代

- 每次只优化1-2个问题点
- 用 `calculate_consistency.py` 量化每次改动的效果
- 记录版本变更

### 3. 建立案例库

- 收集边界案例
- 为每个分数档位准备锚点示例
- 定期更新典型案例

### 4. 人机协同

- 人工定义标准和处理争议
- 模型执行批量评估
- 定期抽检校准

---

## 六、常见问题

### Q: Skill文件放在哪里最合适？

**推荐**：
- Cursor：项目目录下的 `skills/` 或 `.cursor/skills/`
- Claude Code：项目目录下的 `.claude/skills/`

### Q: 如何让AI自动使用skill？

1. 在 `.cursorrules` 或 `CLAUDE.md` 中明确说明skill的用途和位置
2. 在对话开始时提示AI参考skill
3. 使用 `@` 引用具体文件

### Q: 脚本运行报错怎么办？

确保安装了依赖：
```bash
pip install pandas numpy scipy scikit-learn openpyxl
```

### Q: 如何定制skill？

1. Fork skill目录
2. 修改 `SKILL.md` 中的通用框架为你的领域
3. 更新 `assets/` 中的模板
4. 添加领域特定的 `references/`

---

## 七、更新日志

| 日期 | 版本 | 更新内容 |
|-----|------|---------|
| 2026-01 | v1.0 | 初始版本，包含完整的评估方法论和工具 |
