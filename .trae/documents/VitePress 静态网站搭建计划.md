# VitePress 静态网站搭建计划

根据您的需求，我制定了以下实施计划，利用现有的 `docs` 目录（看来已经是一个初始化的 VitePress 项目）来构建您的 AI 学习路线图网站。

## 1. 内容分析与迁移规划
我们将 `mydocs` 中的文档迁移到 VitePress 的标准目录结构中，并进行必要的标准化处理。

- **源目录**: `d:\Dev\aicode\to-be-a-ai-developer\mydocs`
- **目标目录**: `d:\Dev\aicode\to-be-a-ai-developer\docs`
- **文件重命名策略**: 为了 URL 友好，建议将中文文件名映射为英文（保留序号以维持顺序），例如：
    - `00-AI学习总体路线图.md` -> `docs/index.md` (作为主页) 或 `docs/guide/roadmap.md`
    - `01-第一阶段：数学基础.md` -> `docs/guide/01-math-basics.md`
    - ...
- **内容处理**:
    - 为每个文件添加 Frontmatter（元数据），包含 `title`、`description` 等。
    - **关键修复**: 修正文档间的相对链接（如 `00` 文件中引用的 `01` 文件路径），确保点击跳转正常。

## 2. VitePress 项目配置
利用现有的 `docs/.vitepress` 结构进行深度配置。

- **清理**: 删除现有的示例文件 (`api-examples.md`, `markdown-examples.md`)。
- **配置文件 (`config.mts`)**:
    - **站点信息**: 设置标题 "AI Developer Roadmap" 和描述。
    - **主题配置**:
        - **导航栏 (Nav)**: 设置 "首页", "路线图", "资源" 等链接。
        - **侧边栏 (Sidebar)**: 配置为自动生成或手动指定的层级结构，确保左侧显示完整的章节导航。
        - **搜索功能**: 启用 VitePress 原生本地搜索 (`provider: 'local'`)。
        - **社交链接**: 如果有 GitHub 仓库，添加对应图标。

## 3. 功能增强与样式
- **响应式布局**: VitePress 默认支持，我们将检查表格和长代码块在移动端的显示效果。
- **SEO 优化**: 在 `config.mts` 中配置 `head` 标签，添加 keywords 和 description。
- **最后更新时间**: 启用 `lastUpdated` 显示文档更新时间。

## 4. 构建与部署准备
- **脚本配置**: 确保 `package.json` 包含 `docs:dev`, `docs:build`, `docs:preview`。
- **部署文档**: 创建 `DEPLOY.md`，记录如何构建静态文件以及如何部署到 GitHub Pages 或 Vercel。

## 执行步骤
1.  **清理与迁移**: 清空 `docs` 示例，复制并重命名 Markdown 文件。
2.  **链接修复**: 批量替换文档内部的文件引用链接。
3.  **配置编写**: 更新 `config.mts` 实现导航、侧边栏和搜索。
4.  **验证**: 运行本地服务器，检查所有链接和样式。
5.  **文档交付**: 生成构建脚本和部署说明。

请确认是否同意执行此计划？如果同意，我将开始迁移和配置工作。