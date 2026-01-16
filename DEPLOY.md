# 部署指南

本文档说明如何构建和部署本 AI 学习路线图网站。

## 本地开发

1.  **安装依赖**
    ```bash
    npm install
    # 或者如果不使用 package.json 中的默认 vitepress
    npm add -D vitepress vue
    ```

2.  **启动开发服务器**
    ```bash
    npm run docs:dev
    ```
    访问 `http://localhost:5173` 查看实时预览。

## 构建生产版本

运行以下命令生成静态文件：
```bash
npm run docs:build
```
构建产物将位于 `docs/.vitepress/dist` 目录。

## 部署流程

### GitHub Pages 详细部署指南

GitHub Pages 是一个免费的静态站点托管服务，非常适合部署 VitePress 文档。

#### 1. 准备工作

确保您的代码已经推送到 GitHub 仓库。

#### 2. 配置 Base URL (关键步骤)

如果您的网站部署在 `https://<USERNAME>.github.io/<REPO>/`（例如仓库名为 `ai-roadmap`），您**必须**在 `docs/.vitepress/config.mts` 中设置 `base` 属性。

*   **User/Org Page** (`https://<USERNAME>.github.io/`)：不需要配置 `base`（默认为 `/`）。
*   **Project Page** (`https://<USERNAME>.github.io/<REPO>/`)：需要设置为 `/<REPO>/`。

**修改示例**：
如果您的 GitHub 仓库名称是 `ai-roadmap`，请修改 `docs/.vitepress/config.mts`：

```typescript
export default defineConfig({
  base: '/ai-roadmap/', // 注意前后都有斜杠
  title: "AI Developer Roadmap",
  // ...
})
```

#### 3. 配置 GitHub Actions

在项目根目录创建或确认存在 `.github/workflows/deploy.yml` 文件。此文件定义了自动构建和部署的流程。

**文件内容** (`.github/workflows/deploy.yml`)：

```yaml
name: Deploy VitePress site to Pages

on:
  # 在推送到 main 分支时触发部署
  push:
    branches: [main]

  # 允许从 Actions 选项卡手动运行工作流
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0 # 如果未启用 lastUpdated，则不需要
      
      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm # 使用 npm 缓存加速构建
          
      - name: Setup Pages
        uses: actions/configure-pages@v5
        
      - name: Install dependencies
        run: npm ci
        
      - name: Build with VitePress
        run: npm run docs:build
        
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: docs/.vitepress/dist

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    needs: build
    runs-on: ubuntu-latest
    name: Deploy
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

#### 4. GitHub 仓库设置

1.  打开您的 GitHub 仓库页面。
2.  点击顶部的 **Settings** (设置) 选项卡。
3.  在左侧侧边栏中，向下滚动并点击 **Pages**。
4.  在 **Build and deployment** 部分：
    *   **Source**: 选择 **GitHub Actions** (这一步非常重要，不要选 Deploy from a branch)。
5.  设置完成后，页面上方可能会提示 "GitHub Pages source saved."。

#### 5. 触发部署

1.  将上述更改（包括 `deploy.yml` 和 `config.mts` 的修改）提交并推送到 GitHub `main` 分支。
2.  GitHub Actions 会自动检测到推送并开始构建。
3.  您可以点击仓库顶部的 **Actions** 选项卡查看构建进度。
4.  构建成功后（通常显示绿色的勾），进入 **Settings > Pages**，您将看到网站的访问链接（例如 `https://username.github.io/ai-roadmap/`）。

#### 常见问题

*   **样式丢失/404错误**：通常是因为 `base` 配置错误。请检查 `config.mts` 中的 `base` 是否与您的仓库名称一致，并且以 `/` 开头和结尾。
*   **构建失败**：请检查 Actions 日志。常见原因是依赖安装失败或 Markdown 语法错误。

### Vercel / Netlify

1.  在 Vercel/Netlify 导入 GitHub 仓库。
2.  设置构建命令：`npm run docs:build`
3.  设置输出目录：`docs/.vitepress/dist`
4.  点击部署即可。
