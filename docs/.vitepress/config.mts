import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import mathjax3 from 'markdown-it-mathjax3'

// https://vitepress.dev/reference/site-config
export default withMermaid(defineConfig({
  title: "AI Developer Roadmap",
  description: "AI工程师的完整学习路线图",
  lang: 'zh-CN',
  
  markdown: {
    config: (md) => {
      md.use(mathjax3)
    }
  },

  vite: {
    optimizeDeps: {
      include: [
        'mermaid',
        'dayjs',
        'dayjs/plugin/duration'
      ]
    }
  },

  lastUpdated: true,//显示md最后更新时间
  base: '/ai-developer-roadmap/',

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: '首页', link: '/' },
      { text: '路线图', link: '/guide/00-roadmap/' },
      { text: '第一阶段', link: '/guide/01-math-basics/' },
      {
        text: '相关资源',
        items: [
          {
            text: '常用工具 & 框架',
            items: [
              { text: 'PyTorch 官网', link: 'https://pytorch.org/' },
              { text: 'NumPy 文档', link: 'https://numpy.org/doc/stable/' },
              { text: 'LangChain 文档', link: 'https://python.langchain.com/docs/get_started/introduction' },
              { text: 'Anaconda', link: 'https://www.anaconda.com/download' }
            ]
          },
          {
            text: '数学与可视化',
            items: [
              { text: 'Desmos 图形计算器', link: 'https://www.desmos.com/' },
              { text: 'Seeing Theory (概率)', link: 'https://seeing-theory.brown.edu/' },
              { text: '梯度下降可视化', link: 'https://losslandscape.com/' }
            ]
          },
          {
            text: '优质教程',
            items: [
              { text: '3Blue1Brown (数学)', link: 'https://space.bilibili.com/88461692/channel/seriesdetail?sid=1528931' },
              { text: 'StatQuest (统计)', link: 'https://space.bilibili.com/23910356' }
            ]
          },
          {
            text: 'API 服务',
            items: [
              { text: 'DeepSeek Platform', link: 'https://platform.deepseek.com/' },
              { text: 'SiliconFlow', link: 'https://siliconflow.cn/' }
            ]
          }
        ]
      }
    ],

    sidebar: [
      {
        text: '学习路线',
        items: [
          { text: '总体路线图', link: '/guide/00-roadmap/' },
          {
            text: '1. 数学基础补强',
            collapsed: false,
            items: [
              { text: '阶段概览', link: '/guide/01-math-basics/' },
              { text: '1.1 线性代数', link: '/guide/01-math-basics/01-linear-algebra' },
              { text: '1.2 微积分', link: '/guide/01-math-basics/02-calculus' },
              { text: '1.3 概率统计', link: '/guide/01-math-basics/03-probability-statistics' },
              { text: '1.4 综合项目', link: '/guide/01-math-basics/04-projects' }
            ]
          },
          {
            text: '2. Python数据处理',
            collapsed: false,
            items: [
              { text: '阶段概览', link: '/guide/02-python-data/' },
              { text: '2.1 NumPy进阶', link: '/guide/02-python-data/01-numpy' },
              { text: '2.2 Pandas数据分析', link: '/guide/02-python-data/02-pandas' },
              { text: '2.3 数据可视化', link: '/guide/02-python-data/03-visualization' },
              { text: '2.4 Jupyter最佳实践', link: '/guide/02-python-data/04-jupyter' },
              { text: '2.5 综合项目', link: '/guide/02-python-data/05-projects' }
            ]
          },
          {
            text: '3. 机器学习基础',
            collapsed: false,
            items: [
              { text: '阶段概览', link: '/guide/03-ml-basics/' },
              { text: '3.1 基础概念', link: '/guide/03-ml-basics/01-concepts' },
              { text: '3.2 数据预处理', link: '/guide/03-ml-basics/02-preprocessing' },
              { text: '3.3 监督学习-回归', link: '/guide/03-ml-basics/03-regression' },
              { text: '3.4 监督学习-分类', link: '/guide/03-ml-basics/04-classification' },
              { text: '3.5 评估与调优', link: '/guide/03-ml-basics/05-evaluation' },
              { text: '3.6 无监督学习', link: '/guide/03-ml-basics/06-unsupervised' },
              { text: '3.7 综合项目', link: '/guide/03-ml-basics/07-projects' }
            ]
          },
          {
            text: '4. 深度学习',
            collapsed: false,
            items: [
              { text: '阶段概览', link: '/guide/04-deep-learning/' },
              { text: '4.1 神经网络基础', link: '/guide/04-deep-learning/01-neural-networks' },
              { text: '4.2 PyTorch基础', link: '/guide/04-deep-learning/02-pytorch' },
              { text: '4.3 卷积神经网络', link: '/guide/04-deep-learning/03-cnn' },
              { text: '4.4 循环神经网络', link: '/guide/04-deep-learning/04-rnn' },
              { text: '4.5 Transformer', link: '/guide/04-deep-learning/05-transformer' },
              { text: '4.6 综合项目', link: '/guide/04-deep-learning/06-projects' }
            ]
          },
          {
            text: '5. 大语言模型应用',
            collapsed: false,
            items: [
              { text: '阶段概览', link: '/guide/05-llm-apps/' },
              { text: '5.1 大模型基础', link: '/guide/05-llm-apps/01-llm-basics' },
              { text: '5.2 Prompt工程', link: '/guide/05-llm-apps/02-prompt-engineering' },
              { text: '5.3 LangChain框架', link: '/guide/05-llm-apps/03-langchain' },
              { text: '5.4 RAG系统', link: '/guide/05-llm-apps/04-rag' },
              { text: '5.5 Agent开发', link: '/guide/05-llm-apps/05-agents' },
              { text: '5.6 综合项目', link: '/guide/05-llm-apps/06-projects' }
            ]
          },
          {
            text: '6. AI工程化',
            collapsed: false,
            items: [
              { text: '阶段概览', link: '/guide/06-ai-engineering/' },
              { text: '6.1 模型服务化', link: '/guide/06-ai-engineering/01-model-serving' },
              { text: '6.2 容器化部署', link: '/guide/06-ai-engineering/02-containerization' },
              { text: '6.3 Java集成AI', link: '/guide/06-ai-engineering/03-java-integration' },
              { text: '6.4 性能优化', link: '/guide/06-ai-engineering/04-optimization' },
              { text: '6.5 MLOps基础', link: '/guide/06-ai-engineering/05-mlops' },
              { text: '6.6 综合项目', link: '/guide/06-ai-engineering/06-projects' }
            ]
          }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/vuejs/vitepress' }
    ],

    search: {
      provider: 'local'
    },

    footer: {
      message: '从零开始的AI应用之路.',
      copyright: 'Copyright © 2026 AI Developer Roadmap'
    }
  }
}))
