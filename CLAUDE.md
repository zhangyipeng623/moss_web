# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 常用命令

```bash
# Python 依赖管理（使用 uv）
uv sync                          # 安装依赖
uv add <包名>                     # 添加依赖

# 离线分析脚本（核心入口）
uv run python -m analysis.run_analysis portrait --data-path <目录> --user-name <用户名> [--batch]
uv run python -m analysis.run_analysis recommender --data-file <xlsx文件> --portraits-dir <画像目录>
uv run python -m analysis.run_analysis retier --portraits-dir <画像目录>

# 在线推演实验
python main.py --config <path/to/calibration_profile.yaml>

# 前端
cd frontend && npm run dev       # 开发服务器（Vite）
cd frontend && npm run build     # 构建到 backend/static/
cd frontend && npm run lint      # ESLint 检查
```

项目当前没有自动化测试套件（`test.py` 已被 gitignore，属于临时调试文件）。

## 整体架构

系统是一个**社交媒体模拟推演平台**，分为离线分析和在线实验两种运行模式。

### 进程模型

`main.py` 通过 `multiprocessing` 启动两个独立进程：

- **Backend 进程** — FastAPI 服务（端口 8000），提供 REST API 并托管前端静态资源。在 Agent 启动前会通过健康检查确认就绪。
- **Agent 进程** — 运行 `AgentGraph` 循环：遍历已配置的 Agent，每个 Agent 通过 LangChain `ChatOpenAI` 决策，调用 Backend API 执行动作（发帖/评论/点赞/转发/引用）。

每次运行在 `runs/<运行ID>/` 下生成独立归档，包含 `moss.db`、`backend.log`、`agent.log`、`experiment.snapshot.json`。

### 目录结构

| 目录 | 职责 |
|------|------|
| `analysis/` | 离线工具：用户画像生成（多阶段 LLM 流水线）、推荐参数反推（ABM/EM）、影响力重分级 |
| `backend/` | FastAPI 应用：REST 路由、DAO（SQLite + sqlite-vec 向量存储）、推荐服务、时间服务 |
| `moss_agent_client/` | Agent 决策循环、平台 API 封装（`RemotePlatform`）、记忆系统、提示词构建 |
| `core/` | 实验配置加载（`ExperimentConfig` Pydantic 模型）、运行上下文归档 |
| `frontend/` | React 19 + Vite + Tailwind v4 单页应用；构建产物输出到 `backend/static/` |
| `configs/experiments/` | 实验 JSON 配置文件 + `agents.csv` 批量 Agent 配置 |

### 数据流转

1. **离线**：原始 Excel 数据 → `portrait --batch` → 画像 JSON（含 `influence_tier` L4/L5 分级）输出到 `analysis_outputs/portraits/`
2. **离线**：内容观测 Excel → `recommender` → 最优权重 JSON 输出到 `analysis_outputs/recommender/`
3. **在线**：`main.py` 加载实验配置 → 启动 Backend + Agent → Agent 通过 `agents.csv` 读取画像 JSON → 对 Backend API 执行决策循环 → Feed 排序使用 `SocialRecSys`（SentenceTransformer 嵌入 + 加权打分）

### Agent 画像体系

Agent 通过 CSV 配置（`agents.csv`）消费预生成的画像 JSON，使用 `profile_mode=default` + `profile_path=<json路径>`。画像 JSON 包含结构化字段：`stable_profile`（长期兴趣、价值立场、表达风格、社交角色）、`behavior_profile`（活跃模式、动作偏好、触发规则）、`agent_profile`（身份/兴趣/价值/风格/行为/互动摘要 + 表达/行动/回避规则）、`simulation_init`（情绪、目标、关注话题）。

### 推荐服务（`backend/services/social_recsys.py`）

使用 `SentenceTransformer` 生成内容嵌入向量（模型名由 YAML `embedding.model_name` 配置，默认 `Alibaba-NLP/gte-multilingual-base`），通过 `sqlite-vec` 存储。Feed 排序综合四个加权分数：`W_CHRONO`（时间衰减）、`W_BELIEF`（立场/兴趣亲和度）、`W_POP`（热度）、`W_RAND`（探索噪声）。权重由 `calibration_profile.yaml` 经 `SocialRecSys.configure()` 自动注入，映射关系为 `w_time→W_CHRONO`, `w_i→W_BELIEF`, `w_pop→W_POP`, `w_rand→W_RAND`。

### Backend API 概览

- `POST /api/v1/login` — 注册/登录 Agent
- `GET /api/v1/feed?user_id=&limit=` — 个性化信息流
- `GET /api/v1/posts?limit=&offset=` — 全量帖子列表
- `POST /api/v1/posts` — 发帖
- `POST /api/v1/comments` — 评论
- `POST /api/v1/posts/like|repost|quote` — 互动操作
- `GET /api/v1/traces` — 近期行为轨迹
- `GET/POST /api/v1/time` — 模拟时间控制
- `GET /api/v1/health` — 健康检查（绕过 SPA 挂载）

### 离线分析流水线详情

**画像生成**（`analysis/user_portrait_generator.py`）对每个用户执行 5 阶段 LLM 流水线：(A) 并行分块证据抽取 → (B) 稳定画像（兴趣、立场、风格、角色）→ (C) 行为画像（活跃模式、动作概率、触发规则）→ (D) 单主人格 Agent 画像压缩 → (E) 模拟初始状态。每阶段独立重试（最多 3 次）。批量模式下，先预计算全部用户的 `account_influence`，按 P84 百分位切分为 L4/L5 两级，再调 LLM 生成画像并注入层级信息。

**推荐参数反推**（`analysis/recommender_parameter_inference.py`）使用**向量化 ABM + Optuna EM**引擎（v8）：
- 语义处理：Kernel PCA (RBF) 将用户画像嵌入投影到一维立场轴，结合话题嵌入计算兴趣匹配
- 种群合成：种子用户（来自画像 JSON）通过精英保留 + Zipf 影响力分布扩展为全量仿真种群
- 向量化仿真：`VectorizedABMEngine` 支持 Soft Backfire 信念更新（立场冲突时极端用户有概率反而强化原有立场）
- EM 校准：E 步用 Optuna 校准每条内容的 `p_base` 概率，M 步搜索全局权重 `w_i/w_pop/w_time/w_rand`
- 权重映射：`w_i → W_BELIEF`, `w_pop → W_POP`, `w_time → W_CHRONO`, `w_rand → W_RAND`

### 环境变量

复制 `.env.example` 为 `.env`，设置 `API_KEY`、`BASE_URL`（模型服务地址）、`BACKEND_URL`（默认 `127.0.0.1`）、`BACKEND_PORT`（默认 `8000`）。分析脚本通过 `python-dotenv` 自动从 analysis 目录或项目根目录加载 `.env`。
