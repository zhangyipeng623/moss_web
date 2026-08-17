# Repository Guidelines

## 项目结构与模块组织

本仓库是面向舆情模拟推演的 MOSS Web 实验平台。`main.py` 是入口，负责创建运行归档并启动后端与 Agent 两个进程。`backend/` 提供 FastAPI 接口、SQLite DAO、推荐服务和静态资源托管；`backend/static/` 是前端构建产物。`moss_agent_client/` 放置 Agent 调度、提示词、记忆和平台访问逻辑。`core/` 管理运行时、配置模型和画像解析。`analysis/` 是离线画像、推荐参数推断和重分层流水线。`frontend/` 是 React/Vite 前端源码。`configs/experiments/` 存放实验样例与 CSV Agent 配置，`runs/` 存放单次实验归档。

## 构建、测试与开发命令

```bash
uv sync
python main.py --config <path/to/calibration_profile.yaml>
uv run python -m analysis.run_analysis portrait --data-path <dir> --user-name <name> --batch
uv run python -m analysis.run_analysis recommender --data-file <xlsx> --portraits-dir <dir>
uv run python -m analysis.run_analysis retier --portraits-dir <dir>
cd frontend && npm run dev
cd frontend && npm run build
cd frontend && npm run lint
```

`main.py --config` 必须使用 YAML 校准文件，不直接加载旧的 `configs/experiments/default.json`。运行实验前必须先执行前端构建，因为后端只从 `backend/static/` 提供 SPA。Vite 开发服务器会把 `/api` 代理到本地 `:8000`。
`uv sync` 安装 Python 依赖；`analysis.run_analysis` 三个子命令分别负责画像生成、推荐参数推断和层级重算。

## 编码风格与命名约定

Python 使用 4 空格缩进，模块和函数采用 `snake_case`，类采用 `PascalCase`。TypeScript/React 组件采用 `PascalCase` 文件名与组件名，普通工具函数使用 `camelCase`。变量、函数和类型命名保持英文；文档、代码注释和提交说明使用简体中文。当前没有统一 Python formatter 或 type checker；前端以 ESLint、TypeScript 和 Vite 构建结果为准。

## 测试指南

仓库尚未配置正式测试框架，`test.py` 为 gitignored 的本地临时验证文件。修改后至少运行相关命令：前端改动执行 `cd frontend && npm run lint && npm run build`；后端或配置模型改动可用 `python3 -m py_compile <files>` 做语法检查，并用最小 YAML 配置手动启动关键路径。新增复杂逻辑时，优先补充可重复的脚本或说明验证步骤。

## 提交与 Pull Request 指南

历史提交混合使用中文短句和 `feat:` 前缀。新提交建议使用简体中文、动宾结构，必要时保留类型前缀，例如 `feat: 添加 simple 画像模式` 或 `修复推荐权重映射`。PR 应说明变更目的、影响范围、验证命令和配置迁移点；涉及前端界面时附截图，涉及实验结果时说明使用的数据、YAML 配置和 `runs/<timestamp>/` 归档位置。

## 配置与运行注意事项

根目录 `.env` 需提供 `API_KEY` 与 `BASE_URL`；`BACKEND_URL` 默认 `127.0.0.1`，`BACKEND_PORT` 默认 `8000`。离线流程顺序为 `portrait --batch`、`recommender` 生成 `calibration_profile.yaml`、再用该 YAML 运行 `main.py`。注意 `sqlite-vec` 依赖原生扩展，不同平台可能需要单独处理安装问题。

推荐嵌入模型由 `calibration_profile.yaml` 的 `embedding.model_name` 统一配置（默认 `BAAI/bge-m3`），离线 `recommender` 与在线 `SocialRecSys` 必须使用同一模型，否则校准出的兴趣权重失效。推荐四维权重由 YAML 经 `SocialRecSys.configure()` 自动注入，无需手动同步。大小模型分层（core 大模型 + mass 小模型）方案见 `docs/plan/大小模型分层与模型配置方案.md`，尚未实现。
