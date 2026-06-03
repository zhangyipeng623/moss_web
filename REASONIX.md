# REASONIX.md — MOSS Web

## Stack
- **Python 3.12+** — backend + Agent + offline analysis
- **FastAPI** — REST API server (`backend/`)
- **React 19 + Vite + Tailwind v4** — SPA (`frontend/`)
- **LangChain / LangChain-OpenAI** — Agent LLM decision loop
- **SQLite + sqlite-vec** — per-run DB with vector embeddings
- **SentenceTransformers** (`all-MiniLM-L6-v2`) — feed ranking embeddings
- **Optuna v8** — recommender parameter inference (`analysis/`)
- **uv** — Python package manager (`uv.lock`)

## Layout
| Dir | Purpose |
|-----|---------|
| `main.py` | Entry — multiprocess orchestration (Backend + Agent) |
| `core/` | ExperimentConfig, RunContext, CalibrationProfile, agent profile resolver |
| `backend/` | FastAPI: routers (`api.py`), DAO, services (`social_recsys.py`, `time_service.py`) |
| `moss_agent_client/` | Agent loop, RemotePlatform client, memory system, prompt builder |
| `frontend/` | React SPA; build output → `backend/static/` (`vite.config.ts` `outDir`) |
| `analysis/` | Offline: portrait generator (5-stage LLM), recommender inference (ABM+EM) |
| `configs/experiments/` | JSON configs, `agents.csv`, `profiles/` portrait JSONs |
| `runs/` | Per-run archives: `<ts>/` each with `moss.db`, logs, config snapshot |

## Commands
```bash
uv sync                                          # install Python deps
python main.py --config <calibration_profile>    # run experiment
uv run python -m analysis.run_analysis portrait --data-path <dir> --user-name <name> [--batch]
uv run python -m analysis.run_analysis recommender --data-file <xlsx> --portraits-dir <dir>
cd frontend && npm run dev                       # Vite dev (proxies /api → :8000)
cd frontend && npm run build                     # tsc + vite build → backend/static/
cd frontend && npm run lint                      # ESLint
```

## Conventions
- **Per-run isolation**: each run → `runs/<timestamp>/` with independent DB, logs, config snapshot
- **Frontend build target**: Vite `outDir: ../backend/static`, `emptyOutDir: true`
- **Agent profiles**: CSV-driven (`agents.csv`), `profile_mode=default` + `profile_path=<json>`
- **Recommender weights**: `SocialRecSys.configure()` called at startup overrides module constants
- **No test framework**: no pytest/ruff/mypy configured; `test.py` is gitignored

## Watch out for
- **Config drift**: `main.py --config` expects YAML (`CalibrationProfile`), but README references `default.json` — verify format before editing
- **Frontend must be built** before running experiment (backend serves from `backend/static/`)
- **`sqlite-vec`** requires native extension wheel; may fail on unsupported platforms
- **Agent waits for health check**: `GET /api/v1/health` must pass (≤30s) before Agent process spawns
- **`CLAUDE.md` exists**: auto-generated for another tool; may diverge from this file
