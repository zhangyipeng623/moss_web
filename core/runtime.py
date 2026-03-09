import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIG_SNAPSHOT_NAME = "experiment.snapshot.json"
BACKEND_LOG_NAME = "backend.log"
AGENT_LOG_NAME = "agent.log"
DATABASE_NAME = "moss.db"


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: Path
    backend_log_path: Path
    agent_log_path: Path
    database_path: Path
    config_snapshot_path: Path


def create_run_context() -> RunContext:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = RUNS_DIR / f"{run_id}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return RunContext(
        run_id=run_dir.name,
        run_dir=run_dir,
        backend_log_path=run_dir / BACKEND_LOG_NAME,
        agent_log_path=run_dir / AGENT_LOG_NAME,
        database_path=run_dir / DATABASE_NAME,
        config_snapshot_path=run_dir / CONFIG_SNAPSHOT_NAME,
    )


def export_run_context(context: RunContext) -> None:
    os.environ["MOSS_RUN_ID"] = context.run_id
    os.environ["MOSS_RUN_DIR"] = str(context.run_dir)
    os.environ["MOSS_DB_PATH"] = str(context.database_path)


def configure_logging(
    log_file: Path,
    logger_name: str,
    console_enabled: bool = True,
) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    if console_enabled:
        handlers.insert(0, logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.ERROR)
    return logging.getLogger(logger_name)


def save_config_snapshot(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
