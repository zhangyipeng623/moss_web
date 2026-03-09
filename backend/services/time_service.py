from datetime import datetime, timedelta
from enum import Enum

from backend.services.logger_service import logger


class TimeMode(str, Enum):
    STEP = "step"
    TIME = "time"


class TimeService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TimeService, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return

        # Default configuration
        self.mode = TimeMode.STEP
        self.start_time = datetime.now()
        self.real_start_time = datetime.now()
        self.time_scale = (
            1.0  # In STEP mode: seconds per step; In TIME mode: multiplier
        )
        self.current_step = 0

        self.initialized = True
        logger.info("TimeService initialized")

    def initialize(
        self,
        start_time: datetime,
        mode: TimeMode = TimeMode.STEP,
        time_scale: float = 1.0,
    ):
        """Re-initialize with specific parameters"""
        self.start_time = start_time
        self.mode = mode
        self.time_scale = time_scale
        self.real_start_time = datetime.now()
        self.current_step = 0
        logger.info(
            f"TimeService configured: mode={mode}, start_time={start_time}, scale={time_scale}"
        )

    def get_current_time(self) -> datetime:
        if self.mode == TimeMode.STEP:
            # In STEP mode, time advances by time_scale seconds per step
            delta_seconds = self.current_step * self.time_scale
            return self.start_time + timedelta(seconds=delta_seconds)

        elif self.mode == TimeMode.TIME:
            # In TIME mode, time advances based on real time elapsed * scale
            real_elapsed = datetime.now() - self.real_start_time
            simulated_elapsed = real_elapsed * self.time_scale
            return self.start_time + simulated_elapsed

        return self.start_time

    def update_step(self):
        self.current_step += 1
        logger.info(f"Step updated to {self.current_step}")


# Global instance
time_service = TimeService()
