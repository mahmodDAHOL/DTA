"""Logging the outputs to log file."""
import logging
from datetime import datetime, timezone

from .constants import project_path

LOG_FILE = f"{datetime.now(timezone.utc).strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_folder = project_path.joinpath("logs")
log_folder.mkdir(exist_ok=True)
LOG_FILE_PATH = log_folder.joinpath(LOG_FILE)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
