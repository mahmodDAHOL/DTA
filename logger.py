import logging
from datetime import datetime
from pathlib import Path

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
project_path = Path.cwd()
log_folder = project_path.joinpath("logs")
log_folder.mkdir(exist_ok=True)
LOG_FILE_PATH = log_folder.joinpath(LOG_FILE)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
