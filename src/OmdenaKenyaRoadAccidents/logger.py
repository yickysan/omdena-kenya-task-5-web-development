import logging
from pathlib import Path
from datetime import datetime

HERE = Path(__file__)

LOG_DIR = f"{datetime.now().strftime('%m_%d_%Y')}"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = HERE.parent.parent.parent / "logs"
logs_path.mkdir(parents=True, exist_ok=True)

LOG_DIR_PATH = logs_path / LOG_DIR
LOG_DIR_PATH.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = logs_path / LOG_DIR_PATH / LOG_FILE

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)