from pathlib import Path
import sys
import logging

SCRIPT_DIR = Path(__file__).resolve().parent


def get_logger(log_dir, name, log_filename, level=logging.INFO):
    log_path = Path(log_dir)
    if not log_path.is_absolute():
        log_path = SCRIPT_DIR / log_path

    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler = logging.FileHandler(log_path / log_filename)
    file_handler.setFormatter(file_formatter)

    console_formatter = logging.Formatter("%(asctime)s - %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    print("Log directory:", log_path)

    return logger
