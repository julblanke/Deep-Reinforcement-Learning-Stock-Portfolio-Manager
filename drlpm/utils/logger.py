import os
import logging
from logging import handlers


class Logger:
    """Initializes logger."""

    @staticmethod
    def initialize_logger() -> None:
        """Initializes INFO logger with handler for output in logs file."""
        os.makedirs("./logs", exist_ok=True)

        logging.basicConfig(level=logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s "
        )
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        if not any(isinstance(handler, logging.handlers.RotatingFileHandler) for handler in logger.handlers):
            file_handler = logging.handlers.RotatingFileHandler(
                os.path.join("./logs/logs.log"), maxBytes=1024 * 1024, backupCount=3
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)
