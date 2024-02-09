import logging
import pandas as pd
from drlpm.utils.logger import Logger
from drlpm.data_processing.indicators.moving_averages import MovingAverages


WINDOW_SIZES = [20, 50, 100, 200]


class IndicatorLoader:
    """Handles anything related to loading of technical indicator data."""

    @staticmethod
    def add_indicators(data: pd.DataFrame, stock_symbols: list) -> None:
        """Adds technical indicators to given dataframe.

        Args:
            data (pd.DataFrame): Dataframe consisting of Open, High, Low and Close values of stocks
            stock_symbols (list(str)): Stock symbols of user defined stocks
        """
        Logger.initialize_logger()
        logger = logging.getLogger()

        for window in WINDOW_SIZES:
            MovingAverages.add_sma_window(data=data, stock_symbols=stock_symbols, window_size=window)
            logger.info(f"Successfully added sma_{window} to data.")
        for window in WINDOW_SIZES:
            MovingAverages.add_ema_window(data=data, stock_symbols=stock_symbols, window_size=window)
            logger.info(f"Successfully added ema_{window} to data.")
