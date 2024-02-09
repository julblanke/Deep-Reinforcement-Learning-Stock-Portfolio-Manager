import os
import logging
import pandas as pd
from drlpm.utils.logger import Logger
from drlpm.data_processing.stocks.stock_loader import StockLoader
from drlpm.data_processing.indicators.indicator_loader import IndicatorLoader


class Dataloader:
    """Loads data from Yahoo."""

    def __init__(self, stock_symbols: list, period: str, interval: str, update_data: bool) -> None:
        """Constructor.

        Args:
            stock_symbols (list(str)): Stock symbols of user defined stocks
            period (str): Time period for data to take into account -- in yfinance terms -- e.g. '2y'
            interval (str): Data points frequency -- in yfinance terms -- e.g. '1d'
            update_data (bool): Whether to update stock data
        """
        self.stock_symbols = stock_symbols
        self.period = period
        self.interval = interval
        self.update_data = update_data
        self.n_samples = None

        Logger.initialize_logger()
        self.logger = logging.getLogger()

        os.makedirs("./drlpm/data", exist_ok=True)

    def get_data(self) -> pd.DataFrame:
        """Calls the respective dataloader to get dataframe. Drops nan values due to indicators.

        Returns:
            data (pd.DataFrame): Data of given stocks -- includes OHLC data and indicators
        """
        self.logger.info("Loading stock data from yfinance..")
        data = StockLoader.get_stock_data(stock_symbols=self.stock_symbols,
                                          period=self.period,
                                          interval=self.interval,
                                          update_data=self.update_data)

        self.logger.info("Calculating indicators..")
        IndicatorLoader.add_indicators(data=data, stock_symbols=self.stock_symbols)

        data.dropna(inplace=True)       # gets rid of nan created by sma/ema calc
        return data
