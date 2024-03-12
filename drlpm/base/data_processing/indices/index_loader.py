import os
import logging
import pandas as pd
from drlpm.base.data_processing.stocks.scraper_yahoo import ScraperYahoo


class IndexLoader:
    """Handles anything related to loading of index data."""

    @staticmethod
    def add_indices(data: pd.DataFrame, period: str, interval: str, update_data: bool) -> None:
        """Scrapes index data from Yahoo if update_data is True, otherwise reads .csv from project data directory.

        Args:
            data (pd.DataFrame): Dataframe containing OHLC data of stock and indicators
            period (str): Time period for data to take into account -- in yfinance terms -- e.g. '2y'
            interval (str): Data points frequency -- in yfinance terms -- e.g. '1d'
            update_data (bool): Whether to update stock data
        """
        logger = logging.getLogger()

        indices_to_track = ["SPY", "QQQ", "SMH", "XLV", "XLP", "XLE", "XLF", "XLI", "XLU", "XLB", "XLK", "KRE"]
        for ticker in indices_to_track:
            if update_data:
                ticker_data = (ScraperYahoo(stock_symbol=ticker,
                                            period=period,
                                            interval=interval)
                               .get_stock_data())
                data[ticker] = ticker_data.iloc[::-1]["Close"]
                os.makedirs("./drlpm/data/indices/", exist_ok=True)
                ticker_data.to_csv(f"./drlpm/data/indices/{ticker}.csv", index=False)
                logger.info(f"Successfully loaded '{ticker}' data from yfinance.")
            else:
                ticker_data = pd.read_csv(f"./drlpm/data/indices/{ticker}.csv")
                data[ticker] = ticker_data.iloc[::-1]["Close"]
                logger.info(f"Successfully loaded '{ticker}' data from local .csv file.")
