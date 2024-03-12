import os
import logging
import pandas as pd
from tqdm import tqdm
from drlpm.base.data_processing.stocks.scraper_yahoo import ScraperYahoo


class StockLoader:
    """Handles anything related to loading of stock data."""

    @staticmethod
    def get_stock_data(stock_symbols: list, period: str, interval: str, update_data: bool) -> pd.DataFrame:
        """Scrapes stock data from Yahoo if update_data is True, otherwise reads .csv from project data directory.

        Args:
            stock_symbols (list(str)): Stock symbols of user defined stocks
            period (str): Time period for data to take into account -- in yfinance terms -- e.g. '2y'
            interval (str): Data points frequency -- in yfinance terms -- e.g. '1d'
            update_data (bool): Whether to update stock data

        Returns:
            conc_df (pd.DataFrame): Concatenated dataframe consisting of Open, High, Low and Close values of stocks
        """
        logger = logging.getLogger()

        conc_df = pd.DataFrame()
        for stock in tqdm(stock_symbols):
            if update_data:
                stock_data = (ScraperYahoo(stock_symbol=stock,
                                           period=period,
                                           interval=interval)
                              .get_stock_data())
                os.makedirs("./drlpm/data/stocks/", exist_ok=True)
                stock_data.to_csv(f"./drlpm/data/stocks/{stock}.csv", index=False)
                logger.info(f"Successfully loaded '{stock}' data from yfinance.")
            else:
                stock_data = pd.read_csv(f"./drlpm/data/stocks/{stock}.csv")
                logger.info(f"Successfully loaded '{stock}' data from local .csv file.")

            stock_data.columns = [f"{col}_{stock}" for col in stock_data.columns]
            conc_df = pd.concat([conc_df, stock_data], axis=1)

        conc_df.dropna(inplace=True)  # prevents mismatch of datapoints but basically changes period
        reversed_data = conc_df.iloc[::-1]
        return reversed_data
