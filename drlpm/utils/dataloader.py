import os
import logging
import pandas as pd
from tqdm import tqdm
from drlpm.utils.logger import Logger
from drlpm.utils.yahoo_fin.scraper_yahoo import ScraperYahoo


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

    def get_data(self) -> pd.DataFrame:
        """Scrapes stock data from Yahoo if update_data is True, otherwise reads .csv from project data directory.

        Returns:
            conc_df (pd.DataFrame): Concatenated dataframe consisting of Open, High, Low and Close values of stocks
        """
        os.makedirs("./drlpm/data/stocks", exist_ok=True)

        conc_df = pd.DataFrame()
        for stock in tqdm(self.stock_symbols):
            if self.update_data:
                stock_data = (ScraperYahoo(stock_symbol=stock,
                                           period=self.period,
                                           interval=self.interval)
                              .get_stock_data())
                stock_data.to_csv(f"./drlpm/data/stocks/{stock}.csv", index=False)
            else:
                stock_data = pd.read_csv(f"./drlpm/data/stocks/{stock}.csv")
            stock_data.columns = [f"{col}_{stock}" for col in stock_data.columns]
            conc_df = pd.concat([conc_df, stock_data], axis=1)

        conc_df.dropna(inplace=True)        # prevents mismatch of datapoints but basically changes period
        conc_df.sort_index(ascending=False, inplace=True)
        return conc_df
