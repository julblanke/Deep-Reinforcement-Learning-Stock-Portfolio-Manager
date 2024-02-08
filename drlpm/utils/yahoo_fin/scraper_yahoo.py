import pandas as pd
import yfinance as yf


class ScraperYahoo:
    """Scrapes yahoo finance data for stock history data."""

    def __init__(self, stock_symbol: str, period: str, interval: str) -> None:
        """Constructor.

        Args:
            stock_symbol (str): Stock as stock-symbol, e.g. 'AAPL'
            period (str): Time period for data to take into account -- in yfinance terms -- e.g. '2y'
            interval (str): Data points frequency -- in yfinance terms -- e.g. '1d'
        """
        self.stock_symbol = stock_symbol
        self.period = period
        self.interval = interval
        self.df = {}

    def get_stock_data(self) -> pd.DataFrame:
        """Gets Open, High, Low and Close for given stock.

        Returns:
            self.df (pd.DataFrame): Open, High, Low and Close for given stock
        """
        ticker_inst = yf.Ticker(self.stock_symbol)
        df_ = ticker_inst.history(period=self.period, interval=self.interval)
        self.df = df_[["Open", "High", "Low", "Close"]]
        self.df = self.df.sort_index(ascending=False)
        return self.df
