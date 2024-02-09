import pandas as pd


class MovingAverages:
    """Class to calculate moving averages."""

    @staticmethod
    def add_sma_window(data: pd.DataFrame, stock_symbols: list, window_size: int) -> None:
        """Calculates SMA.

        Args:
            data (pd.DataFrame): Dataframe consisting of Open, High, Low and Close values of stocks
            stock_symbols (list(str)): Stock symbols of user defined stocks
            window_size (int): Window size for moving averages calculation
        """
        for stock in stock_symbols:
            data[f'sma{window_size}_{stock}'] = data[f'Close_{stock}'].rolling(window=window_size).mean()

    @staticmethod
    def add_ema_window(data: pd.DataFrame, stock_symbols: list, window_size: int) -> None:
        """Calculates EMA.

        Args:
            data (pd.DataFrame): Dataframe consisting of Open, High, Low and Close values of stocks
            stock_symbols (list(str)): Stock symbols of user defined stocks
            window_size (int): Window size for moving averages calculation
        """
        for stock in stock_symbols:
            data[f'ema{window_size}_{stock}'] = (data[f'Close_{stock}'].ewm(span=window_size,
                                                                            adjust=False,
                                                                            min_periods=window_size)
                                                 .mean())
