import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Visualizer:
    """Creates portfolio performance graphs as .png in streamlit directory."""

    def __init__(self, data: pd.DataFrame, stock_symbols: list, model_info: list) -> None:
        """Constructor.

        Args:
            data (pd.DataFrame): Data of given stocks -- includes OHLC data and indicators
            stock_symbols (list(str)): Stock symbols of user defined stocks
            model_info (list(dict)): Model info - contains data to visualize performance and share amount
        """
        self.data = data
        self.stock_symbols = stock_symbols
        self.model_info = model_info

        # gather info metrics
        self.total_values = list()
        self.portfolio_values = list()
        self.cash_values = list()
        self.stock_shares = list()
        for info in self.model_info:
            self.total_values.append(info[0]["total"])
            self.portfolio_values.append(info[0]["portfolio"])
            self.cash_values.append(info[0]["cash"])
            self.stock_shares.append(info[0]["stocks"])

        # misc
        self.x_axis_sample_amount = self.data.shape[0]

    def create_graphs(self) -> None:
        """Calls class to create the different graphs. Resets output directory."""
        if os.path.isdir("./streamlit/result_images"):
            shutil.rmtree("./streamlit/result_images")
        os.makedirs("./streamlit/result_images/performance", exist_ok=True)
        os.makedirs("./streamlit/result_images/shares", exist_ok=True)

        self._plot_total_values()
        self._plot_portfolio_values()
        self._plot_cash()
        self._plot_stock_shares()

    def _plot_total_values(self) -> None:
        """Plots the total capital graph -- total capital given by portfolio value + cash."""
        x = np.linspace(0, self.x_axis_sample_amount - 1, self.x_axis_sample_amount)
        y = self.total_values
        plt.plot(x, y)
        plt.title(f"Total")
        plt.savefig(os.path.join("./streamlit/result_images/performance", f"total.png"))
        plt.close()

    def _plot_portfolio_values(self) -> None:
        """Plots the portfolio graph."""
        x = np.linspace(0, self.x_axis_sample_amount - 1, self.x_axis_sample_amount)
        y = self.portfolio_values
        plt.plot(x, y)
        plt.title(f"Portfolio")
        plt.savefig(os.path.join("./streamlit/result_images/performance", f"portfolio.png"))
        plt.close()

    def _plot_cash(self) -> None:
        """Plots the cash graph."""
        x = np.linspace(0, self.x_axis_sample_amount - 1, self.x_axis_sample_amount)
        y = self.cash_values
        plt.plot(x, y)
        plt.title(f"Cash")
        plt.savefig(os.path.join("./streamlit/result_images/performance", f"cash.png"))
        plt.close()

    def _plot_stock_shares(self) -> None:
        """Plots the stock shares graphs."""
        stocks_list = self.stock_shares
        for stock in self.stock_symbols:
            stock_shares = [datapoint[stock] for datapoint in stocks_list]
            x = np.linspace(0, self.x_axis_sample_amount - 1, self.x_axis_sample_amount)
            y = stock_shares
            plt.plot(x, y)
            plt.title(f"{stock} share amount")
            plt.savefig(os.path.join("./streamlit/result_images/shares", f"{stock}_shares.png"))
            plt.close()
