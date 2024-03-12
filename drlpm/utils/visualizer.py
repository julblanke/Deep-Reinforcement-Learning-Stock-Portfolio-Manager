import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SingleAgentVisualizer:
    """Creates portfolio performance graphs as .png in streamlit directory."""

    def __init__(self, data: pd.DataFrame, stock_symbols: list, algo_info: list) -> None:
        """Constructor.

        Args:
            data (pd.DataFrame): Data of given stocks -- includes OHLC data and indicators and indices
            stock_symbols (list(str)): Stock symbols of user defined stocks
            algo_info (list(dict)): Algo info - contains data to visualize performance and share amount
        """
        self.data = data
        self.stock_symbols = stock_symbols
        self.algo_info = algo_info
        self.x_axis_sample_amount = self.data.shape[0]

        # gather info metrics
        self.total_values = list()
        self.portfolio_values = list()
        self.cash_values = list()
        self.stock_shares = list()
        for info in self.algo_info:
            self.total_values.append(info[0]["total"])
            self.portfolio_values.append(info[0]["portfolio"])
            self.cash_values.append(info[0]["cash"])
            self.stock_shares.append(info[0]["stocks"])

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


class MultiAgentVisualizer:
    """Creates portfolio performance graphs as .png in streamlit directory."""

    def __init__(self, data: pd.DataFrame, stock_symbols: list, algo_info: list) -> None:
        """Constructor.

        Args:
            data (pd.DataFrame): Data of given stocks -- includes OHLC data and indicators and indices
            stock_symbols (list(str)): Stock symbols of user defined stocks
            algo_info (list(dict)): Algo info - contains data to visualize performance and share amount
        """
        self.data = data
        self.stock_symbols = stock_symbols
        self.algo_info = algo_info
        self.agent_ids = algo_info[0].keys()
        self.x_axis_sample_amount = self.data.shape[0]

        # gather info metrics
        self.total_values = {agent: [] for agent in self.agent_ids}
        self.portfolio_values = {agent: [] for agent in self.agent_ids}
        self.cash_values = {agent: [] for agent in self.agent_ids}
        self.stock_shares = {agent: [] for agent in self.agent_ids}

        for timestep in self.algo_info:
            for agent, agent_dict in timestep.items():
                self.total_values[agent].append(agent_dict["total"])
                self.portfolio_values[agent].append(agent_dict["portfolio"])
                self.cash_values[agent].append(agent_dict["cash"])
                self.stock_shares[agent].append(agent_dict["stocks"])

    def create_graphs(self) -> None:
        """Calls class to create the different graphs. Resets output directory."""
        if os.path.isdir("./streamlit/result_images"):
            shutil.rmtree("./streamlit/result_images")
        os.makedirs("./streamlit/result_images/performance", exist_ok=True)
        os.makedirs("./streamlit/result_images/shares", exist_ok=True)

        for agent in self.agent_ids:
            self._plot_total_values(agent_id=agent)
            self._plot_portfolio_values(agent_id=agent)
            self._plot_cash(agent_id=agent)
            self._plot_stock_shares(agent_id=agent)

    def _plot_total_values(self, agent_id: str) -> None:
        """Plots the total capital graph -- total capital given by portfolio value + cash.

        Args:
            agent_id (str): Name of agent
        """
        x = np.linspace(0, self.x_axis_sample_amount - 1, self.x_axis_sample_amount)
        y = self.total_values[agent_id]
        plt.plot(x, y)
        plt.title(f"Total {agent_id}")
        plt.savefig(os.path.join("./streamlit/result_images/performance", f"{agent_id}_total.png"))
        plt.close()

    def _plot_portfolio_values(self, agent_id: str) -> None:
        """Plots the portfolio graph.

        Args:
            agent_id (str): Name of agent
        """
        x = np.linspace(0, self.x_axis_sample_amount - 1, self.x_axis_sample_amount)
        y = self.portfolio_values[agent_id]
        plt.plot(x, y)
        plt.title(f"Portfolio {agent_id}")
        plt.savefig(os.path.join("./streamlit/result_images/performance", f"{agent_id}_portfolio.png"))
        plt.close()

    def _plot_cash(self, agent_id: str) -> None:
        """Plots the cash graph.

        Args:
            agent_id (str): Name of agent
        """
        x = np.linspace(0, self.x_axis_sample_amount - 1, self.x_axis_sample_amount)
        y = self.cash_values[agent_id]
        plt.plot(x, y)
        plt.title(f"Cash {agent_id}")
        plt.savefig(os.path.join("./streamlit/result_images/performance", f"{agent_id}_cash.png"))
        plt.close()

    def _plot_stock_shares(self, agent_id: str) -> None:
        """Plots the stock shares graphs.

        Args:
            agent_id (str): Name of agent
        """
        stocks_list = self.stock_shares[agent_id]
        for stock in self.stock_symbols:
            stock_shares = [datapoint[stock] for datapoint in stocks_list]
            x = np.linspace(0, self.x_axis_sample_amount - 1, self.x_axis_sample_amount)
            y = stock_shares
            plt.plot(x, y)
            plt.title(f"{stock} share amount {agent_id}")
            plt.savefig(os.path.join("./streamlit/result_images/shares", f"{agent_id}_{stock}_shares.png"))
            plt.close()
