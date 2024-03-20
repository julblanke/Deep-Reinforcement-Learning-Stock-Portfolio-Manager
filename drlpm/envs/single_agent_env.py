import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class SingleAgentStockTradingEnv(gym.Env):
    """RL Environment for stock trading portfolio manager.

    Currently, the environment tracks cash, portfolio value, number of shares per stock and a total capital.
    The environment takes as input these parameters as well as an observation per time step consisting of the
    Open, High, Low and Close values of every stock given, their respective sma and ema with window size of
    20, 50, 100, 200 and the following list of indices:
    ["SPY", "QQQ", "SMH", "XLV", "XLP", "XLE", "XLF", "XLI", "XLU", "XLB", "XLK", "KRE"]

    A continuous action vector of shape [1, 2 * nr_stocks] with values having a range of [0, 1] represents
    the buy and sell action of every stock (e.g. [AAPL_buy, AAPL_sell, MSFT_buy, MSFT_sell] for AAPL and MSFT).
    The buy factor in range [0, 1] decides, how much money of the current cash is used to buy the respective stock.
    The sell factor in range [0, 1] decides, how much money of the current investment in the respective stock is sold.
    E.g.: If action vector for [AAPL_buy, AAPL_sell, MSFT_buy, MSFT_sell] is [0.5, 0, 0.25, 0.5]:
            1) All the stocks buy factors will be added together and nothing will happen if sum is above 1.
               Reason being, that the cash will not be updated after every buy of a stock since this leads to
               higher weight on the first stock that gets processed because of a reducing cash amount the further
               we continue the buy actions (because: buy_amount_stock = buy_factor_stock * cash).
            2) 0% of owned stocks in AAPL will be sold
            3) 50% of owned stocks of MSFT will be sold; money will instantly be added to cash
            4) If sum of factors are below or equal 1, every stock will buy with the respective buy factor on the
               total cash available during the time step. Hence, 0.5 * cash will flow into AAPL and 0.25 * cash will
               flow into MSFT. Afterward, cash is updated, and we are left with 0.25% of the previous cash.
    """
    def __init__(self, data: pd.DataFrame, stock_symbols: list, initial_balance: float) -> None:
        """Constructor.

        Args:
            data (pd.DataFrame): Data of given stocks -- includes OHLC data and indicators and indices
            stock_symbols (list(str)): Stock symbols of user defined stocks
            initial_balance (float): Initial account balance
        """
        # static
        self.render_mode = "human"
        self.data = data
        self.stock_symbols = stock_symbols
        self.initial_balance = initial_balance
        self.nr_stock_symbols = len(stock_symbols)
        self.n_samples = data.shape[0]

        # define spaces
        self.action_space = spaces.Box(low=np.zeros(2 * self.nr_stock_symbols),
                                       high=np.ones(2 * self.nr_stock_symbols),
                                       dtype=np.float16)
        # NOTE: obs dim given by portfolio values (3) + positions per stock (nr_stocks) + data
        observation_dim = 3 + self.nr_stock_symbols + data.shape[1]
        self.observation_space = spaces.Box(low=0, high=100000000, shape=(1, observation_dim))

        # initial values
        self.state = self.initial_balance
        self.portfolio_value = 0.0
        self.cash = self.initial_balance
        self.owned_stocks = dict()
        for stock in self.stock_symbols:
            self.owned_stocks[stock] = 0
        self.current_step = 0
        self.current_sequence = None

        # memory
        self.last_state_memory = self.initial_balance

    def reset(self, seed: int = None, options: dict = None) -> (np.ndarray, np.ndarray):
        """Environment reset function.

        Args:
            seed (int): Random integer for reproducibility -- not used in this env
            options (dict): Additional information to specify how the environment is reset -- not used in this env

        Returns:
            observation (np.ndarray): Portfolio parameters + stocks Open, High, Low and Close values
            info (dict): Contains information about portfolio values
        """
        self.current_step = 0

        # reset envs parameters
        self.state = self.initial_balance
        self.portfolio_value = 0.0
        self.cash = self.initial_balance
        self.last_state_memory = self.initial_balance
        self.owned_stocks = dict()
        for stock in self.stock_symbols:
            self.owned_stocks[stock] = 0

        observation, sequence = self._get_obs()
        info = {}

        return observation, info

    def step(self, action_vector: np.array) -> (np.ndarray, float, bool, dict, dict):
        """RL step function.

        Args:
            action_dict (dict): Action vector in continuous space of dim [1, 2 * nr_stocks] in range [0, 1]

        Returns:
            observation (np.ndarray): Portfolio parameters + stocks Open, High, Low and Close values
            reward (float): Reward; given by Total value minus previous step Total value
            terminated (bool): Bool if env hits defined termination condition
            info (dict): Contains information about portfolio values
        """
        # get current share prices
        observation, sequence = self._get_obs()
        current_share_price_all_stocks = dict()
        for stock in self.stock_symbols:
            current_share_price_all_stocks[stock] = (float(sequence[f"Close_{stock}"]))

        # take actions
        self._take_action(action_vector=action_vector,
                          current_share_price_all_stocks=current_share_price_all_stocks,
                          stock_symbols=self.stock_symbols)

        # define reward function
        reward = self.state - self.last_state_memory
        self.last_state_memory = self.state

        # define "terminated"
        if self.current_step == (self.n_samples - 1):
            terminated = True
        else:
            terminated = False

        # define "info"
        info = {
            "total": self.state,
            "portfolio": self.portfolio_value,
            "cash": self.cash,
            "stocks": self.owned_stocks
        }

        # update episode
        self.current_step += 1

        return observation, reward, terminated, False, info

    def render(self, mode="human") -> None:
        """Render function.

        Args:
            mode (str): Rendering mode -- not used in this env
        """
        pass

    def _take_action(self, action_vector: np.array, current_share_price_all_stocks: dict, stock_symbols: list) -> None:
        """Acts on the continuous action vector to buy and sell positions and update portfolio values.

        Args:
            action_dict (dict): Action vector in continuous space of dim [1, 2 * nr_stocks] in range [0, 1]
            current_share_price_all_stocks (dict): Current share price of given stocks
            stock_symbols (list): Stock symbols of user defined stocks
        """
        # check if buying and selling exceeds capital by sum current sell price and compare to sum of buy prices
        sell_values_action_vector = action_vector[1::2]
        buy_values_action_vector = action_vector[::2]

        # create dicts for easier access
        sell_values_action_vector_dict = {}
        buy_values_action_vector_dict = {}
        for i in range(len(stock_symbols)):
            sell_values_action_vector_dict[stock_symbols[i]] = sell_values_action_vector[i]
            buy_values_action_vector_dict[stock_symbols[i]] = buy_values_action_vector[i]

        # sum sell power
        total_sell_capital = 0
        for stock, n_shares in self.owned_stocks.items():
            sell_capital_stock = current_share_price_all_stocks[stock] * n_shares * sell_values_action_vector_dict[stock]
            total_sell_capital += sell_capital_stock
        potential_buy_capital = total_sell_capital + self.cash

        # total necessary buy capital
        necessary_buy_capital = potential_buy_capital * sum(buy_values_action_vector)

        # define actual action step
        if necessary_buy_capital > potential_buy_capital:
            pass
        else:
            # sell shares
            for stock, n_shares in self.owned_stocks.items():
                amount_invested = current_share_price_all_stocks[stock] * n_shares
                sell_amount = amount_invested * sell_values_action_vector_dict[stock]
                new_amount_invested = amount_invested - sell_amount
                new_share_amount = new_amount_invested / current_share_price_all_stocks[stock]
                self.owned_stocks[stock] = new_share_amount
                self.cash += sell_amount

            # buy shares
            for stock, n_shares in self.owned_stocks.items():
                buy_amount = buy_values_action_vector_dict[stock] * self.cash
                new_n_shares = n_shares + buy_amount / current_share_price_all_stocks[stock]
                self.owned_stocks[stock] = new_n_shares

            # update cash after buying
            self.cash -= necessary_buy_capital

        # update portfolio values
        total_portfolio_value = 0
        for stock, n_shares in self.owned_stocks.items():
            total_portfolio_value += current_share_price_all_stocks[stock] * n_shares
        self.portfolio_value = total_portfolio_value
        self.state = total_portfolio_value + self.cash

    def _get_obs(self) -> (np.ndarray, np.ndarray):
        """Returns observation and sequence based on current time step.

        Returns:
            obs (np.ndarray): Current observation -- portfolio parameters + stocks Open, High, Low and Close values
            sequence (np.ndarray): Current Open, High, Low and Close values of stocks
        """
        sequence = self.data.iloc[self.current_step, :].T
        sequence_ndarray = self.data.iloc[self.current_step, :].to_numpy()
        general_info = np.array([self.state, self.portfolio_value, self.cash] + list(self.owned_stocks.values()))
        obs = np.concatenate([general_info, sequence_ndarray])
        return obs, sequence
