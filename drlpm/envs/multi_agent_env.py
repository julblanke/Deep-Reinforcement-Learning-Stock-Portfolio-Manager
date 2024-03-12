import numpy as np
from ray.rllib import MultiAgentEnv
from gymnasium.spaces import Box, Dict


class MultiAgentStockTradingEnv(MultiAgentEnv):
    """Multi Agent RL Environment for stock trading portfolio manager.

    Currently, the environment tracks cash, portfolio value, number of shares per stock and a total capital.
    The environment takes as input these parameters for every agent individually as well as an observation per time step
    consisting of the Open, High, Low and Close values of every stock given, their respective sma and ema with window
    size of 20, 50, 100, 200 and the following list of indices:
    ["SPY", "QQQ", "SMH", "XLV", "XLP", "XLE", "XLF", "XLI", "XLU", "XLB", "XLK", "KRE"]

    A continuous action vector of shape [1, 2 * nr_stocks] with values having a range of [0, 1] represents
    the buy and sell action of every stock (e.g. [AAPL_buy, AAPL_sell, MSFT_buy, MSFT_sell] for AAPL and MSFT)
    for every agent.
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

    def __init__(self, env_config) -> None:
        """Constructor.

        Args:
            env_config (dict):
                data (pd.DataFrame): Data of given stocks -- includes OHLC data and indicators and indices
                stock_symbols (list(str)): Stock symbols of user defined stocks
                initial_balance (float): Initial account balance
                agent_ids (list): Names of agents
        """
        super().__init__()

        # static
        self.config = env_config
        self.data = self.config["data"]
        self.stock_symbols = self.config["stock_symbols"]
        self.initial_balance = self.config["initial_balance"]
        self.nr_stock_symbols = len(self.config["stock_symbols"])
        self.n_samples = self.config["data"].shape[0]
        self._agent_ids = self.config["agent_ids"]
        # define spaces
        self.action_space = Box(low=np.zeros(2 * self.nr_stock_symbols),
                                high=np.ones(2 * self.nr_stock_symbols),
                                dtype=np.float32)

        # NOTE: obs dim given by portfolio values (3) + positions per stock (nr_stocks) + data
        self._obs_space_in_preferred_format = True
        observation_dim = 3 + self.nr_stock_symbols + self.data.shape[1]
        self.observation_space = Dict({agent: Box(low=0, high=1000000, shape=(1, observation_dim))
                                      for agent in self._agent_ids})

        # initial values
        self.state = {agent: self.initial_balance for agent in self._agent_ids}
        self.portfolio_value = {agent: 0.0 for agent in self._agent_ids}
        self.cash = {agent: self.initial_balance for agent in self._agent_ids}
        self.owned_stocks = {agent: dict() for agent in self._agent_ids}
        for agent, dict_ in self.owned_stocks.items():
            for stock in self.stock_symbols:
                dict_[stock] = 0
        self.current_step = 0
        self.current_sequence = None

        # memory
        self.last_state_memory = {agent: self.initial_balance for agent in self._agent_ids}

    def reset(self, *, seed=None, options=None) -> (dict, dict):
        """Environment reset function.

        Args:
            seed (int): Random integer for reproducibility -- not used in this env
            options (dict): Additional information to specify how the environment is reset -- not used in this env

        Returns:
            observations (dict): Portfolio parameters + stocks Open, High, Low and Close values for every agent
            infos (dict): Currently empty dict for every agent -- applied due to signature
        """
        self.current_step = 0

        # reset envs parameters
        self.state = {agent: self.initial_balance for agent in self._agent_ids}
        self.portfolio_value = {agent: 0.0 for agent in self._agent_ids}
        self.cash = {agent: self.initial_balance for agent in self._agent_ids}
        self.last_state_memory = {agent: self.initial_balance for agent in self._agent_ids}
        self.owned_stocks = {agent: dict() for agent in self._agent_ids}
        for agent, dict_ in self.owned_stocks.items():
            for stock in self.stock_symbols:
                dict_[stock] = 0

        observations, sequence = self._get_obs()
        infos = {agent: {} for agent in self._agent_ids}

        return observations, infos

    def step(self, action_dict: dict) -> (dict, dict, dict, dict, dict):
        """RL step function.

        Args:
            action_dict (dict): Action vector in continuous space of dim [2 * nr_stocks, 1] in range [0, 1]
                                as dict for every agent -- key: agent_id, value: action_vector

        Returns:
            observations (dict): Portfolio parameters + stocks Open, High, Low and Close values
            rewards (dict): Reward; given by Total value minus previous step Total value
            terminateds (dict): Bool if env hits defined termination condition
            truncateds (dict): Bool if env is terminated by externally defined condition
            infos (dict): Contains information about portfolio values
        """
        # get current share prices
        observations, sequence = self._get_obs()
        current_share_price_all_stocks = dict()
        for stock in self.stock_symbols:
            current_share_price_all_stocks[stock] = (float(sequence[f"Close_{stock}"]))

        # take actions
        self._take_action(action_dict=action_dict,
                          current_share_price_all_stocks=current_share_price_all_stocks,
                          stock_symbols=self.stock_symbols)

        # define reward function
        rewards = {agent: (self.state[agent] - self.last_state_memory[agent]) for agent in self._agent_ids}
        self.last_state_memory = {agent: self.state[agent] for agent in self._agent_ids}

        # define "terminateds"
        terminateds = {}
        if self.current_step == (self.n_samples - 1):
            terminateds["__all__"] = True
        else:
            terminateds["__all__"] = False

        # define "truncateds"
        truncateds = {"__all__": False}

        # define "info"
        info = {agent: {
            "total": self.state[agent],
            "portfolio": self.portfolio_value[agent],
            "cash": self.cash[agent],
            "stocks": self.owned_stocks[agent]
        } for agent in self._agent_ids}

        # update episode
        self.current_step += 1

        return observations, rewards, terminateds, truncateds, info

    def render(self) -> None:
        """Render function."""
        pass

    def _take_action(self, action_dict: dict, current_share_price_all_stocks: dict, stock_symbols: list) -> None:
        """Acts on the continuous action vector to buy and sell positions and update portfolio values.

        Args:
            action_dict (dict): Action vector in continuous space of dim [2 * nr_stocks, 1] in range [0, 1]
                            as dict for every agent -- key: agent_id, value: action_vector
            current_share_price_all_stocks (dict): Current share price of given stocks
            stock_symbols (list): Stock symbols of user defined stocks
        """
        for agent in self._agent_ids:
            # check if buying and selling exceeds capital by sum current sell price and compare to sum of buy prices
            sell_values_action_vector = action_dict[agent][1::2]
            buy_values_action_vector = action_dict[agent][::2]

            # create dicts for easier access
            sell_values_action_vector_dict = {}
            buy_values_action_vector_dict = {}
            for i in range(len(stock_symbols)):
                sell_values_action_vector_dict[stock_symbols[i]] = sell_values_action_vector[i]
                buy_values_action_vector_dict[stock_symbols[i]] = buy_values_action_vector[i]

            # sum sell power
            total_sell_capital = 0
            for stock, n_shares in self.owned_stocks[agent].items():
                sell_capital_stock = (current_share_price_all_stocks[stock] * n_shares *
                                      sell_values_action_vector_dict[stock])
                total_sell_capital += sell_capital_stock
            potential_buy_capital = total_sell_capital + self.cash[agent]

            # total necessary buy capital
            necessary_buy_capital = potential_buy_capital * sum(buy_values_action_vector)

            # define actual action step
            if necessary_buy_capital > potential_buy_capital:
                pass
            else:
                # sell shares
                for stock, n_shares in self.owned_stocks[agent].items():
                    amount_invested = current_share_price_all_stocks[stock] * n_shares
                    sell_amount = amount_invested * sell_values_action_vector_dict[stock]
                    new_amount_invested = amount_invested - sell_amount
                    new_share_amount = new_amount_invested / current_share_price_all_stocks[stock]
                    self.owned_stocks[agent][stock] = new_share_amount
                    self.cash[agent] += sell_amount

                # buy shares
                for stock, n_shares in self.owned_stocks[agent].items():
                    buy_amount = buy_values_action_vector_dict[stock] * self.cash[agent]
                    new_n_shares = n_shares + buy_amount / current_share_price_all_stocks[stock]
                    self.owned_stocks[agent][stock] = new_n_shares

                # update cash after buying
                self.cash[agent] -= necessary_buy_capital

            # update portfolio values
            total_portfolio_value = 0
            for stock, n_shares in self.owned_stocks[agent].items():
                total_portfolio_value += current_share_price_all_stocks[stock] * n_shares
            self.portfolio_value[agent] = total_portfolio_value
            self.state[agent] = total_portfolio_value + self.cash[agent]

    def _get_obs(self) -> (dict, np.ndarray):
        """Returns observations and sequence based on current time step.

        Returns:
            observations (dict): Current observation -- portfolio parameters + stocks Open, High, Low and Close values
                                 as dict for every agent -- key: agent_id, value: observation (np.ndarray)
            sequence (np.ndarray): Current Open, High, Low and Close values of stocks
        """
        sequence = self.data.iloc[self.current_step, :].T
        sequence_flattened = self.data.iloc[self.current_step, :].to_numpy()
        observations = {}
        for agent in self._agent_ids:
            general_info = np.array([self.state[agent], self.portfolio_value[agent], self.cash[agent]]
                                    + list(self.owned_stocks[agent].values()))
            obs = np.concatenate([general_info, sequence_flattened])
            observations[agent] = np.array(obs).reshape(1, -1)
        return observations, sequence
