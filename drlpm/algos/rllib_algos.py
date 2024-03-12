import random
import pandas as pd
from typing import Any
from gymnasium.spaces import Box
from ray.rllib.env import EnvContext
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig
from drlpm.algos.abstract_algo import AbstractAlgo


class RLlibAlgo(AbstractAlgo):
    """Able to call rllib algos by name. Only works for algos with config setup like RllibPPO below.
       Note: Further algos will be added -- until now, only PPO is supported.
    """
    def __init__(self, algo_name: str) -> None:
        """Constructor.

        Args:
            algo_name (str): Name of the algo
        """
        super().__init__(algo_name=algo_name)
        self.algo_name_pyclass_mapping = {
            "PPO": RllibPPO
        }

    def get_algo(self, **kwargs) -> Any:
        """Return the corresponding reinforcement learning algo specified by user.

        Args:
            kwargs (dict):
                env (MultiAgentStockTradingEnv): The Rllib MultiAgentEnv environment for which to create the algo
                env_config (EnvContext):
                    data (pd.DataFrame): Data of given stocks -- includes OHLC data and indicators and indices
                    stock_symbols (list): List of stock as stock-symbols, e.g. 'AAPL'
                    initial_balance (float): Initial account balance
                    agent_ids (list): Names of agents
                config (dict): Dictionary of user defined configurations
                data (pd.DataFrame): Data of given stocks -- includes OHLC data and indicators and indices

        Returns:
            (Any): Respective algo -- defined by user
        """
        return self.algo_name_pyclass_mapping[self.algo_name](env=kwargs["env"],
                                                              env_config=kwargs["env_config"],
                                                              config=kwargs["config"],
                                                              data=kwargs["data"])()


class RllibPPO:
    """PPO for multi agent environment."""
    def __init__(self, env: Any, env_config: EnvContext, config: dict, data: pd.DataFrame) -> None:
        """Constructor.

        Args:
            env (MultiAgentStockTradingEnv): The Rllib MultiAgentEnv environment for which to create the algo
            env_config (EnvContext):
                data (pd.DataFrame): Data of given stocks -- includes OHLC data and indicators and indices
                stock_symbols (list): List of stock as stock-symbols, e.g. 'AAPL'
                initial_balance (float): Initial account balance
                agent_ids (list): Names of agents
            config (dict): Dictionary of user defined configurations
            data (pd.DataFrame): Data of given stocks -- includes OHLC data and indicators and indices
        """
        self.env = env
        self.env_config = env_config
        self.config = config
        # NOTE: obs dim given by portfolio values (3) + positions per stock (nr_stocks) + data
        self.observation_shape = (1, 3 + len(config["stock_symbols"]) + data.shape[1])
        self.boundary = config["initial_balance"] * 100     # applies a factor of 100 to obs Box spaces

    def __call__(self) -> Any:
        """Returns PPO algo.

        Note: The current version is a template on how to implement an algo for the multi agent environment into this
              repository. The agents differ only in random gamma values which does not justify the multi agent
              environment since hyperparameter tuning could achieve the same.
              In order to do some serious stuff, use the high customization options of rllib and add a class here.

        Returns:
            (Any): PPO algo with rllib -- with defined policies
        """
        policies = {}
        for agent in self.config["agent_ids"]:
            policies[agent] = PolicySpec(
                                None, Box(low=-self.boundary, high=self.boundary, shape=self.observation_shape),
                                self.env.action_space, {"gamma": RllibPPO._sample_gamma()}
                              )

        return (PPOConfig()
                .environment(env="marl_env", env_config=self.env_config, disable_env_checking=True)
                .multi_agent(
                    policies=policies,
                    policy_mapping_fn=(lambda aid, episode, **kw: self.config["agent_ids"][int(aid[-1]) - 1]),
                )
                .resources(num_gpus=self.config["num_gpus"])
                .build())

    @staticmethod
    def _sample_gamma() -> float:
        """Samples values for gamma.

        Returns:
            (float): Random number between 0.7 and 1
        """
        return max(0.7, min(random.random(), 1))
