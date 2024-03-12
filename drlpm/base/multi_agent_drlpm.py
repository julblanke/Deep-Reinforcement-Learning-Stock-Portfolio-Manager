import os
import ray
import copy
import logging
import pandas as pd
from typing import Any
from ray.rllib.env import EnvContext
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from drlpm.algos.rllib_algos import RLlibAlgo
from ray.rllib.algorithms.algorithm import Algorithm
from drlpm.utils.visualizer import MultiAgentVisualizer
from drlpm.envs.multi_agent_env import MultiAgentStockTradingEnv


class MultiAgentDrlpm:
    """Class to run multi agent stock portfolio manager."""
    @staticmethod
    def run_multi_agent(config: dict, data: pd.DataFrame) -> None:
        """Run a multi agent environment with rllib algos.

        Args:
            config (dict): User input as yml configuration file
            data (pd.DataFrame): Data of given stocks -- includes OHLC data and indicators and indices
        """
        ray.init()
        logger = logging.getLogger()

        # mapping of algo name to respective python class
        algo_name_pyclass_dict = {
            "PPO": RLlibAlgo
        }

        _env_args = {"data": data, "stock_symbols": config["stock_symbols"],
                     "initial_balance": config["initial_balance"], "agent_ids": config["agent_ids"]}
        env = MultiAgentStockTradingEnv(env_config=_env_args)
        logger.info("Created environment.")

        register_env("marl_env", lambda env_config: MultiAgentStockTradingEnv(env_config=env_config))
        env_config = EnvContext(env_config=_env_args, worker_index=config["worker_index"],
                                num_workers=config["num_workers"])

        algo = algo_name_pyclass_dict[config["algo"]](algo_name=config["algo"]).get_algo(env=env,
                                                                                         env_config=env_config,
                                                                                         config=config,
                                                                                         data=data)
        MultiAgentDrlpm.train_and_eval(algo=algo, config=config, env=env, data=data)

    @staticmethod
    def train_and_eval(algo: Any, config: dict, env: MultiAgentStockTradingEnv, data: pd.DataFrame) -> None:
        """Train and evaluate algo.

        Args:
            algo: Algo to train and evaluate
            config (dict): User input as yml configuration file
            env (MultiAgentStockTradingEnv): The Rllib MultiAgentEnv environment for which to create the algo
            data (pd.DataFrame): Data of given stocks -- includes OHLC data and indicators and indices
        """
        logger = logging.getLogger()

        # train
        path_to_checkpoint = "./rllib_checkpoints/"
        if config["algo_reload"]:
            algo = Algorithm.from_checkpoint(path_to_checkpoint)
        else:
            timesteps_total = 0
            user_timesteps = config["train_timesteps"]
            os.makedirs(path_to_checkpoint, exist_ok=True)
            while timesteps_total < user_timesteps:
                result = algo.train()
                timesteps_total = result["timesteps_total"]
                print(pretty_print(result))
                algo.save(checkpoint_dir=path_to_checkpoint)

        # eval
        evaluation_results = []
        obs, info = env.reset()
        terminateds = {"__all__": False}
        while not all(terminateds.values()):
            action_dict = {}
            for i, agent in enumerate(config["agent_ids"]):
                action = algo.compute_single_action(
                    observation=obs[agent],
                    policy_id=agent,  # <- default value
                )
                action_dict[agent] = action
            obs, rewards, terminateds, truncateds, infos = env.step(action_dict=action_dict)
            infos_copy = copy.deepcopy(infos)   # prevents "stocks" in nested dict being overwritten
            evaluation_results.append(infos_copy)

        algo.stop()
        ray.shutdown()
        logger.info("Finished training algo.")

        (MultiAgentVisualizer(data=data, stock_symbols=config["stock_symbols"], algo_info=evaluation_results)
         .create_graphs())
        logger.info("Created graphs.")
