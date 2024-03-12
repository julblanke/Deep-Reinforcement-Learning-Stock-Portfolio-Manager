import logging
import torch as th
import pandas as pd
from drlpm.utils.visualizer import SingleAgentVisualizer
from drlpm.envs.single_agent_env import SingleAgentStockTradingEnv
from drlpm.algos.stable_baselines_algos import StableBaselinesAlgos


class SingleAgentDrlpm:
    """Class to run single agent stock portfolio manager."""
    @staticmethod
    def run_single_agent(config: dict, data: pd.DataFrame, device: th.device) -> None:
        """Run a single agent environment with stable baselines 3 algos.

        Args:
            config (dict): User input as yml configuration file
            data (pd.DataFrame): Data of given stocks -- includes OHLC data and indicators and indices
            device (th.device): Device to run on
        """
        logger = logging.getLogger()

        # create environment
        env = SingleAgentStockTradingEnv(data=data,
                                         stock_symbols=config["stock_symbols"],
                                         initial_balance=config["initial_balance"])
        logger.info("Created environment.")

        # create algo
        algo = StableBaselinesAlgos(algo_name=config["algo"]).get_algo(env=env,
                                                                       verbose=1,
                                                                       device=device,
                                                                       tensorboard_log="./logs")

        # train and eval
        SingleAgentDrlpm.train_and_eval(algo=algo,
                                        stock_symbols=config["stock_symbols"],
                                        data=data,
                                        train_timesteps=config["train_timesteps"])

    @staticmethod
    def train_and_eval(algo, stock_symbols: list, data: pd.DataFrame, train_timesteps: int) -> None:
        """Train and evaluate algo.

        Args:
            algo: Algo to train and evaluate
            stock_symbols (list): List of stock as stock-symbols, e.g. 'AAPL'
            data (pd.DataFrame): Dataframe with OHLC of stocks with indicators and additional indices
            train_timesteps (int): Number of algo time steps for training
        """
        logger = logging.getLogger()

        algo.learn(total_timesteps=train_timesteps)
        logger.info("Finished training algo.")

        vec_env = algo.get_env()
        obs = vec_env.reset()
        states = None
        vec_env.reset()
        algo_info = list()
        while True:
            action, states = algo.predict(obs,
                                          state=states,
                                          deterministic=True)
            obs, rewards, terminated, info = vec_env.step(action)

            # store for visualization
            algo_info.append(info)

            if terminated:
                print("info", info)
                (SingleAgentVisualizer(data=data,
                                       stock_symbols=stock_symbols,
                                       algo_info=algo_info)
                 .create_graphs())
                logger.info("Created graphs.")
                break
