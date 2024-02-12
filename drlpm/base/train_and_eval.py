import logging
import pandas as pd
from drlpm.utils.logger import Logger
from drlpm.utils.visualizer import Visualizer


class TrainAndEval:
    """Trains and evaluates model."""

    @staticmethod
    def train_and_eval(model, stock_symbols: list, data: pd.DataFrame, train_timesteps: int) -> None:
        """Train and evaluate model.

        Args:
            model: Model to train and evaluate
            stock_symbols (list): List of stock as stock-symbols, e.g. 'AAPL'
            data (pd.DataFrame): Dataframe with OHLC of stocks with indicators and additional indices
            train_timesteps (int): Number of model time steps for training
        """
        Logger.initialize_logger()
        logger = logging.getLogger()

        model.learn(total_timesteps=train_timesteps)
        logger.info("Finished training model.")

        vec_env = model.get_env()
        obs = vec_env.reset()
        states = None
        vec_env.reset()
        model_info = list()
        while True:
            action, states = model.predict(obs,
                                           state=states,
                                           deterministic=True)
            obs, rewards, terminated, info = vec_env.step(action)

            # store for visualization
            model_info.append(info)

            if terminated:
                print("info", info)
                (Visualizer(data=data,
                            stock_symbols=stock_symbols,
                            model_info=model_info)
                 .create_graphs())
                logger.info("Created graphs.")
                break
