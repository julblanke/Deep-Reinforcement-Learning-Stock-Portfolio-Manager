import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import typer
import logging
import torch as th
from typing import List
from stable_baselines3 import PPO
from drlpm.utils.logger import Logger
from drlpm.utils.visualizer import Visualizer
from drlpm.envs.stock_env import StockTradingEnv
from drlpm.data_processing.dataloader import Dataloader


DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")


class DrlPortfolioManager:
    """Deep Reinforcement Learning Portfolio Manager."""

    @staticmethod
    def run(stock_symbols: list, initial_balance: float, train_timesteps: int, period: str,
            interval: str, update_data: bool) -> None:
        """Run the portfolio manager.

        Args:
            stock_symbols (list): List of stock as stock-symbols, e.g. 'AAPL'
            initial_balance (float): Initial account balance
            train_timesteps (int): Number of model time steps for training
            period (str): Time period for data to take into account -- in yfinance terms -- e.g. '2y'
            interval (str): Data points frequency -- in yfinance terms -- e.g. '1d'
            update_data (bool): Whether to update stock data
        """
        Logger.initialize_logger()
        logger = logging.getLogger()

        # load data and create environment
        data = (Dataloader(stock_symbols=stock_symbols,
                           period=period,
                           interval=interval,
                           update_data=update_data)
                .get_data())
        logger.info("Loading data finished.")

        env = StockTradingEnv(data=data,
                              stock_symbols=stock_symbols,
                              initial_balance=initial_balance)
        logger.info("Created environment.")

        # create and train ppo model
        model = PPO('MlpPolicy',
                    env=env,
                    verbose=1,
                    device=DEVICE,
                    tensorboard_log="./logs")
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
        logger.info("Done!")

def main(stock_symbols: List[str] = typer.Argument(..., help="Define stock symbols to be considered for portfolio."),
         initial_balance: float = typer.Option(..., help="Initial account balance."),
         train_timesteps: int = typer.Option(..., help="Number of model time steps for training."),
         period: str = typer.Option(..., help="Define time period to take into account."),
         interval: str = typer.Option(..., help="Define time interval of stock values."),
         update_data: bool = typer.Option(..., help="Update stock data of given stocks.")
         ) -> None:
    """Main func for typer.

    Args:
        stock_symbols (list): List of stock as stock-symbols, e.g. 'AAPL'
        initial_balance (float): Initial account balance
        train_timesteps (int): Number of model time steps for training
        period (str): Time period for data to take into account -- in yfinance terms -- e.g. '2y'
        interval (str): Data points frequency -- in yfinance terms -- e.g. '1d'
        update_data (bool): Whether to update stock data
    """
    DrlPortfolioManager.run(stock_symbols=stock_symbols,
                            initial_balance=initial_balance,
                            train_timesteps=train_timesteps,
                            period=period,
                            interval=interval,
                            update_data=update_data)


if __name__ == "__main__":
    typer.run(main)
