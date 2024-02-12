import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import typer
import logging
import torch as th
from typing import List
from drlpm.utils.logger import Logger
from drlpm.base.dataloader import Dataloader
from drlpm.envs.stock_env import StockTradingEnv
from drlpm.base.train_and_eval import TrainAndEval
from drlpm.agents.stable_baselines_models import StableBaselinesModels


DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")


class DrlPortfolioManager:
    """Deep Reinforcement Learning Portfolio Manager."""

    @staticmethod
    def run(model: str, stock_symbols: list, initial_balance: float, train_timesteps: int, period: str,
            interval: str, update_data: bool) -> None:
        """Run the portfolio manager.

        Args:
            model (str): Reinforcement learning model to use
            stock_symbols (list): List of stock as stock-symbols, e.g. 'AAPL'
            initial_balance (float): Initial account balance
            train_timesteps (int): Number of model time steps for training
            period (str): Time period for data to take into account -- in yfinance terms -- e.g. '2y'
            interval (str): Data points frequency -- in yfinance terms -- e.g. '1d'
            update_data (bool): Whether to update stock data
        """
        Logger.initialize_logger()
        logger = logging.getLogger()

        # mapping of model name to respective python class
        model_name_pyclass_dict = {
            "PPO": StableBaselinesModels,
            "A2C": StableBaselinesModels,
            "DDPG": StableBaselinesModels,
            "SAC": StableBaselinesModels,
            "TD3": StableBaselinesModels
        }

        # load data
        data = (Dataloader(stock_symbols=stock_symbols,
                           period=period,
                           interval=interval,
                           update_data=update_data)
                .get_data())
        logger.info("Loading data finished.")

        # create environment
        env = StockTradingEnv(data=data,
                              stock_symbols=stock_symbols,
                              initial_balance=initial_balance)
        logger.info("Created environment.")

        # create model
        model = model_name_pyclass_dict[model](model_name=model).get_model(env=env,
                                                                           verbose=1,
                                                                           device=DEVICE,
                                                                           tensorboard_log="./logs")

        # train and eval
        TrainAndEval.train_and_eval(model=model,
                                    stock_symbols=stock_symbols,
                                    data=data,
                                    train_timesteps=train_timesteps)

        logger.info("Done!")


def main(stock_symbols: List[str] = typer.Argument(..., help="Define stock symbols to be considered for portfolio."),
         model: str = typer.Option(..., help="Choose reinforcement learning model. Currently supported: \n"
                                             "PPO, A2C, DDPG, SAC, TD3"),
         initial_balance: float = typer.Option(..., help="Initial account balance."),
         train_timesteps: int = typer.Option(..., help="Number of model time steps for training."),
         period: str = typer.Option(..., help="Define time period to take into account."),
         interval: str = typer.Option(..., help="Define time interval of stock values."),
         update_data: bool = typer.Option(..., help="Update stock data of given stocks.")
         ) -> None:
    """Main func for typer.

    Args:
        stock_symbols (list): List of stock as stock-symbols, e.g. 'AAPL'
        model (str): Reinforcement learning model to use
        initial_balance (float): Initial account balance
        train_timesteps (int): Number of model time steps for training
        period (str): Time period for data to take into account -- in yfinance terms -- e.g. '2y'
        interval (str): Data points frequency -- in yfinance terms -- e.g. '1d'
        update_data (bool): Whether to update stock data
    """
    DrlPortfolioManager.run(model=model,
                            stock_symbols=stock_symbols,
                            initial_balance=initial_balance,
                            train_timesteps=train_timesteps,
                            period=period,
                            interval=interval,
                            update_data=update_data)


if __name__ == "__main__":
    typer.run(main)
