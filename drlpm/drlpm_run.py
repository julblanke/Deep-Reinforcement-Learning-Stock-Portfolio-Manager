import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import typer
import logging
import torch as th
from drlpm.utils.logger import Logger
from drlpm.utils.config_reader import ConfigReader
from drlpm.base.multi_agent_drlpm import MultiAgentDrlpm
from drlpm.base.single_agent_drlpm import SingleAgentDrlpm
from drlpm.base.data_processing.dataloader import Dataloader


DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")


class DrlPortfolioManager:
    """Deep Reinforcement Learning Portfolio Manager."""

    @staticmethod
    def run(config_path: str = "./examples/multi_agent.yaml") -> None:
        """Run the portfolio manager.

        Args:
            config_path (str): Path to yaml config file
        """
        Logger.initialize_logger()
        logger = logging.getLogger()
        config = ConfigReader.execute(config_path=config_path)

        data = (Dataloader(stock_symbols=config["stock_symbols"],
                           period=config["period"],
                           interval=config["interval"],
                           update_data=config["update_data"])
                .get_data())
        logger.info("Loading data finished.")

        if config["drlpm_type"] == "single_agent":
            SingleAgentDrlpm.run_single_agent(config=config, data=data, device=DEVICE)
        elif config["drlpm_type"] == "multi_agent":
            MultiAgentDrlpm.run_multi_agent(config=config, data=data)
        else:
            raise KeyError(f"{config['drlpm_type']} not known."
                           f" Only 'single_agent' and 'multi_agent' known.")
        logger.info("Done!")


def main(config_path: str = typer.Option(..., help="Define path to yml config file.")) -> None:
    """Main func for typer.

    Args:
        config_path (str): Path to yaml config file
    """
    DrlPortfolioManager.run(config_path=config_path)


if __name__ == "__main__":
    typer.run(main)
