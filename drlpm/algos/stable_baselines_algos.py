import importlib
from typing import Union
from drlpm.algos.abstract_algo import AbstractAlgo
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3


class StableBaselinesAlgos(AbstractAlgo):
    """Able to call stable baselines 3 algos by name."""

    def __init__(self, algo_name: str) -> None:
        """Constructor.

        Args:
            algo_name (str): Name of the algo
        """
        super().__init__(algo_name=algo_name)
        module = importlib.import_module("stable_baselines3")
        self.algo_class = getattr(module, algo_name)

    def get_algo(self, **kwargs) -> Union[PPO, A2C, DDPG, SAC, TD3]:
        """Return the corresponding reinforcement learning algo specified by user.

        Args:
            kwargs (dict):
                env (gym.Env): The Gym environment for which to create the algo
                verbose (int): Verbosity level for logging messages during training
                device (str): The device on which to perform computations ('cpu' or 'cuda')
                tensorboard_log (str): The directory path where TensorBoard logs will be saved

        Returns:
            Union(PPO, A2C, DDPG, SAC, TD3): Respective algo -- defined by user
        """
        return self.algo_class('MlpPolicy',
                               env=kwargs["env"],
                               verbose=kwargs["verbose"],
                               device=kwargs["device"],
                               tensorboard_log=kwargs["tensorboard_log"])
