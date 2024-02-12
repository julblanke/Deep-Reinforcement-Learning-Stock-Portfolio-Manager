import gymnasium as gym
from abc import ABC, abstractmethod


class AbstractModel(ABC):
    """Abstract base class for reinforcement learning models."""

    @abstractmethod
    def __init__(self, model_name: str) -> None:
        """Constructor.

        Args:
            model_name (str): Name of the model
        """
        self.model_name = model_name

    @abstractmethod
    def get_model(self, env: gym.Env, verbose: int, device: str, tensorboard_log: str) -> None:
        """Return the corresponding reinforcement learning model specified by user.

        Args:
            env (gym.Env): The Gym environment for which to create the model
            verbose (int): Verbosity level for logging messages during training
            device (str): The device on which to perform computations ('cpu' or 'cuda')
            tensorboard_log (str): The directory path where TensorBoard logs will be saved
        """
        pass
