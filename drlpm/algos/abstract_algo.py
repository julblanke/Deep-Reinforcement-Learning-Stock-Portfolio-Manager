from typing import Any
from abc import ABC, abstractmethod


class AbstractAlgo(ABC):
    """Abstract base class for reinforcement learning algos."""

    @abstractmethod
    def __init__(self, algo_name: str) -> None:
        """Constructor.

        Args:
            algo_name (str): Name of the algo
        """
        self.algo_name = algo_name

    @abstractmethod
    def get_algo(self, **kwargs) -> Any:
        """Return the corresponding reinforcement learning algo specified by user.

        Args:
            kwargs (dict): Contains the parameters of the reinforcement learning algos
        """
        pass
