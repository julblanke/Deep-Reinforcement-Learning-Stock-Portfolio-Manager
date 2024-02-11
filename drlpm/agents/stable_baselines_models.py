import gymnasium as gym
from typing import Union
from drlpm.agents.abstract_model import AbstractModel
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3


class StableBaselinesModels(AbstractModel):
    """Able to call stable baselines 3 models by name."""

    def __init__(self, model: str) -> None:
        """Constructor.

        Args:
            model (str): Name of the model
        """
        super().__init__(model=model)
        self.model_name_pyclass_dict = {
            "PPO": PPO,
            "A2C": A2C,
            "DDPG": DDPG,
            "SAC": SAC,
            "TD3": TD3
        }

    def get_model(self, env: gym.Env, verbose: int, device: str, tensorboard_log: str)\
            -> Union[PPO, A2C, DDPG, SAC, TD3]:
        """Return the corresponding reinforcement learning model specified by user.

        Args:
            env (gym.Env): The Gym environment for which to create the model
            verbose (int): Verbosity level for logging messages during training
            device (str): The device on which to perform computations ('cpu' or 'cuda')
            tensorboard_log (str): The directory path where TensorBoard logs will be saved

        Returns:
            Union(PPO, A2C, DDPG, SAC, TD3): Respective model -- defined by user
        """
        return self.model_name_pyclass_dict[self.model]('MlpPolicy',
                                                        env=env,
                                                        verbose=verbose,
                                                        device=device,
                                                        tensorboard_log=tensorboard_log)
