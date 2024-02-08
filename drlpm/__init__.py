from gymnasium.envs.registration import register

register(
     id="StockTradingEnv-v0",
     entry_point="drlpm.envs:StockTradingEnv",
     max_episode_steps=40,
)
