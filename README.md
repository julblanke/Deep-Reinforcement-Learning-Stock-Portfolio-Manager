# Deep-Reinforcement-Learning-Stock-Portfolio-Manager
![pixlr_banner](./static/pixlr_banner.png)
*Source: pixlr image generator*
 
Welcome to the Stock-Trading-Portfolio-Manager with Deep-Reinforcement-Learning! 
This project, developed in my free time, aims to provide a comprehensive solution for 
managing stock trading portfolios using reinforcement learning techniques. Unlike 
traditional portfolio managers, this repository differentiates itself by offering a 
flexible framework that can accommodate an arbitrary number of stocks and features a 
continuous action space, making it suitable for a wide range of trading strategies.

It supports a couple of stable baselines 3 models for single-agent as well as a basic
multi-agent setting with PPO. The Rllib framework used for MARL is highly customizable,
therefore my goal was to set up a modular codebase to somewhat easily add custom
multi-agent settings fitting the individual purpose rather than providing a wide
range of different algos and policies.

## Key Features

- **Flexible Stock Selection**: The portfolio manager can accommodate an arbitrary number of stocks, allowing users to customize their portfolios based on their preferences and investment strategies.

- **Continuous Action Space**: The reinforcement learning framework features a continuous action space with dimensions `[1, (buy, sell) * number_of_stocks]`, enabling the buying and selling of any available amount of stocks at any given time step. This provides users with fine-grained control over trading decisions.

- **Variable Time Period**: Users have the flexibility to specify the time period for which historical market data will be considered, allowing for analysis and optimization over different time horizons.

- **Variable Data Points Interval**: The framework supports variable data points intervals, enabling users to customize the granularity of historical market data used for training and evaluation.

- **Lightweight Codebase for Customization**: The codebase is designed to be lightweight and easily customizable, making it straightforward for users to extend and modify the functionality to suit their specific requirements and trading strategies.

- **Graphical User Interface (GUI) with Streamlit**: The project includes a graphical user interface built with Streamlit, providing an intuitive and interactive environment for users to interact with the portfolio manager, visualize trading strategies, and analyze portfolio performance.

- **Easy YAML Configuration**: Configure your settings effortlessly using YAML configuration files. YAML offers a human-readable syntax and is easy to understand and modify.
## Limitations

- **No Transaction Costs**: The current implementation does not consider transaction costs associated with buying and selling stocks.

- **Limited Data Source**: Historical market data is sourced exclusively from Yahoo Finance (yfinance), limiting the available data to what is provided by this source.

- **No Short Selling**: The framework does not support short selling, i.e., selling stocks that the user does not own with the intention of buying them back at a lower price.

- **No Leveraging**: Leveraging, the practice of borrowing funds to increase investment exposure, is not supported in the current implementation.

- **No Price Impact on Purchase**: The framework does not model the price impact on purchases, which can occur when buying large quantities of stocks in the market.

## Disclaimer

The Stock-Trading-Portfolio-Manager with Deep-Reinforcement-Learning is provided for educational and informational purposes only. It is not intended to be a substitute for professional financial advice or to be used as a tool for making investment decisions. 

Trading stocks involves inherent risks, including the risk of loss of capital. The performance of trading strategies generated by the software may vary and past performance is not indicative of future results. 

Users of this software are solely responsible for their investment decisions and should consult with a qualified financial advisor before making any investment decisions. The creators of this software disclaim any and all liability for any investment losses or other damages resulting from the use of this software.

By using this software, you acknowledge and agree to the terms of this disclaimer.

## Getting Started

To get started with using the Stock Trading Portfolio Manager, follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine using Git:
    ```
    git clone https://github.com/julblanke/Deep-Reinforcement-Learning-Stock-Portfolio-Manager
    ```

2. **Install Dependencies**: Navigate to the project directory and install the required dependencies using pip:
    ```
    pip install -r requirements.txt
    ```


3. **Run the Application**: Once the dependencies are installed, you can run the application from the project directory:
    ```
    python drlpm/drlpm_run.py --config-path="./examples/multi_agent.yml" 
    ```
    Or use streamlit GUI via
    ```
    streamlit run streamlit/streamlit_app.py
    ```
    ![streamlit_home](./static/streamlit_home.png)

4. **Performance**: Watch performance of your model in the streamlit directory
    ```
    cd streamlit/result_images
    ```
    or use the streamlit GUI to watch performance of portfolio
    ![streamlit_performance](./static/streamlit_performance.png)
    and to get insight in share amounts.
    ![streamlit_shares](./static/streamlit_shares.png)

5. **Customize and run**: Customize by changing the yaml-configurations.
    ```
    # Single Agent
   
    drlpm_type: single_agent
    stock_symbols: ["AAPL", "NVDA", "ASML"]
    initial_balance: 10000
    algo: PPO
    interval: "1d"
    period: "3y"
    train_timesteps: 20000
    update_data: True
    ```
    ```
    # Multi Agent
   
    drlpm_type: multi_agent
    stock_symbols: ["AAPL", "NVDA", "ASML"]
    initial_balance: 10000
    algo: PPO
    interval: "1d"
    period: "3y"
    train_timesteps: 20000
    update_data: True
    
    agent_ids: ["agent1", "agent2", "agent3", "agent4"]
    worker_index: 0
    num_workers: 1
    num_gpus: 0
    algo_reload: False
    ```
    or write your own custom multi-agent algo and or policies like this..
    ```
    class RllibPPO:
    """PPO for multi agent environment."""
    def __init__(self, env: Any, env_config: EnvContext, config: dict, data: pd.DataFrame) -> None:
        """Constructor.

        Args:
            env (MultiAgentStockTradingEnv): The Rllib MultiAgentEnv environment for which to create the algo
            env_config (EnvContext):
                data (pd.DataFrame): Data of given stocks -- includes OHLC data and indicators and indices
                stock_symbols (list): List of stock as stock-symbols, e.g. 'AAPL'
                initial_balance (float): Initial account balance
                agent_ids (list): Names of agents
            config (dict): Dictionary of user defined configurations
            data (pd.DataFrame): Data of given stocks -- includes OHLC data and indicators and indices
        """
        self.env = env
        self.env_config = env_config
        self.config = config
        # NOTE: obs dim given by portfolio values (3) + positions per stock (nr_stocks) + data
        self.observation_shape = (1, 3 + len(config["stock_symbols"]) + data.shape[1])
        self.boundary = config["initial_balance"] * 100     # applies a factor of 100 to obs Box spaces

    def __call__(self) -> Any:
        """Returns PPO algo.

        Note: The current version is a template on how to implement an algo for the multi agent environment into this
              repository. The agents differ only in random gamma values which does not justify the multi agent
              environment since hyperparameter tuning could achieve the same.
              In order to do some serious stuff, use the high customization options of rllib and add a class here.

        Returns:
            (Any): PPO algo with rllib -- with defined policies
        """
        policies = {}
        for agent in self.config["agent_ids"]:
            policies[agent] = PolicySpec(
                                None, Box(low=-self.boundary, high=self.boundary, shape=self.observation_shape),
                                self.env.action_space, {"gamma": RllibPPO._sample_gamma()}
                              )

        return (PPOConfig()
                .environment(env="marl_env", env_config=self.env_config, disable_env_checking=True)
                .multi_agent(
                    policies=policies,
                    policy_mapping_fn=(lambda aid, episode, **kw: self.config["agent_ids"][int(aid[-1]) - 1]),
                )
                .resources(num_gpus=self.config["num_gpus"])
                .build())

    @staticmethod
    def _sample_gamma() -> float:
        """Samples values for gamma.

        Returns:
            (float): Random number between 0.7 and 1
        """
        return max(0.7, min(random.random(), 1))
    ```
## Documentation

The main source of documentation is given by the docstrings and the code itself
which can be accessed either directly or through Sphinx. In order to see Sphinx-doc,
simply open 
```
docs/build/html/index.html
```
in your browser.

## Outlook

The current version of the repository is in the early stages of development and may exhibit instability and limited functionality. It is important to note that the software is not suitable for production use at this time.

I am actively working on an advanced version of the portfolio manager that incorporates more sophisticated algorithms and improved learning capabilities. This advanced version aims to address the limitations of the current version and provide a more robust and reliable solution for managing stock trading portfolios using reinforcement learning techniques.

Thank you for your interest in the project, and stay tuned for updates on its development progress and future releases.


## License

This project is licensed under the MIT License, which means you are free to use, modify, 
and distribute the code for any purpose. See the license file for more details.
