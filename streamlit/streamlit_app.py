import sys
sys.path.append('.')

import os
import streamlit as st
from PIL import Image
from drlpm.drlpm_run import DrlPortfolioManager


class Gui:
    """ Graphical User Interface with streamlit. """

    @staticmethod
    def run_streamlit():
        # config home page
        st.set_page_config(layout="wide")
        st.title("Deep Reinforcement Learning Portfolio Manager")
        banner = os.path.join("./static/", f"pixlr_banner.png")
        image = Image.open(banner)
        st.image(image, caption='Source: pixrl image generator')
        tab1, tab2, tab3 = st.tabs(["Training and Simulation", "Performance", "Shares"])

        # sidebar for user input -- config
        with st.sidebar:
            st.header("Configuration Stock Symbols")
            st.session_state.user_input = st.text_area("Stock symbols", "AAPL NVDA")
            st.session_state.stock_symbols = st.session_state.user_input.split()
            st.session_state.initial_balance = st.text_input("Initial account balance", "10000")
            st.session_state.train_timesteps = st.text_input("Training timesteps", "10000")
            st.session_state.period = st.text_input("Period in yfinance termonology", "2y")
            st.session_state.interval = st.text_input("Interval in yfinance termonology", "1d")
            st.session_state.update_data = st.checkbox("Check if data should be updated", value=False)

        # tab1
        with tab1:
            st.header("Train and run")
            done = False
            if st.button(f'Train and run'):
                with st.spinner('Running...'):
                    DrlPortfolioManager.run(stock_symbols=st.session_state.stock_symbols,
                                            initial_balance=float(st.session_state.initial_balance),
                                            train_timesteps=int(st.session_state.train_timesteps),
                                            period=st.session_state.period,
                                            interval=st.session_state.interval,
                                            update_data=st.session_state.update_data)
                    st.success('Done.')
                done = True

        # tab2
        with tab2:
            if done:
                st.header("Performance")
                img_src_dir = "./streamlit/result_images/performance"

                # Display images side by side
                col1, col2 = st.columns(2)
                with col1:
                    # total
                    figure_file = os.path.join(img_src_dir, f"total.png")
                    image = Image.open(figure_file)
                    st.image(image, use_column_width=True)

                    # portfolio
                    figure_file = os.path.join(img_src_dir, f"portfolio.png")
                    image = Image.open(figure_file)
                    st.image(image, use_column_width=True)

                with col2:
                    figure_file = os.path.join(img_src_dir, f"cash.png")
                    image = Image.open(figure_file)
                    st.image(image, use_column_width=True)

        # tab3
        with tab3:
            if done:
                st.header("Shares")

                img_src_dir = "./streamlit/result_images/shares"
                files = os.listdir(img_src_dir)
                images = list()
                for file in files:
                    stock_symbol = file.split(".")[0]
                    figure_file = os.path.join(img_src_dir, f"{stock_symbol}.png")
                    image = Image.open(figure_file)
                    images.append(image)

                # Display images side by side
                col1, col2 = st.columns(2)
                first_half_images = images[:len(images) // 2]
                second_half_images = images[len(images) // 2:]
                with col1:
                    for img in second_half_images:
                        st.image(img, use_column_width=True)

                with col2:
                    for img in first_half_images:
                        st.image(img, use_column_width=True)


if __name__ == "__main__":
    Gui().run_streamlit()
