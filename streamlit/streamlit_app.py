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
            st.session_state.user_config_input = st.text_input("config path", "./examples/multi_agent.yml")

        # tab1
        with tab1:
            st.header("Train and run")
            done = False
            if st.button(f'Train and run'):
                with st.spinner('Running...'):
                    DrlPortfolioManager.run(config_path=st.session_state.user_config_input)
                    st.success('Done.')
                done = True

        # tab2
        with tab2:
            if done:
                st.header("Performance")
                img_src_dir = "./streamlit/result_images/performance"

                files = sorted(os.listdir(img_src_dir))
                images_total = list()
                images_portfolio = list()
                images_cash = list()
                for file in files:
                    figure_file = os.path.join(img_src_dir, file)

                    if "total" in file:
                        image = Image.open(figure_file)
                        images_total.append(image)
                    if "portfolio" in file:
                        image = Image.open(figure_file)
                        images_portfolio.append(image)
                    if "cash" in file:
                        image = Image.open(figure_file)
                        images_cash.append(image)

                # Display images side by side
                col1, col2, col3 = st.columns(3)
                with col1:
                    # total
                    for image in images_total:
                        st.image(image, use_column_width=True)

                with col2:
                    # portfolio
                    for image in images_portfolio:
                        st.image(image, use_column_width=True)

                with col3:
                    for image in images_cash:
                        st.image(image, use_column_width=True)

        # tab3
        with tab3:
            if done:
                st.header("Shares")

                img_src_dir = "./streamlit/result_images/shares"
                files = sorted(os.listdir(img_src_dir))
                images = list()
                for file in files:
                    figure_file = os.path.join(img_src_dir, file)
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
