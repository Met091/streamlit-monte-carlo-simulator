# Monte Carlo Simulator

This Streamlit application performs Monte Carlo simulations to model potential outcomes of a trading strategy based on user-defined parameters.

## Features

* Simulates thousands of trading paths based on:
    * Initial Balance
    * Risk per Trade (%)
    * Win Rate (%)
    * Risk-Reward Ratio
    * Average Trades per Month
    * Total Simulation Months
* Calculates and displays:
    * Overall summary statistics (Median, Best, Worst final balance)
    * Distribution of final balances (Histogram)
    * Detailed analysis of Best, Worst, and Median-case paths (Return, Max Drawdown, Win Rate, Streaks)
    * Equity curve chart comparing Best, Worst, and Overall Median paths over time.
* Uses Plotly for interactive charts.
* Includes logging for monitoring and debugging.

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit app:**
    ```bash
    streamlit run streamlit_app.py
    ```

## Deployment to Streamlit Cloud

1.  Push this repository (containing `streamlit_app.py` and `requirements.txt`) to GitHub.
2.  Go to [share.streamlit.io](https://share.streamlit.io/).
3.  Click "New app", connect your GitHub account, select the repository, branch, and the `streamlit_app.py` file.
4.  Click "Deploy!".

## Disclaimer

This simulation uses simplified assumptions. Real-world trading involves complexities not modeled here. Use this tool for educational purposes only. Past performance and simulation results are not indicative of future returns.
