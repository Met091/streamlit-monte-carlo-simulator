import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple, Dict, Any, Optional
import logging # Import logging module
import traceback # To log tracebacks

# --- Logging Configuration ---
# Configure basic logging to capture info level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Default parameters for the simulation if not provided by the user
DEFAULT_CONFIG = {
    "initial_balance": 10000.0,
    "risk_percentage": 1.0,
    "win_rate": 50.0,
    "risk_reward_ratio": 1.5,
    "trades_per_month": 20,
    "total_months": 12,
    "n_simulations": 10000 # Fixed number of simulations for consistency
}

# --- Monte Carlo Simulator Class ---
class MonteCarloSimulator:
    """
    Encapsulates the logic for running Monte Carlo trading simulations.
    Includes logging for key steps and error handling.

    Attributes:
        initial_balance (float): Starting capital for the simulation.
        risk_decimal (float): Risk per trade expressed as a decimal (e.g., 1% = 0.01).
        win_rate_decimal (float): Probability of a winning trade as a decimal (e.g., 50% = 0.5).
        risk_reward_ratio (float): The ratio of potential profit to potential loss for each trade.
        trades_per_month (int): The number of trades simulated per month.
        total_months (int): The total duration of the simulation in months.
        n_simulations (int): The number of independent simulation paths to run.
        total_trades (int): The total number of trades in a single simulation path.
        results (Dict[str, Any]): A dictionary to store the outcomes of the simulations.
    """
    def __init__(self,
                 initial_balance: float,
                 risk_percentage: float,
                 win_rate: float,
                 risk_reward_ratio: float,
                 trades_per_month: int,
                 total_months: int,
                 n_simulations: int) -> None:
        """Initializes the MonteCarloSimulator with user-defined parameters."""
        logger.info("Initializing MonteCarloSimulator...")
        # Assign parameters to instance variables
        self.initial_balance: float = initial_balance
        self.risk_decimal: float = risk_percentage / 100.0
        self.win_rate_decimal: float = win_rate / 100.0
        self.risk_reward_ratio: float = risk_reward_ratio
        self.trades_per_month: int = trades_per_month
        self.total_months: int = total_months
        self.n_simulations: int = n_simulations
        # Calculate total trades per simulation path
        self.total_trades: int = self.trades_per_month * self.total_months

        # Input validation
        if not all([initial_balance > 0,
                    risk_percentage >= 0, # Allow 0% risk, though unusual
                    win_rate >= 0, win_rate <= 100,
                    risk_reward_ratio > 0,
                    trades_per_month > 0,
                    total_months > 0,
                    n_simulations > 0]):
            error_msg = "Invalid simulation parameters. Check ranges and positivity."
            logger.error(error_msg + f" Values: initial_balance={initial_balance}, risk%={risk_percentage}, win_rate%={win_rate}, R:R={risk_reward_ratio}, trades/mo={trades_per_month}, months={total_months}, sims={n_simulations}")
            raise ValueError(error_msg) # Raise error to stop execution

        # Initialize results dictionary
        self.results: Dict[str, Any] = {}
        logger.info("MonteCarloSimulator initialized successfully.")

    def _calculate_max_drawdown(self, balances: np.ndarray) -> float:
        """
        Calculates the maximum drawdown percentage from a series of balances.
        Drawdown is the percentage decline from a peak balance to a subsequent trough.

        Args:
            balances (np.ndarray): An array of account balances over time.

        Returns:
            float: The maximum drawdown percentage. Returns 0.0 if fewer than 2 balances.
        """
        if balances.size < 2:
            return 0.0 # Cannot calculate drawdown with less than 2 data points

        peak = balances[0] # Initialize peak with the first balance
        max_drawdown = 0.0 # Initialize max drawdown

        # Iterate through balances to find peaks and troughs
        for balance in balances:
            if balance > peak:
                peak = balance # Update peak if a new high is reached
            # Calculate drawdown from the current peak
            drawdown = (peak - balance) / peak if peak > 0 else 0.0
            # Update maximum drawdown found so far
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown * 100 # Return as percentage

    def _analyze_trade_outcomes(self, trade_outcomes: np.ndarray) -> Tuple[int, int, float]:
        """
        Analyzes a sequence of trade outcomes (wins/losses) for a single path.

        Args:
            trade_outcomes (np.ndarray): A boolean array where True represents a win, False a loss.

        Returns:
            Tuple[int, int, float]: A tuple containing:
                - Maximum consecutive wins.
                - Maximum consecutive losses.
                - Actual win rate percentage for this path.
        """
        if trade_outcomes.size == 0:
            return 0, 0, 0.0 # Handle empty array case

        max_wins, max_losses = 0, 0
        current_wins, current_losses = 0, 0
        total_wins = 0

        # Iterate through each trade outcome
        for outcome in trade_outcomes:
            if outcome: # If it's a win (True)
                total_wins += 1
                current_wins += 1
                current_losses = 0 # Reset loss streak
                max_wins = max(max_wins, current_wins) # Update max win streak
            else: # If it's a loss (False)
                current_losses += 1
                current_wins = 0 # Reset win streak
                max_losses = max(max_losses, current_losses) # Update max loss streak

        # Calculate the actual win rate for this specific path
        actual_win_rate = (total_wins / trade_outcomes.size) * 100 if trade_outcomes.size > 0 else 0.0

        return max_wins, max_losses, actual_win_rate

    def _calculate_path_balances(self, trade_outcomes_for_path: np.ndarray) -> List[float]:
        """
        Calculates the monthly balance history for a single simulation path based on its trade outcomes.

        Args:
            trade_outcomes_for_path (np.ndarray): Boolean array of win/loss outcomes for this path.

        Returns:
            List[float]: A list containing the account balance at the start (month 0) and end of each month.
                         The list will have `total_months + 1` elements.
        """
        balance = self.initial_balance
        # Initialize history with the starting balance
        monthly_balance_history = [self.initial_balance]

        # Iterate through each month of the simulation
        for month in range(self.total_months):
            # Determine the slice of trades for the current month
            start_trade_index = month * self.trades_per_month
            end_trade_index = start_trade_index + self.trades_per_month
            # Ensure we don't go beyond the total number of trades
            path_trades_this_month = trade_outcomes_for_path[start_trade_index:min(end_trade_index, self.total_trades)]

            # Simulate trades within the month
            for is_win in path_trades_this_month:
                if balance <= 0: # Stop trading if balance is wiped out
                    break
                # Calculate amount to risk based on current balance
                amount_to_risk = balance * self.risk_decimal
                # Update balance based on trade outcome
                if is_win:
                    balance += amount_to_risk * self.risk_reward_ratio # Win
                else:
                    balance -= amount_to_risk # Loss
                # Ensure balance doesn't go below zero
                balance = max(0.0, balance)

            # Record the balance at the end of the month
            monthly_balance_history.append(max(0.0, balance))

            # If balance is zero, fill remaining months with zero
            if balance <= 0:
                remaining_months = self.total_months - (month + 1)
                monthly_balance_history.extend([0.0] * remaining_months)
                break # Exit month loop

        # Ensure the history list has the correct length (total_months + 1)
        # This handles cases where the simulation ended early due to zero balance
        while len(monthly_balance_history) < self.total_months + 1:
             monthly_balance_history.append(monthly_balance_history[-1]) # Append last known balance

        # Return the history, ensuring it's exactly total_months + 1 long
        return monthly_balance_history[:self.total_months + 1]

    def run_simulations(self) -> None:
        """
        Runs the full suite of Monte Carlo simulations.
        Generates trade outcomes, calculates paths, and analyzes results.
        Uses Streamlit progress bar for user feedback.
        """
        logger.info(f"Starting Monte Carlo simulation: {self.n_simulations} paths, {self.total_trades} trades each.")
        # Initialize Streamlit progress bar
        st_progress_bar = st.progress(0.0, text="Generating trade outcomes...")

        try:
            # Generate all trade outcomes efficiently using NumPy
            # Creates a 2D array: (n_simulations x total_trades) of booleans (True=Win, False=Loss)
            all_trade_outcomes_np = np.random.rand(self.n_simulations, self.total_trades) < self.win_rate_decimal
            logger.info("Trade outcomes generated.")
            st_progress_bar.progress(0.05, text="Simulating paths...") # Update progress

            # Initialize arrays/lists to store results
            final_balances = np.zeros(self.n_simulations)
            all_monthly_balances_list = [] # List to store monthly balance history for each path

            # Loop through each simulation path
            for i in range(self.n_simulations):
                # Get the trade outcomes for the current path
                trade_outcomes_for_path = all_trade_outcomes_np[i]
                # Calculate the monthly balance history for this path
                path_monthly_balances = self._calculate_path_balances(trade_outcomes_for_path)
                # Store the final balance for this path
                final_balances[i] = path_monthly_balances[-1]
                # Store the full monthly balance history
                all_monthly_balances_list.append(path_monthly_balances)

                # Update progress bar periodically to avoid slowing down
                update_frequency = max(1, self.n_simulations // 20) # Update roughly 20 times
                if (i + 1) % update_frequency == 0:
                    progress = 0.05 + 0.90 * ((i + 1) / self.n_simulations) # Scale progress between 5% and 95%
                    st_progress_bar.progress(progress, text=f"Simulating paths... ({i+1}/{self.n_simulations})")

            logger.info("All simulation paths calculated.")
            st_progress_bar.progress(0.95, text="Analyzing results...") # Update progress

            # Analyze the results across all simulations
            best_case_index = np.argmax(final_balances) # Index of the path with the highest final balance
            worst_case_index = np.argmin(final_balances) # Index of the path with the lowest final balance
            median_final_balance = np.median(final_balances) # The 50th percentile final balance
            # Find the index of the path closest to the median final balance
            median_case_index = np.abs(final_balances - median_final_balance).argmin()

            # Store all calculated results in the instance's results dictionary
            self.results = {
                "final_balances": final_balances, # Array of final balances for all paths
                "all_trade_outcomes": all_trade_outcomes_np, # 2D array of all trade outcomes
                "best_case_index": int(best_case_index),
                "worst_case_index": int(worst_case_index),
                "median_case_index": int(median_case_index),
                "median_final_balance": float(median_final_balance),
                "best_final_balance": float(final_balances[best_case_index]),
                "worst_final_balance": float(final_balances[worst_case_index]),
                "all_monthly_balances": all_monthly_balances_list # List of lists (monthly balances per path)
            }
            logger.info("Simulation results analyzed and stored.")
            st_progress_bar.progress(1.0, text="Analysis complete.") # Final progress update

        except Exception as e:
            # Log any errors that occur during the simulation
            logger.error(f"Error during simulation run: {e}", exc_info=True) # Log traceback
            st_progress_bar.progress(1.0, text="Simulation failed.") # Update progress bar status
            raise e # Re-raise the exception to be caught by the main app logic

    def get_results_summary(self) -> Optional[Dict[str, float]]:
        """
        Returns a summary dictionary of the overall simulation results.

        Returns:
            Optional[Dict[str, float]]: Dictionary with key summary stats (median, best, worst final balance),
                                        or None if results are not available.
        """
        if not self.results:
            logger.warning("Attempted to get results summary, but simulation results are empty.")
            return None
        # Extract key summary statistics from the stored results
        return {
            "Median Final Balance": self.results["median_final_balance"],
            "Best Final Balance": self.results["best_final_balance"],
            "Worst Final Balance": self.results["worst_final_balance"]
        }

    def get_path_details(self, path_index: int) -> Optional[Dict[str, Any]]:
        """
        Calculates and returns detailed metrics for a specific simulation path identified by its index.

        Args:
            path_index (int): The index of the simulation path to analyze (0 to n_simulations-1).

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing detailed metrics for the path (final balance,
                                      return, drawdown, win rate, streaks, monthly curve), or None if index is invalid
                                      or results are missing.
        """
        # Validate path_index and check if results are available
        if not self.results or path_index < 0 or path_index >= self.n_simulations:
            logger.warning(f"Attempted to get details for invalid path index: {path_index} or results missing.")
            return None

        logger.info(f"Calculating details for path index: {path_index}")
        try:
            # Retrieve necessary data for the specified path
            path_outcomes = self.results["all_trade_outcomes"][path_index]
            path_curve = self.results["all_monthly_balances"][path_index] # List of monthly balances
            path_curve_np = np.array(path_curve) # Convert to NumPy array for calculations

            # Calculate metrics for this path
            final_bal = path_curve[-1] # Final balance is the last element
            # Calculate percentage return
            ret_perc = ((final_bal / self.initial_balance) - 1) * 100 if self.initial_balance > 0 else 0.0
            # Calculate maximum drawdown for this path
            max_dd = self._calculate_max_drawdown(path_curve_np)
            # Analyze trade outcomes for streaks and actual win rate
            max_c_wins, max_c_loss, actual_wr = self._analyze_trade_outcomes(path_outcomes)

            # Compile details into a dictionary
            details = {
                "Result Balance": final_bal,
                "Return %": ret_perc,
                "Maximum Drawdown %": max_dd,
                "Max Consecutive Wins": max_c_wins,
                "Max Consecutive Losses": max_c_loss,
                "Actual Win Rate %": actual_wr,
                "Monthly Balance Curve": path_curve # Include the balance history
            }
            logger.info(f"Successfully calculated details for path index: {path_index}")
            return details
        except Exception as e:
            # Log errors during detail calculation for a specific path
            logger.error(f"Error calculating details for path index {path_index}: {e}", exc_info=True)
            return None # Return None if calculation fails

    def get_median_equity_curve(self) -> Optional[np.ndarray]:
        """
        Calculates the median balance across all simulations for each month.
        This provides a representative equity curve progression over time.

        Returns:
            Optional[np.ndarray]: A NumPy array representing the median balance at each month-end
                                  (including month 0), or None if data is unavailable or calculation fails.
        """
        # Check if necessary results data is present
        if not self.results or "all_monthly_balances" not in self.results:
            logger.warning("Attempted to get median equity curve, but results/balances are missing.")
            return None

        all_balances = self.results["all_monthly_balances"] # Get the list of monthly balance lists
        if not all_balances: # Check if the list is empty
            logger.warning("Attempted to get median equity curve, but balance list is empty.")
            return None

        try:
            # Convert the list of lists into a 2D NumPy array (simulations x months)
            padded_balances = np.array(all_balances)
            # Basic check for expected shape
            if padded_balances.ndim != 2:
                 logger.error(f"Median curve calculation error: Expected 2D array, got {padded_balances.ndim}D.")
                 return None
            # Calculate the median along the simulations axis (axis=0) for each month
            median_curve = np.median(padded_balances, axis=0)
            logger.info("Median equity curve calculated successfully.")
            return median_curve
        except Exception as e:
            # Log errors during median calculation
            logger.error(f"Error calculating median equity curve: {e}", exc_info=True)
            return None

# --- Streamlit App Layout and Logic ---

# Configure Streamlit page settings
st.set_page_config(
    page_title="Trading Monte Carlo Simulator",
    page_icon="📊",
    layout="wide", # Use wide layout for better chart display
    initial_sidebar_state="expanded" # Keep sidebar open by default
)

# Main title and caption
st.title("📊 Monte Carlo Trading Strategy Simulator")
st.caption("Simulate potential trading outcomes based on your strategy parameters.")

# --- Simulation Parameters (Sidebar Inputs) ---
# Use sidebar for input parameters
st.sidebar.header("Simulation Parameters")
with st.sidebar:
    # Input fields for simulation parameters with default values and help text
    initial_balance_input = st.number_input(
        "Initial Balance ($)", min_value=1.0, value=DEFAULT_CONFIG["initial_balance"],
        step=100.0, help="Your starting capital for the simulation."
    )
    risk_percentage_input = st.number_input(
        "Risk per Trade (%)", min_value=0.01, max_value=100.0, value=DEFAULT_CONFIG["risk_percentage"],
        step=0.01, format="%.2f", help="Percentage of your current account balance risked on each trade."
    )
    win_rate_input = st.number_input(
        "Win Rate (%)", min_value=0.0, max_value=100.0, value=DEFAULT_CONFIG["win_rate"],
        step=0.1, format="%.2f", help="The historical or expected probability of a trade being profitable."
    )
    risk_reward_ratio_input = st.number_input(
        "Risk-Reward Ratio", min_value=0.01, value=DEFAULT_CONFIG["risk_reward_ratio"],
        step=0.1, format="%.2f", help="The ratio of your average profit target to your average stop loss (e.g., 1.5 means you aim to win 1.5 times your risk)."
    )
    trades_per_month_input = st.number_input(
        "Average Trades per Month", min_value=1, value=DEFAULT_CONFIG["trades_per_month"],
        step=1, help="The average number of trades you expect to execute per month."
    )
    total_months_input = st.number_input(
        "Total Simulation Months", min_value=1, value=DEFAULT_CONFIG["total_months"],
        step=1, help="The duration for which the simulation should run."
    )
    # Button to trigger the simulation
    run_button = st.button("Run Simulation", key="run_sim_button", type="primary")

# --- Main App Logic ---

# Execute simulation only when the button is clicked
if run_button:
    # Placeholders for dynamically updating sections
    summary_placeholder = st.empty()
    details_placeholder = st.container() # Use container for potentially larger content
    dist_placeholder = st.empty()
    chart_placeholder = st.empty()
    info_placeholder = st.empty() # For status messages

    try:
        logger.info("Run Simulation button clicked. Initializing simulator...")
        # Create an instance of the simulator with user inputs
        simulator = MonteCarloSimulator(
            initial_balance=initial_balance_input,
            risk_percentage=risk_percentage_input,
            win_rate=win_rate_input,
            risk_reward_ratio=risk_reward_ratio_input,
            trades_per_month=trades_per_month_input,
            total_months=total_months_input,
            n_simulations=DEFAULT_CONFIG["n_simulations"] # Use fixed number from config
        )

        # Display running message and start simulation
        info_placeholder.info(f"🚀 Running {DEFAULT_CONFIG['n_simulations']:,} simulations... This may take a moment.", icon="⏳")
        simulator.run_simulations() # Execute the core simulation logic
        info_placeholder.success(f"✅ Simulations complete! Displaying results.", icon="🎉") # Success message
        logger.info("Attempting to display simulation results.")

        # --- Display Overall Summary ---
        summary = simulator.get_results_summary()
        if summary:
            with summary_placeholder.container(): # Use container to group metrics
                st.subheader("Overall Simulation Summary")
                col1, col2, col3 = st.columns(3) # Use columns for layout
                col1.metric("Median Final Balance", f"${summary['Median Final Balance']:,.2f}", help="The 50th percentile outcome.")
                col2.metric("Best Final Balance", f"${summary['Best Final Balance']:,.2f}", help="The highest balance achieved in any simulation.")
                col3.metric("Worst Final Balance", f"${summary['Worst Final Balance']:,.2f}", help="The lowest balance achieved (can be $0).")
            logger.info("Overall summary displayed.")
        else:
             # Handle case where summary couldn't be retrieved
             logger.error("Failed to retrieve simulation summary.")
             summary_placeholder.error("Error: Could not retrieve simulation summary.", icon="⚠️")
             st.stop() # Stop execution if summary fails

        # --- Display Detailed Scenario Analysis ---
        with details_placeholder:
            st.subheader("Detailed Scenario Analysis")
            try:
                logger.info("Retrieving detailed path metrics...")
                # Get details for best, worst, and median-example paths
                best_details = simulator.get_path_details(simulator.results["best_case_index"])
                worst_details = simulator.get_path_details(simulator.results["worst_case_index"])
                median_example_details = simulator.get_path_details(simulator.results["median_case_index"])

                # Check if all details were retrieved successfully
                if not all([best_details, worst_details, median_example_details]):
                    logger.error("Failed to retrieve one or more detailed path metrics.")
                    st.error("Error: Could not retrieve all detailed path metrics.", icon="⚠️")
                else:
                    logger.info("Detailed path metrics retrieved successfully.")
                    # Define scenarios to display
                    scenarios = {
                        "Best Case Path": best_details,
                        "Worst Case Path": worst_details,
                        "Median Case Path (Example)": median_example_details
                    }
                    # Use columns to display scenario details side-by-side
                    cols = st.columns(len(scenarios))
                    for i, (name, data) in enumerate(scenarios.items()):
                        with cols[i]:
                            st.markdown(f"**{name}**")
                            st.markdown(f"Ending Balance: `${data['Result Balance']:,.2f}`")
                            st.markdown(f"Total Return: `{data['Return %']:.2f}%`")
                            st.markdown(f"Max Drawdown: `{data['Maximum Drawdown %']:.2f}%`")
                            st.markdown(f"Actual Win Rate: `{data['Actual Win Rate %']:.2f}%`")
                            st.markdown(f"Max Cons. Wins: `{data['Max Consecutive Wins']}`")
                            st.markdown(f"Max Cons. Losses: `{data['Max Consecutive Losses']}`")
                    logger.info("Detailed scenario analysis displayed.")
            except Exception as e:
                # Catch errors specifically during the display of details
                logger.error(f"Error displaying detailed scenario analysis: {e}", exc_info=True)
                st.error(f"Error displaying detailed scenario analysis: {e}", icon="🚨")

        # --- Display Distribution Plot ---
        with dist_placeholder.container(): # Use container for the plot
            st.subheader("Distribution of Final Balances")
            try:
                logger.info("Generating distribution plot...")
                final_balances = simulator.results.get("final_balances")
                # Check if final balance data is available
                if final_balances is None or final_balances.size == 0:
                    logger.error("Final balances data is missing or empty for distribution plot.")
                    st.warning("Could not generate distribution plot: Final balances data missing.", icon="⚠️")
                else:
                    # Create DataFrame for Plotly
                    final_balances_df = pd.DataFrame(final_balances, columns=['Final Balance'])
                    # Create histogram using Plotly Express
                    fig_hist = px.histogram(
                        final_balances_df, x='Final Balance', nbins=100, # Adjust nbins as needed
                        title=f'Distribution of Final Balances ({DEFAULT_CONFIG["n_simulations"]:,} Simulations)',
                        labels={'Final Balance': 'Final Account Balance ($)'},
                        template='plotly_dark' # Use a dark theme
                    )
                    fig_hist.update_layout(yaxis_title="Number of Simulations")
                    # Add vertical lines for median, best, worst if summary exists
                    if summary:
                        fig_hist.add_vline(x=summary['Median Final Balance'], line_dash="dash", line_color="yellow", annotation_text="Median")
                        fig_hist.add_vline(x=summary['Best Final Balance'], line_dash="dash", line_color="lightgreen", annotation_text="Best")
                        fig_hist.add_vline(x=summary['Worst Final Balance'], line_dash="dash", line_color="red", annotation_text="Worst")
                    # Display the plot in Streamlit
                    st.plotly_chart(fig_hist, use_container_width=True)
                    logger.info("Distribution plot displayed.")
            except Exception as e:
                # Catch errors during plot generation
                logger.error(f"Error generating distribution plot: {e}", exc_info=True)
                st.error(f"Error generating distribution plot: {e}", icon="🚨")

        # --- Display Equity Curve Chart ---
        with chart_placeholder.container(): # Use container for the chart
            st.subheader("Equity Curve Simulation")
            try:
                logger.info("Generating equity curve plot...")
                # Retrieve necessary data: median curve and specific path curves
                median_equity_curve = simulator.get_median_equity_curve()
                # Check if data from previous steps (best/worst details) and median curve are available
                if 'best_details' in locals() and best_details and \
                   'worst_details' in locals() and worst_details and \
                   median_equity_curve is not None:

                    months_axis = list(range(simulator.total_months + 1)) # X-axis for the plot
                    max_len = simulator.total_months + 1 # Expected length of curves

                    # Prepare data dictionary for DataFrame creation
                    plot_data = {'Month': months_axis}
                    curves_to_plot = {
                        'Best Case Path': best_details.get('Monthly Balance Curve'),
                        'Worst Case Path': worst_details.get('Monthly Balance Curve'),
                        'Overall Median Path': median_equity_curve # This is already a numpy array or None
                    }

                    valid_curves = True # Flag to track if all necessary curves are valid
                    # Validate and process each curve
                    for name, curve in curves_to_plot.items():
                        if curve is None:
                            logger.error(f"Curve data for '{name}' is missing.")
                            st.warning(f"Could not plot equity curve: Missing data for '{name}'.", icon="⚠️")
                            valid_curves = False; break # Stop if any curve is missing
                        # Ensure curve is a list or numpy array before processing
                        if isinstance(curve, (list, np.ndarray)):
                            # Ensure curve has the correct length (pad/truncate if necessary)
                            processed_curve = list(curve[:max_len]) # Truncate if too long
                            if len(processed_curve) < max_len: # Pad if too short (e.g., balance went to 0)
                                logger.warning(f"Curve data for '{name}' has length {len(processed_curve)}, expected {max_len}. Padding with last value.")
                                processed_curve.extend([processed_curve[-1]] * (max_len - len(processed_curve)))
                            plot_data[name] = processed_curve
                        else:
                            logger.error(f"Curve data for '{name}' is not a list or array, but {type(curve)}.")
                            st.warning(f"Could not plot equity curve: Invalid data type for '{name}'.", icon="⚠️")
                            valid_curves = False; break # Stop if data type is wrong

                    # Proceed only if all curves were valid
                    if valid_curves:
                        # Create DataFrame from the processed data
                        df_plot = pd.DataFrame(plot_data)
                        # Final check on DataFrame validity
                        if df_plot.empty or 'Month' not in df_plot.columns or len(df_plot.columns) < 2 :
                             logger.error("DataFrame for equity curve plot is invalid or empty after processing.")
                             st.warning("Could not plot equity curve: Failed to prepare plot data.", icon="⚠️")
                        else:
                            # Melt DataFrame for Plotly Express line chart
                            df_melted = pd.melt(df_plot, id_vars=['Month'], var_name='Scenario', value_name='Balance ($)')
                            # Create line chart
                            fig_line = px.line(
                                df_melted, x='Month', y='Balance ($)', color='Scenario',
                                title='Simulated Equity Curves Over Time',
                                labels={'Balance ($)': 'Account Balance ($)'},
                                template='plotly_dark', # Use dark theme
                                # Define specific colors for clarity
                                color_discrete_map={
                                    'Best Case Path': 'lightgreen',
                                    'Worst Case Path': 'red',
                                    'Overall Median Path': 'yellow'
                                }
                            )
                            # Customize layout and traces
                            fig_line.update_layout(
                                xaxis_title="Month Number",
                                yaxis_title="Account Balance ($)",
                                legend_title_text='Scenario',
                                hovermode="x unified" # Show tooltips for all lines at once
                            )
                            # Make median line dashed, others solid with markers
                            fig_line.for_each_trace(lambda t: t.update(mode='lines+markers', marker=dict(size=4)) if t.name != 'Overall Median Path' else t.update(mode='lines', line=dict(dash='dash')))
                            # Display the chart
                            st.plotly_chart(fig_line, use_container_width=True)
                            logger.info("Equity curve plot displayed.")
                else:
                    # Handle case where essential data for the plot is missing
                    logger.error("Missing necessary data for equity curve plot (median, best, or worst curves).")
                    st.warning("Could not generate equity curve plot: Missing required data.", icon="⚠️")
            except Exception as e:
                # Catch errors during equity curve plot generation
                logger.error(f"Error generating equity curve plot: {e}", exc_info=True)
                st.error(f"Error generating equity curve plot: {e}", icon="🚨")

    # Catch specific expected errors like invalid inputs
    except ValueError as e:
        logger.error(f"Input Validation Error: {e}", exc_info=True)
        st.error(f"Input Error: {e}", icon="🚨")
        info_placeholder.empty() # Clear any status message
    # Catch any other unexpected errors during the simulation run
    except Exception as e:
        logger.critical(f"CRITICAL ERROR during simulation execution: {e}", exc_info=True)
        st.error(f"A critical error occurred during the simulation: {e}", icon="🔥")
        info_placeholder.empty() # Clear any status message
        # st.exception(e) # Optionally uncomment to show full traceback in the Streamlit app for debugging

# Display initial message if the simulation hasn't been run yet
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to see the results.")

# --- Explanation Section ---
# Use an expander to keep the main view clean
with st.expander("ℹ️ How this simulation works & Metrics Explained"):
    # --- IMPORTANT ---
    # Use f-string for dynamic values like n_simulations.
    # Double curly braces {{ or }} are needed for literal braces within f-strings.
    explanation_text = f"""
    This tool uses the **Monte Carlo method** to simulate **{DEFAULT_CONFIG['n_simulations']:,}** possible future scenarios for your trading strategy based on the parameters you provide in the sidebar.

    **Simulation Process:**
    1.  **Trade Outcome Generation**: For each of the {DEFAULT_CONFIG['n_simulations']:,} simulation paths, a sequence of {simulator.total_trades if 'simulator' in locals() else DEFAULT_CONFIG['trades_per_month']*DEFAULT_CONFIG['total_months']} trade outcomes (win or loss) is randomly generated based on your input `Win Rate (%)`. This is done efficiently using NumPy.
    2.  **Path Simulation**: The simulator then calculates the account balance month-by-month for each path:
        * It starts with your `Initial Balance ($)`.
        * For each simulated trade, it calculates the amount to risk based on the `Risk per Trade (%)` of the *current* balance (this implements compounding).
        * The balance is increased by (Risked Amount * `Risk-Reward Ratio`) for a win, or decreased by the Risked Amount for a loss.
        * The balance cannot go below zero. If it hits zero, trading stops for that path.
        * The balance is recorded at the end of each month.
    3.  **Results Analysis**: After all {DEFAULT_CONFIG['n_simulations']:,} paths are simulated, the tool analyzes the results.

    **Overall Simulation Summary:**
    * **Median Final Balance**: This is the middle value if you lined up all {DEFAULT_CONFIG['n_simulations']:,} final balances from lowest to highest. It represents a 'typical' or 50th percentile outcome.
    * **Best/Worst Final Balance**: These are the absolute highest and lowest final balances achieved across *all* simulation paths, showing the potential range of outcomes (including the possibility of losing everything).

    **Distribution of Final Balances (Histogram):**
    * This chart groups the final balances from all simulations into bins and shows how many simulations ended in each range. It helps you visualize the probability of different outcomes. A tall bar means many simulations ended with a balance in that range. Vertical lines mark the Median, Best, and Worst final balances for reference.

    **Detailed Scenario Analysis (Based on Specific Paths):**
    The app identifies three specific simulation paths out of the {DEFAULT_CONFIG['n_simulations']:,} runs: the one that resulted in the **Best Final Balance**, the one ending in the **Worst Final Balance**, and one whose final balance was **closest to the overall Median**. It then analyzes the performance *within* these specific paths:
    * **Ending Balance**: The final account balance for that *single* simulation path.
    * **Total Return %**: The percentage gain or loss for that *single* path compared to the initial balance.
    * **Max Drawdown %**: The largest percentage drop from a peak balance to a subsequent low *within that specific path's* monthly balance history. A high drawdown indicates significant volatility and risk, even if the path ended profitably.
    * **Actual Win Rate %**: The actual win percentage calculated from the sequence of trades *within that specific path*. Due to random chance, this can differ from the input `Win Rate (%)` for any single path.
    * **Max Cons. Wins/Losses**: The longest winning or losing streak encountered *during that specific path*. This helps understand potential psychological challenges.

    **Equity Curve Chart:**
    * This chart visualizes how the account balance potentially grows (or shrinks) over the simulated months.
    * **Best/Worst Case Path**: These lines show the actual month-by-month balance progression from the single simulations that resulted in the highest and lowest final balances. They represent specific, possible journeys.
    * **Overall Median Path (dashed line)**: This line is different. At *each month*, it shows the *median balance* calculated across *all* {DEFAULT_CONFIG['n_simulations']:,} simulations *up to that point*. It represents a more typical or central progression over time, generally smoother than any single path.

    **Disclaimer**: Monte Carlo simulations are based on the assumptions you provide. Real-world trading is more complex and involves factors not modeled here, such as changing market conditions, transaction costs (fees/slippage), unexpected events, and emotional decision-making. This tool is for educational and illustrative purposes to understand potential outcomes based on statistical inputs, not a guarantee of future results.
    """
    st.markdown(explanation_text) # Display the formatted explanation text
