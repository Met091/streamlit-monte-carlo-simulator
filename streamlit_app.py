import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple, Dict, Any, Optional
import logging
import traceback
import math # For isnan checks

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_CONFIG = {
    "initial_balance": 10000.0,
    "risk_percentage": 1.0,
    "win_rate": 50.0,
    "risk_reward_ratio": 1.5,
    "trades_per_month": 20,
    "total_months": 12,
    "n_simulations": 10000, # Keep fixed for consistency unless user input is desired
    "default_ruin_threshold_perc": 50.0, # Default Ruin % (of initial balance)
    "default_profit_target_perc": 100.0 # Default Target % (e.g., 100% = double balance)
}

# --- Monte Carlo Simulator Class ---
class MonteCarloSimulator:
    """
    Enhanced Monte Carlo simulator with advanced risk and performance metrics.
    """
    def __init__(self,
                 initial_balance: float,
                 risk_percentage: float,
                 win_rate: float,
                 risk_reward_ratio: float,
                 trades_per_month: int,
                 total_months: int,
                 n_simulations: int) -> None:
        """Initializes the MonteCarloSimulator."""
        logger.info("Initializing MonteCarloSimulator...")
        self.initial_balance: float = initial_balance
        self.risk_decimal: float = risk_percentage / 100.0
        self.win_rate_decimal: float = win_rate / 100.0
        self.risk_reward_ratio: float = risk_reward_ratio
        self.trades_per_month: int = trades_per_month
        self.total_months: int = total_months
        self.n_simulations: int = n_simulations
        self.total_trades: int = self.trades_per_month * self.total_months

        # Input validation
        if not all([initial_balance > 0, risk_percentage >= 0, win_rate >= 0, win_rate <= 100,
                    risk_reward_ratio > 0, trades_per_month > 0,
                    total_months > 0, n_simulations > 0]):
            error_msg = "Invalid simulation parameters. Check ranges and positivity."
            logger.error(f"Initialization failed: {error_msg} Values: {initial_balance=}, {risk_percentage=}, {win_rate=}, {risk_reward_ratio=}, {trades_per_month=}, {total_months=}, {n_simulations=}")
            raise ValueError(error_msg)

        self.results: Dict[str, Any] = {} # Initialize results dictionary
        logger.info("MonteCarloSimulator initialized successfully.")

    # --- Helper Methods ---

    # *** METHOD RESTORED ***
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
    # *** END OF RESTORED METHOD ***

    def _calculate_drawdown_details(self, balances: np.ndarray) -> Tuple[float, int]:
        """
        Calculates max drawdown percentage and longest drawdown duration (in months) for a single path.

        Args:
            balances (np.ndarray): Array of account balances (monthly).

        Returns:
            Tuple[float, int]: (max_drawdown_percentage, longest_drawdown_duration_months).
        """
        if balances.size < 2:
            return 0.0, 0

        peak = balances[0]
        max_drawdown_perc = 0.0
        longest_dd_duration = 0
        current_dd_duration = 0
        in_drawdown = False

        # Iterate through balances starting from the first month-end balance
        for balance in balances[1:]:
            if balance >= peak:
                peak = balance
                if in_drawdown:
                    # Exited drawdown, record duration if it's the longest
                    longest_dd_duration = max(longest_dd_duration, current_dd_duration)
                    in_drawdown = False
                    current_dd_duration = 0 # Reset duration counter
            else:
                # Calculate drawdown percentage
                drawdown = (peak - balance) / peak if peak > 0 else 0.0
                max_drawdown_perc = max(max_drawdown_perc, drawdown)
                # Track duration
                if not in_drawdown:
                    in_drawdown = True # Entered drawdown
                current_dd_duration += 1

        # If the path ends while still in a drawdown, check its duration
        if in_drawdown:
            longest_dd_duration = max(longest_dd_duration, current_dd_duration)

        return max_drawdown_perc * 100, longest_dd_duration

    def _calculate_profit_factor(self, balances: List[float]) -> Optional[float]:
        """Calculates the profit factor for a single path's balance curve."""
        if len(balances) < 2:
            return None
        monthly_changes = np.diff(np.array(balances))
        gross_profit = np.sum(monthly_changes[monthly_changes > 0])
        gross_loss = np.abs(np.sum(monthly_changes[monthly_changes < 0]))

        if gross_loss == 0:
            # If no losses, profit factor is infinite if profit > 0, or 1.0 if profit is also 0
            return np.inf if gross_profit > 0 else 1.0

        return gross_profit / gross_loss


    def _calculate_path_balances(self, trade_outcomes_for_path: np.ndarray) -> List[float]:
        """Calculates the monthly balance history for a single simulation path."""
        # (Same implementation as before)
        balance = self.initial_balance
        monthly_balance_history = [self.initial_balance]
        for month in range(self.total_months):
            start_trade_index = month * self.trades_per_month
            end_trade_index = start_trade_index + self.trades_per_month
            path_trades_this_month = trade_outcomes_for_path[start_trade_index:min(end_trade_index, self.total_trades)]
            for is_win in path_trades_this_month:
                if balance <= 0: break
                amount_to_risk = balance * self.risk_decimal
                if is_win: balance += amount_to_risk * self.risk_reward_ratio
                else: balance -= amount_to_risk
                balance = max(0.0, balance)
            monthly_balance_history.append(max(0.0, balance))
            if balance <= 0:
                remaining_months = self.total_months - (month + 1)
                monthly_balance_history.extend([0.0] * remaining_months)
                break
        while len(monthly_balance_history) < self.total_months + 1:
             monthly_balance_history.append(monthly_balance_history[-1])
        return monthly_balance_history[:self.total_months + 1]

    # --- Main Simulation Logic ---
    def run_simulations(self, ruin_threshold_perc: float, profit_target_abs: float) -> None:
        """
        Runs the full suite of Monte Carlo simulations and calculates advanced metrics.

        Args:
            ruin_threshold_perc (float): Account balance percentage below which is considered ruin.
            profit_target_abs (float): Absolute account balance target.
        """
        logger.info(f"Starting Monte Carlo simulation: {self.n_simulations} paths, {self.total_trades} trades each.")
        st_progress_bar = st.progress(0.0, text="Generating trade outcomes...")

        try:
            # --- 1. Generate Trade Outcomes ---
            logger.debug("Generating trade outcomes array...")
            all_trade_outcomes_np = np.random.rand(self.n_simulations, self.total_trades) < self.win_rate_decimal
            logger.info("Trade outcomes generated.")
            st_progress_bar.progress(0.05, text="Simulating paths & calculating balances...")

            # --- 2. Simulate Paths & Calculate Balances ---
            final_balances = np.zeros(self.n_simulations)
            all_monthly_balances_list = []
            logger.debug(f"Starting balance simulation loop for {self.n_simulations} paths...")
            for i in range(self.n_simulations):
                trade_outcomes_for_path = all_trade_outcomes_np[i]
                path_monthly_balances = self._calculate_path_balances(trade_outcomes_for_path)
                final_balances[i] = path_monthly_balances[-1]
                all_monthly_balances_list.append(path_monthly_balances)
                # Update progress bar periodically
                update_frequency = max(1, self.n_simulations // 40) # Update more often due to next step
                if (i + 1) % update_frequency == 0 or i == self.n_simulations - 1:
                    progress = 0.05 + 0.45 * ((i + 1) / self.n_simulations) # Progress up to 50%
                    st_progress_bar.progress(progress, text=f"Simulating balances... ({i+1}/{self.n_simulations})")
            logger.info("All simulation path balances calculated.")
            st_progress_bar.progress(0.50, text="Analyzing path details (drawdowns, RoR, target)...")

            # --- 3. Analyze Individual Path Details ---
            all_max_drawdowns = np.zeros(self.n_simulations)
            all_longest_dd_durations = np.zeros(self.n_simulations, dtype=int)
            ruined_count = 0
            target_reached_count = 0
            ruin_threshold_value = self.initial_balance * (ruin_threshold_perc / 100.0)

            logger.debug(f"Starting path analysis loop for {self.n_simulations} paths...")
            all_monthly_balances_np = np.array(all_monthly_balances_list) # Convert for faster processing
            for i in range(self.n_simulations):
                path_balances_np = all_monthly_balances_np[i, :]
                # Calculate drawdown details for each path
                max_dd_perc, longest_dd = self._calculate_drawdown_details(path_balances_np)
                all_max_drawdowns[i] = max_dd_perc
                all_longest_dd_durations[i] = longest_dd

                # Check for Ruin (any point during the simulation)
                if np.min(path_balances_np) <= ruin_threshold_value:
                    ruined_count += 1
                # Check for Target Reached (based on final balance)
                if final_balances[i] >= profit_target_abs:
                    target_reached_count += 1

                # Update progress bar periodically
                update_frequency = max(1, self.n_simulations // 40)
                if (i + 1) % update_frequency == 0 or i == self.n_simulations - 1:
                    progress = 0.50 + 0.45 * ((i + 1) / self.n_simulations) # Progress from 50% to 95%
                    st_progress_bar.progress(progress, text=f"Analyzing paths... ({i+1}/{self.n_simulations})")
            logger.info("Individual path analysis complete (Drawdowns, RoR, Target).")
            st_progress_bar.progress(0.95, text="Calculating final metrics & percentiles...")

            # --- 4. Calculate Aggregate Metrics & Percentiles ---
            logger.debug("Calculating aggregate metrics...")
            # Basic Stats
            best_case_index = np.argmax(final_balances)
            worst_case_index = np.argmin(final_balances)
            median_final_balance = np.median(final_balances)
            median_case_index = np.abs(final_balances - median_final_balance).argmin()

            # Advanced Metrics
            risk_of_ruin = (ruined_count / self.n_simulations) * 100
            prob_of_target = (target_reached_count / self.n_simulations) * 100

            # Calmar Ratio
            median_max_drawdown = np.median(all_max_drawdowns)
            years = self.total_months / 12.0
            if years <= 0 or self.initial_balance <= 0:
                median_annualized_return = 0.0
            else:
                # Avoid issues with zero or negative median final balance for CAGR calc
                safe_median_final = max(1e-9, median_final_balance) # Use a tiny positive number if median is <= 0
                median_annualized_return = ((safe_median_final / self.initial_balance)**(1 / years) - 1) * 100

            if median_max_drawdown <= 0:
                 # If median drawdown is 0, Calmar is infinite if return > 0, else 0
                 calmar_ratio = np.inf if median_annualized_return > 0 else 0.0
            else:
                 calmar_ratio = median_annualized_return / median_max_drawdown

            # Profit Factor (for Median Path)
            median_path_curve = all_monthly_balances_list[median_case_index]
            profit_factor = self._calculate_profit_factor(median_path_curve)

            # Longest Drawdown Durations (Median/Worst Case)
            median_longest_dd = np.median(all_longest_dd_durations)
            worst_case_longest_dd = all_longest_dd_durations[worst_case_index]

            # Percentile Curves
            percentiles_to_calc = [10, 25, 75, 90]
            percentile_curves = {}
            if all_monthly_balances_np.size > 0: # Ensure array is not empty
                for p in percentiles_to_calc:
                    percentile_curves[p] = np.percentile(all_monthly_balances_np, p, axis=0)
            else:
                 logger.warning("Cannot calculate percentile curves, balance array is empty.")


            logger.debug("Final metrics calculation complete.")

            # --- 5. Store Results ---
            self.results = {
                "final_balances": final_balances,
                "all_trade_outcomes": all_trade_outcomes_np, # Keep for detailed path analysis
                "all_monthly_balances": all_monthly_balances_list, # Raw list for detailed path lookups
                "all_max_drawdowns": all_max_drawdowns, # Needed for histogram
                "all_longest_dd_durations": all_longest_dd_durations, # Raw data
                "best_case_index": int(best_case_index),
                "worst_case_index": int(worst_case_index),
                "median_case_index": int(median_case_index),
                "median_final_balance": float(median_final_balance),
                "best_final_balance": float(final_balances[best_case_index]),
                "worst_final_balance": float(final_balances[worst_case_index]),
                "risk_of_ruin_perc": float(risk_of_ruin),
                "probability_of_target_perc": float(prob_of_target),
                "calmar_ratio": float(calmar_ratio) if not math.isnan(calmar_ratio) else 0.0, # Store 0 for NaN Calmar
                "profit_factor_median_path": float(profit_factor) if profit_factor is not None else None,
                "median_max_drawdown_perc": float(median_max_drawdown),
                "median_longest_dd_months": int(median_longest_dd),
                "worst_case_longest_dd_months": int(worst_case_longest_dd),
                "percentile_curves": percentile_curves, # Dict of percentile arrays
                "median_equity_curve": np.median(all_monthly_balances_np, axis=0) if all_monthly_balances_np.size > 0 else np.array([]),
            }
            logger.info("Simulation results analyzed and stored successfully.")
            st_progress_bar.progress(1.0, text="Analysis complete.")

        except Exception as e:
            logger.error(f"Error during simulation run: {e}", exc_info=True)
            st_progress_bar.progress(1.0, text="Simulation failed!")
            raise e # Re-raise

    # --- Getter Methods ---
    def get_results_summary(self) -> Optional[Dict[str, float]]:
        """Returns basic summary stats."""
        if not self.results: return None
        # Check if essential keys exist before returning
        required_keys = ["median_final_balance", "best_final_balance", "worst_final_balance"]
        if not all(key in self.results for key in required_keys):
            logger.error(f"Results summary missing required keys. Available: {list(self.results.keys())}")
            return None
        summary_data = {
            "Median Final Balance": self.results.get("median_final_balance"),
            "Best Final Balance": self.results.get("best_final_balance"),
            "Worst Final Balance": self.results.get("worst_final_balance")
        }
        # Check for None values which might indicate issues
        if None in summary_data.values():
            logger.warning(f"Results summary contains None values: {summary_data}")
            # Decide if returning partial data is okay or return None
            # For now, return None if any key value is None
            return None
        return summary_data


    def get_advanced_metrics(self) -> Optional[Dict[str, Any]]:
         """Returns the calculated advanced metrics."""
         if not self.results: return None
         # Check for required keys
         required_keys = [
             "risk_of_ruin_perc", "probability_of_target_perc", "calmar_ratio",
             "profit_factor_median_path", "median_max_drawdown_perc", "median_longest_dd_months"
         ]
         if not all(key in self.results for key in required_keys):
             logger.error(f"Advanced metrics missing required keys. Available: {list(self.results.keys())}")
             return None

         # Handle potential NaN/inf for display later, but return raw stored value here
         calmar = self.results.get("calmar_ratio")
         pf = self.results.get("profit_factor_median_path")

         return {
             "Risk of Ruin (%)": self.results.get("risk_of_ruin_perc"),
             "Probability of Target (%)": self.results.get("probability_of_target_perc"),
             "Calmar Ratio (Simulated)": calmar, # Return raw value
             "Profit Factor (Median Path)": pf, # Return raw value (can be None or inf)
             "Median Max Drawdown (%)": self.results.get("median_max_drawdown_perc"),
             "Median Longest DD (Months)": self.results.get("median_longest_dd_months"),
         }

    def get_path_details(self, path_index: int) -> Optional[Dict[str, Any]]:
        """Returns detailed metrics for a specific path, including longest DD."""
        # Validate path_index and check if results are available
        if not self.results or path_index < 0 or path_index >= self.n_simulations:
            logger.warning(f"Attempted to get details for invalid path index: {path_index}")
            return None
        # Check for necessary data arrays in results
        required_arrays = ["all_monthly_balances", "all_max_drawdowns", "all_longest_dd_durations", "all_trade_outcomes"]
        if not all(key in self.results for key in required_arrays):
            logger.error(f"Results dictionary incomplete for path details retrieval (Index: {path_index}). Missing keys.")
            return None

        logger.info(f"Calculating details for path index: {path_index}")
        try:
            # Retrieve necessary data for the specified path using the index
            path_curve = self.results["all_monthly_balances"][path_index]
            path_outcomes = self.results["all_trade_outcomes"][path_index]
            max_dd = self.results["all_max_drawdowns"][path_index]
            longest_dd = self.results["all_longest_dd_durations"][path_index]

            # Check if path_curve is valid
            if not isinstance(path_curve, list) or len(path_curve) == 0:
                 logger.error(f"Invalid path curve data for index {path_index}: Type {type(path_curve)}, Length {len(path_curve)}")
                 return None

            # Calculate other metrics using helper methods or direct calculation
            final_bal = path_curve[-1]
            ret_perc = ((final_bal / self.initial_balance) - 1) * 100 if self.initial_balance > 0 else 0.0
            # *** Use the restored method ***
            max_c_wins, max_c_loss, actual_wr = self._analyze_trade_outcomes(path_outcomes)

            details = {
                "Result Balance": final_bal,
                "Return %": ret_perc,
                "Maximum Drawdown %": max_dd,
                "Longest Drawdown Duration (Months)": longest_dd, # Added metric
                "Max Consecutive Wins": max_c_wins,
                "Max Consecutive Losses": max_c_loss,
                "Actual Win Rate %": actual_wr,
                "Monthly Balance Curve": path_curve # Still needed for plotting this specific path
            }
            logger.info(f"Successfully calculated details for path index: {path_index}")
            return details
        except IndexError:
            logger.error(f"IndexError calculating details for path index {path_index}. Results arrays might be inconsistent.", exc_info=True)
            return None
        except Exception as e:
            # Catch any other unexpected error during path detail calculation
            logger.error(f"Unexpected error calculating details for path index {path_index}: {e}", exc_info=True)
            return None

    def get_percentile_curves_data(self) -> Optional[Dict[int, np.ndarray]]:
        """Gets the pre-calculated percentile curves data."""
        return self.results.get("percentile_curves")

    def get_drawdown_distribution_data(self) -> Optional[np.ndarray]:
        """Gets the pre-calculated array of maximum drawdowns for all paths."""
        return self.results.get("all_max_drawdowns")

    def get_median_equity_curve(self) -> Optional[np.ndarray]:
         """Gets the pre-calculated median equity curve."""
         return self.results.get("median_equity_curve")


# --- Streamlit App Layout and Logic ---
st.set_page_config(page_title="Monte Carlo Simulator", page_icon="📊", layout="wide")

st.title("📊 Monte Carlo Simulator")
st.caption("Powered by Trading Mastery Hub")

# --- Simulation Parameters (Sidebar Inputs) ---
st.sidebar.header("Simulation Parameters")
with st.sidebar:
    initial_balance_input = st.number_input(
        "Initial Balance ($)", min_value=1.0, value=DEFAULT_CONFIG["initial_balance"], step=100.0,
        help="Your starting capital."
    )
    risk_percentage_input = st.number_input(
        "Risk per Trade (%)", min_value=0.01, max_value=100.0, value=DEFAULT_CONFIG["risk_percentage"], step=0.01, format="%.2f",
        help="Percentage of current balance risked per trade."
    )
    win_rate_input = st.number_input(
        "Win Rate (%)", min_value=0.0, max_value=100.0, value=DEFAULT_CONFIG["win_rate"], step=0.1, format="%.2f",
        help="Probability of a profitable trade."
    )
    risk_reward_ratio_input = st.number_input(
        "Risk-Reward Ratio", min_value=0.01, value=DEFAULT_CONFIG["risk_reward_ratio"], step=0.1, format="%.2f",
        help="Ratio of profit target to stop loss (e.g., 1.5)."
    )
    trades_per_month_input = st.number_input(
        "Average Trades per Month", min_value=1, value=DEFAULT_CONFIG["trades_per_month"], step=1,
        help="Average trades executed monthly."
    )
    total_months_input = st.number_input(
        "Total Simulation Months", min_value=1, value=DEFAULT_CONFIG["total_months"], step=1,
        help="Simulation duration in months."
    )

    st.sidebar.header("Advanced Metrics Settings")
    ruin_threshold_input = st.number_input(
        "Ruin Threshold (% of Initial Balance)", min_value=0.0, max_value=100.0, value=DEFAULT_CONFIG["default_ruin_threshold_perc"], step=1.0, format="%.1f",
        help="Account balance level (as % of start) considered 'ruin'. Used for Risk of Ruin calculation."
    )
    # Ensure profit target default is calculated based on current initial balance input
    profit_target_default = initial_balance_input * (1 + DEFAULT_CONFIG["default_profit_target_perc"] / 100.0)
    profit_target_input = st.number_input(
        "Profit Target ($)", min_value=float(initial_balance_input), value=float(profit_target_default), step=100.0, format="%.2f",
        help="Absolute account balance target. Used for Probability of Reaching Target calculation."
    )

    run_button = st.button("Run Simulation", key="run_sim_button", type="primary")

# --- Main App Logic ---
if run_button:
    # Placeholders
    summary_placeholder = st.empty()
    adv_metrics_placeholder = st.empty() # Placeholder for new metrics
    details_placeholder = st.container()
    dist_placeholder = st.container() # Will hold both histograms now
    chart_placeholder = st.empty()
    info_placeholder = st.empty()

    try:
        logger.info("Run Simulation button clicked. Initializing simulator...")
        simulator = MonteCarloSimulator(
            initial_balance=initial_balance_input, risk_percentage=risk_percentage_input,
            win_rate=win_rate_input, risk_reward_ratio=risk_reward_ratio_input,
            trades_per_month=trades_per_month_input, total_months=total_months_input,
            n_simulations=DEFAULT_CONFIG["n_simulations"]
        )

        info_placeholder.info(f"🚀 Running {DEFAULT_CONFIG['n_simulations']:,} simulations... Calculating advanced metrics.", icon="⏳")
        # Pass new inputs to the run method
        simulator.run_simulations(
            ruin_threshold_perc=ruin_threshold_input,
            profit_target_abs=profit_target_input
        )
        info_placeholder.success(f"✅ Simulations complete! Displaying results.", icon="🎉")
        logger.info("Attempting to display simulation results.")

        # --- Display Overall Summary ---
        summary = simulator.get_results_summary()
        # Check if summary is not None and contains valid numbers
        if summary and all(isinstance(v, (int, float)) for v in summary.values()):
            with summary_placeholder.container():
                st.subheader("Overall Simulation Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Median Final Balance", f"${summary['Median Final Balance']:,.2f}")
                col2.metric("Best Final Balance", f"${summary['Best Final Balance']:,.2f}")
                col3.metric("Worst Final Balance", f"${summary['Worst Final Balance']:,.2f}")
            logger.info("Overall summary displayed.")
        else:
             logger.error(f"Failed to retrieve valid simulation summary: {summary}")
             summary_placeholder.error("Error: Could not retrieve simulation summary. Check logs.", icon="⚠️")
             st.stop() # Stop if basic summary fails

        # --- Display Advanced Metrics ---
        adv_metrics = simulator.get_advanced_metrics()
        # Check if adv_metrics is not None before proceeding
        if adv_metrics:
             with adv_metrics_placeholder.container():
                  st.subheader("Advanced Risk & Performance Metrics")
                  cols = st.columns(3) # First row
                  # Helper function for safe formatting
                  def format_metric(value, precision=2, suffix="%"):
                      if value is None: return "N/A"
                      if isinstance(value, (int, float)):
                          if math.isinf(value): return "inf"
                          if math.isnan(value): return "N/A"
                          return f"{value:,.{precision}f}{suffix}"
                      return str(value) # Fallback for unexpected types

                  cols[0].metric("Risk of Ruin", format_metric(adv_metrics.get('Risk of Ruin (%)')), help=f"Chance balance dropped below {ruin_threshold_input}% of initial capital.")
                  cols[1].metric("Prob. of Reaching Target", format_metric(adv_metrics.get('Probability of Target (%)')), help=f"Chance final balance reached ${profit_target_input:,.0f}.")
                  cols[2].metric("Calmar Ratio (Sim.)", format_metric(adv_metrics.get('Calmar Ratio (Simulated)'), suffix=""), help="Median Annualized Return / Median Max Drawdown %.")

                  cols = st.columns(3) # Second row
                  cols[0].metric("Profit Factor (Median Path)", format_metric(adv_metrics.get('Profit Factor (Median Path)'), suffix=""), help="Gross Profit / Gross Loss for the median outcome path.")
                  cols[1].metric("Median Max Drawdown", format_metric(adv_metrics.get('Median Max Drawdown (%)')), help="Median of the maximum drawdowns experienced across all paths.")
                  cols[2].metric("Median Longest DD", format_metric(adv_metrics.get('Median Longest DD (Months)'), precision=0, suffix=" Months"), help="Median duration (months) spent below an equity peak across all paths.")

             logger.info("Advanced metrics displayed.")
        else:
             logger.error(f"Failed to retrieve valid advanced metrics dict: {adv_metrics}")
             adv_metrics_placeholder.warning("Could not display advanced metrics. Check logs.", icon="⚠️")


        # --- Display Detailed Scenario Analysis (Updated) ---
        with details_placeholder:
            st.subheader("Detailed Scenario Analysis")
            try:
                logger.info("Retrieving detailed path metrics (incl. longest DD)...")
                best_idx = simulator.results.get("best_case_index")
                worst_idx = simulator.results.get("worst_case_index")
                median_idx = simulator.results.get("median_case_index")

                # Check if indices were found
                if best_idx is None or worst_idx is None or median_idx is None:
                     logger.error("Could not find best/worst/median indices in sim results for detailed analysis.")
                     st.error("Error: Could not identify key simulation paths. Check logs.", icon="⚠️")
                     # Set details to None to prevent errors below
                     best_details, worst_details, median_example_details = None, None, None
                else:
                    # Attempt to get details for each path
                    best_details = simulator.get_path_details(best_idx)
                    worst_details = simulator.get_path_details(worst_idx)
                    median_example_details = simulator.get_path_details(median_idx)

                # Now check if the details were successfully retrieved
                if not all([best_details, worst_details, median_example_details]):
                    logger.error("Failed to retrieve one or more detailed path metrics (best, worst, or median).")
                    # Display error but don't stop the whole app
                    st.error("Error: Could not retrieve all detailed path metrics. Some scenarios might be missing. Check logs.", icon="⚠️")
                    # Attempt to display what we have
                    scenarios = {
                        "Best Case Path": best_details,
                        "Worst Case Path": worst_details,
                        "Median Case Path (Example)": median_example_details
                    }
                    cols = st.columns(len(scenarios))
                    for i, (name, data) in enumerate(scenarios.items()):
                         with cols[i]:
                              st.markdown(f"**{name}**")
                              if data: # Check if data exists for this scenario
                                   st.markdown(f"Ending Balance: `${data.get('Result Balance', 'N/A'):,.2f}`")
                                   st.markdown(f"Total Return: `{data.get('Return %', 'N/A'):.2f}%`")
                                   st.markdown(f"Max Drawdown: `{data.get('Maximum Drawdown %', 'N/A'):.2f}%`")
                                   st.markdown(f"Longest DD: `{data.get('Longest Drawdown Duration (Months)', 'N/A')} Months`")
                                   st.markdown(f"Actual Win Rate: `{data.get('Actual Win Rate %', 'N/A'):.2f}%`")
                                   st.markdown(f"Max Cons. Wins: `{data.get('Max Consecutive Wins', 'N/A')}`")
                                   st.markdown(f"Max Cons. Losses: `{data.get('Max Consecutive Losses', 'N/A')}`")
                              else:
                                   st.markdown("*Data unavailable*")

                else:
                    # All details retrieved successfully
                    scenarios = {"Best Case Path": best_details, "Worst Case Path": worst_details, "Median Case Path (Example)": median_example_details}
                    cols = st.columns(len(scenarios))
                    for i, (name, data) in enumerate(scenarios.items()):
                        with cols[i]:
                            st.markdown(f"**{name}**")
                            st.markdown(f"Ending Balance: `${data.get('Result Balance', 'N/A'):,.2f}`")
                            st.markdown(f"Total Return: `{data.get('Return %', 'N/A'):.2f}%`")
                            st.markdown(f"Max Drawdown: `{data.get('Maximum Drawdown %', 'N/A'):.2f}%`")
                            st.markdown(f"Longest DD: `{data.get('Longest Drawdown Duration (Months)', 'N/A')} Months`")
                            st.markdown(f"Actual Win Rate: `{data.get('Actual Win Rate %', 'N/A'):.2f}%`")
                            st.markdown(f"Max Cons. Wins: `{data.get('Max Consecutive Wins', 'N/A')}`")
                            st.markdown(f"Max Cons. Losses: `{data.get('Max Consecutive Losses', 'N/A')}`")
                    logger.info("Detailed scenario analysis (updated) displayed.")
            except Exception as e:
                # Catch any unexpected error during the display logic for this section
                logger.error(f"Error displaying detailed scenario analysis section: {e}", exc_info=True)
                st.error(f"An error occurred displaying detailed scenario analysis. Check logs.", icon="🚨")


        # --- Display Distribution Plots (Final Balance & Max Drawdown) ---
        with dist_placeholder:
             st.subheader("Distributions")
             col1, col2 = st.columns(2) # Use columns for side-by-side histograms

             # Final Balance Distribution (Existing)
             with col1:
                  try:
                       logger.info("Generating final balance distribution plot...")
                       final_balances = simulator.results.get("final_balances")
                       if final_balances is None or not isinstance(final_balances, np.ndarray) or final_balances.size == 0:
                            logger.error("Final balances data missing/invalid for distribution plot.")
                            st.warning("Final balance distribution unavailable.", icon="⚠️")
                       else:
                            df_final_balance = pd.DataFrame(final_balances, columns=['Final Balance'])
                            fig_hist_balance = px.histogram(
                                 df_final_balance, x='Final Balance', nbins=75, title='Final Balances',
                                 labels={'Final Balance': 'Final Account Balance ($)'}, template='plotly_dark'
                            )
                            fig_hist_balance.update_layout(yaxis_title="Simulations Count", bargap=0.1)
                            # Add median line only if summary is valid
                            if summary and isinstance(summary.get('Median Final Balance'), (int, float)):
                                 fig_hist_balance.add_vline(x=summary['Median Final Balance'], line_dash="dash", line_color="yellow", annotation_text="Median")
                            st.plotly_chart(fig_hist_balance, use_container_width=True)
                            logger.info("Final balance distribution plot displayed.")
                  except Exception as e:
                       logger.error(f"Error generating final balance distribution plot: {e}", exc_info=True)
                       st.error("Error generating final balance distribution plot.", icon="🚨")

             # Max Drawdown Distribution (New)
             with col2:
                  try:
                       logger.info("Generating max drawdown distribution plot...")
                       all_dds = simulator.get_drawdown_distribution_data()
                       if all_dds is None or not isinstance(all_dds, np.ndarray) or all_dds.size == 0:
                            logger.error("Max drawdown data missing/invalid for distribution plot.")
                            st.warning("Max drawdown distribution unavailable.", icon="⚠️")
                       else:
                            df_dds = pd.DataFrame(all_dds, columns=['Max Drawdown %'])
                            # Filter out potential inf values if any (though unlikely for drawdown)
                            df_dds = df_dds[np.isfinite(df_dds['Max Drawdown %'])]
                            fig_hist_dd = px.histogram(
                                 df_dds, x='Max Drawdown %', nbins=75, title='Maximum Drawdowns',
                                 labels={'Max Drawdown %': 'Maximum Drawdown (%)'}, template='plotly_dark'
                            )
                            fig_hist_dd.update_layout(yaxis_title="Simulations Count", bargap=0.1)
                            # Add median line only if adv_metrics is valid
                            median_dd = adv_metrics.get('Median Max Drawdown (%)') if adv_metrics else None
                            if median_dd is not None and isinstance(median_dd, (int, float)):
                                 fig_hist_dd.add_vline(x=median_dd, line_dash="dash", line_color="orange", annotation_text="Median DD")
                            st.plotly_chart(fig_hist_dd, use_container_width=True)
                            logger.info("Max drawdown distribution plot displayed.")
                  except Exception as e:
                       logger.error(f"Error generating max drawdown distribution plot: {e}", exc_info=True)
                       st.error("Error generating max drawdown distribution plot.", icon="🚨")


        # --- Display Equity Curve Chart (with Percentiles) ---
        with chart_placeholder.container():
            st.subheader("Equity Curve Simulation (with Percentile Bands)")
            try:
                logger.info("Generating equity curve plot with percentiles...")
                median_curve = simulator.get_median_equity_curve()
                percentile_data = simulator.get_percentile_curves_data()
                # Retrieve best/worst details again safely
                best_details_local = simulator.get_path_details(simulator.results.get("best_case_index")) if simulator.results.get("best_case_index") is not None else None
                worst_details_local = simulator.get_path_details(simulator.results.get("worst_case_index")) if simulator.results.get("worst_case_index") is not None else None
                best_curve_data = best_details_local.get('Monthly Balance Curve') if best_details_local else None
                worst_curve_data = worst_details_local.get('Monthly Balance Curve') if worst_details_local else None

                # Check if essential data is available and valid
                if median_curve is not None and median_curve.size > 0 and \
                   percentile_data and isinstance(percentile_data, dict) and \
                   best_curve_data is not None and isinstance(best_curve_data, list) and \
                   worst_curve_data is not None and isinstance(worst_curve_data, list):

                    months_axis = list(range(simulator.total_months + 1))
                    fig_line = go.Figure()

                    # Define percentiles for bands
                    p_upper_outer, p_upper_inner = 90, 75
                    p_lower_inner, p_lower_outer = 25, 10

                    # Check if all required percentile keys exist
                    required_percentiles = [p_lower_outer, p_lower_inner, p_upper_inner, p_upper_outer]
                    if not all(p in percentile_data for p in required_percentiles):
                         logger.error(f"Missing required percentile data keys for plotting bands. Available: {list(percentile_data.keys())}")
                         st.warning("Could not plot percentile bands: Missing data. Check logs.", icon="⚠️")
                    else:
                        # Add percentile bands (plot in reverse order for layering)
                        # Outer band (10-90)
                        fig_line.add_trace(go.Scatter(
                            x=months_axis, y=percentile_data[p_upper_outer], fill=None, mode='lines',
                            line_color='rgba(0,100,80,0.2)', name=f'{p_upper_outer}th Percentile', showlegend=False
                        ))
                        fig_line.add_trace(go.Scatter(
                            x=months_axis, y=percentile_data[p_lower_outer], fill='tonexty', mode='lines', # Fill to 90th
                            line_color='rgba(0,100,80,0.2)', name=f'{p_lower_outer}th-{p_upper_outer}th Band'
                        ))
                        # Inner band (25-75)
                        fig_line.add_trace(go.Scatter(
                            x=months_axis, y=percentile_data[p_upper_inner], fill=None, mode='lines',
                            line_color='rgba(0,176,246,0.2)', name=f'{p_upper_inner}th Percentile', showlegend=False
                        ))
                        fig_line.add_trace(go.Scatter(
                            x=months_axis, y=percentile_data[p_lower_inner], fill='tonexty', mode='lines', # Fill to 75th
                            line_color='rgba(0,176,246,0.2)', name=f'{p_lower_inner}th-{p_upper_inner}th Band'
                        ))

                    # Add Median, Best, Worst lines on top
                    fig_line.add_trace(go.Scatter(
                        x=months_axis, y=worst_curve_data, mode='lines', line=dict(color='red', width=1.5), name='Worst Case Path'
                    ))
                    fig_line.add_trace(go.Scatter(
                        x=months_axis, y=best_curve_data, mode='lines', line=dict(color='lightgreen', width=1.5), name='Best Case Path'
                    ))
                    fig_line.add_trace(go.Scatter(
                        x=months_axis, y=median_curve, mode='lines', line=dict(color='yellow', dash='dash', width=2), name='Median Path'
                    ))


                    fig_line.update_layout(
                        title='Simulated Equity Curves Over Time', template='plotly_dark',
                        xaxis_title="Month Number", yaxis_title="Account Balance ($)",
                        hovermode="x unified", legend_title_text='Scenario/Band'
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
                    logger.info("Equity curve plot with percentiles displayed.")
                else:
                    # Log which specific data was missing
                    missing = [k for k, v in {'Median Curve': median_curve, 'Percentile Data': percentile_data, 'Best Curve': best_curve_data, 'Worst Curve': worst_curve_data}.items() if v is None or (isinstance(v, np.ndarray) and v.size==0) or (isinstance(v, dict) and not v)]
                    logger.error(f"Missing necessary data for percentile equity curve plot: {missing}")
                    st.warning(f"Could not generate equity curve plot: Missing required data ({', '.join(missing)}). Check logs.", icon="⚠️")
            except Exception as e:
                logger.error(f"Error generating equity curve plot: {e}", exc_info=True)
                st.error(f"An error occurred generating the equity curve plot. Check logs.", icon="🚨")

    # Error Handling for Initialization or Main Block
    except ValueError as e:
        logger.warning(f"Caught ValueError (likely invalid input): {e}", exc_info=False) # Don't need traceback for input error
        st.error(f"Input Error: {e}. Please check the simulation parameters.", icon="🚨")
        info_placeholder.empty()
    except Exception as e:
        logger.critical(f"Caught unexpected CRITICAL ERROR during simulation or display: {e}", exc_info=True)
        st.error(f"A critical application error occurred. Please check the application logs for details.", icon="🔥")
        info_placeholder.empty()

# Initial state message
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to see the results.")

# --- Explanation Section ---
with st.expander("ℹ️ How this simulation works & Metrics Explained (Advanced)"):
    # Safely access simulator attributes only if simulator exists
    total_trades_display = simulator.total_trades if 'simulator' in locals() and hasattr(simulator, 'total_trades') else DEFAULT_CONFIG['trades_per_month']*DEFAULT_CONFIG['total_months']
    explanation_text = f"""
    This tool uses the **Monte Carlo method** to simulate **{DEFAULT_CONFIG['n_simulations']:,}** possible future scenarios for your trading strategy based on the parameters provided.

    **Simulation Process:**
    1.  **Trade Outcome Generation**: Generates {total_trades_display:,} random win/loss outcomes for each of the {DEFAULT_CONFIG['n_simulations']:,} paths based on `Win Rate (%)`.
    2.  **Path Simulation**: Calculates the month-by-month balance for each path, applying `Risk per Trade (%)` (compounded) and `Risk-Reward Ratio`. Trading stops if balance hits zero.
    3.  **Results Analysis**: Analyzes all simulated paths to calculate summary statistics, distributions, and advanced metrics.

    **Overall Simulation Summary:**
    * **Median/Best/Worst Final Balance**: The 50th percentile, highest, and lowest ending balances across all simulations.

    **Advanced Risk & Performance Metrics:**
    * **Risk of Ruin (%)**: The percentage of simulations where the balance dropped below the specified `Ruin Threshold (%)` of the initial capital *at any point*.
    * **Prob. of Reaching Target (%)**: The percentage of simulations where the *final* balance met or exceeded the specified `Profit Target ($)`.
    * **Calmar Ratio (Simulated)**: Median Annualized Return / Median Max Drawdown %. Higher is generally better risk-adjusted return. Calculated based on median outcomes across all simulations. (Annualized Return = `((Median Final / Initial)^(1 / Years) - 1) * 100`).
    * **Profit Factor (Median Path)**: Gross Profit / Gross Loss for the specific simulation path that ended closest to the overall median final balance. Values > 1 indicate profitability.
    * **Median Max Drawdown (%)**: The 50th percentile of the maximum drawdown percentages experienced in each individual simulation path. Represents a typical worst drawdown.
    * **Median Longest DD (Months)**: The 50th percentile of the longest duration (in months) spent below a previous equity peak in each individual simulation path. Indicates typical time recovery.

    **Distributions:**
    * **Final Balances Histogram**: Shows the frequency of different final balance outcomes.
    * **Maximum Drawdowns Histogram**: Shows the frequency of different maximum drawdown percentages experienced across all paths.

    **Detailed Scenario Analysis (Based on Specific Paths):**
    * Analyzes the specific paths ending with the **Best**, **Worst**, and **Median** final balances.
    * **Ending Balance, Total Return %, Max Drawdown %**: Metrics for that *single* path.
    * **Longest DD (Months)**: The longest time (in months) that *specific* path spent below an equity peak.
    * **Actual Win Rate %, Max Cons. Wins/Losses**: Observed metrics *within that specific path*.

    **Equity Curve Chart (with Percentile Bands):**
    * Visualizes balance progression over time.
    * **Best/Worst Case Path**: Actual balance paths from the simulations ending highest/lowest.
    * **Median Path (dashed line)**: The median balance across *all* simulations at *each month*.
    * **Percentile Bands (e.g., 10th-90th, 25th-75th)**: Shaded areas showing the range within which a certain percentage of simulation paths fell at each month. Provides a view of outcome variability over time.

    **Disclaimer**: Simulations use assumptions. Real trading involves complexities (costs, slippage, market changes, psychology). Use for education, not prediction.
    """
    st.markdown(explanation_text)
