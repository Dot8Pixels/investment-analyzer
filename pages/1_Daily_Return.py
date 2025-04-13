"""
Daily Return Analysis Dashboard
------------------------------------
A Streamlit application for analyzing daily returns of financial assets with statistical metrics,
visualizations, and trading insights.

Features:
- Historical return analysis with customizable rolling statistics
- Distribution analysis compared to normal distribution
- Trading statistics and extreme movement detection
"""

import traceback
from datetime import date
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import kurtosis, norm, skew


# ---- Configuration ----
def setup_page():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="Asset Daily Return Analysis",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ---- Data Functions ----
@st.cache_data
def get_asset_data(
    ticker: str, start_date: date, end_date: date
) -> Optional[pd.DataFrame]:
    """
    Fetch historical price data for the specified asset and calculate daily returns.

    Args:
        ticker: Asset ticker symbol (e.g., "BTC-USD", "NVDA")
        start_date: Start date for data retrieval
        end_date: End date for data retrieval

    Returns:
        DataFrame with OHLCV data and calculated returns, or None if the ticker is invalid
    """
    try:
        # Download historical data
        asset_df = yf.download(ticker, start=start_date, end=end_date)

        # Validate data was received
        if asset_df is None or asset_df.empty:
            raise ValueError(f"No data returned for ticker {ticker}")

        # Clean up multi-level columns if present
        if isinstance(asset_df.columns, pd.MultiIndex):
            asset_df.columns = asset_df.columns.get_level_values(0)

        # Calculate daily returns
        asset_df["Return"] = asset_df["Close"].pct_change()

        return asset_df.dropna()

    except Exception as e:
        stack = traceback.format_stack()
        stack_str = "".join(stack)
        st.error(
            f"""❌ Error fetching data for '{ticker}'. Please enter a valid ticker symbol.
            Check ticker at: https://finance.yahoo.com/lookup/

            Error: {str(e)}
            Traceback: {stack_str}"""
        )
        return None


def calculate_metrics(
    df: pd.DataFrame, rolling_window: int = 30
) -> Tuple[pd.DataFrame, dict]:
    """
    Calculate rolling statistics and summary metrics for returns analysis.

    Args:
        df: DataFrame with a 'Return' column
        rolling_window: Number of days for rolling window calculations (default: 30)

    Returns:
        Tuple of:
        - Updated DataFrame with calculated metrics
        - Dictionary of summary statistics
    """
    # Calculate rolling metrics
    df[f"Mean_{rolling_window}"] = df["Return"].rolling(rolling_window).mean()
    df[f"SD_{rolling_window}"] = df["Return"].rolling(rolling_window).std()

    # Calculate standard deviation bands
    df["+1SD"] = df[f"Mean_{rolling_window}"] + df[f"SD_{rolling_window}"]
    df["-1SD"] = df[f"Mean_{rolling_window}"] - df[f"SD_{rolling_window}"]
    df["+2SD"] = df[f"Mean_{rolling_window}"] + 2 * df[f"SD_{rolling_window}"]
    df["-2SD"] = df[f"Mean_{rolling_window}"] - 2 * df[f"SD_{rolling_window}"]

    # Calculate summary statistics
    stats = {
        "mean_return": df["Return"].mean() * 100,
        "median_return": df["Return"].median() * 100,
        "std_dev": df["Return"].std() * 100,
        "max_return": df["Return"].max() * 100,
        "min_return": df["Return"].min() * 100,
        "skewness": skew(df["Return"]),
        "kurtosis": kurtosis(df["Return"]),
        "rolling_window": rolling_window,
    }

    # Calculate SD bands for reference
    stats["one_sd_pos"] = stats["mean_return"] + stats["std_dev"]
    stats["one_sd_neg"] = stats["mean_return"] - stats["std_dev"]
    stats["two_sd_pos"] = stats["mean_return"] + 2 * stats["std_dev"]
    stats["two_sd_neg"] = stats["mean_return"] - 2 * stats["std_dev"]

    # Latest rolling statistics
    stats["latest_rolling"] = (
        df[
            [
                f"Mean_{rolling_window}",
                f"SD_{rolling_window}",
                "+1SD",
                "-1SD",
                "+2SD",
                "-2SD",
            ]
        ]
        .dropna()
        .iloc[-1]
        * 100
    )

    # Trading day statistics
    stats["total_days"] = len(df)
    stats["up_days"] = (df["Return"] > 0).sum()
    stats["down_days"] = (df["Return"] < 0).sum()
    stats["flat_days"] = (df["Return"] == 0).sum()
    stats["up_days_pct"] = (stats["up_days"] / stats["total_days"]) * 100
    stats["down_days_pct"] = (stats["down_days"] / stats["total_days"]) * 100

    # Average gain/loss on up/down days
    stats["mean_up"] = (
        df.loc[df["Return"] > 0, "Return"].mean() * 100 if stats["up_days"] > 0 else 0
    )
    stats["mean_down"] = (
        df.loc[df["Return"] < 0, "Return"].mean() * 100 if stats["down_days"] > 0 else 0
    )

    # Extreme movement statistics
    stats["extreme_up_days"] = (df["Return"] * 100 > stats["two_sd_pos"]).sum()
    stats["extreme_down_days"] = (df["Return"] * 100 < stats["two_sd_neg"]).sum()
    stats["extreme_up_pct"] = (stats["extreme_up_days"] / stats["total_days"]) * 100
    stats["extreme_down_pct"] = (stats["extreme_down_days"] / stats["total_days"]) * 100

    return df, stats


# ---- Visualization Functions ----
def create_returns_timeseries(df: pd.DataFrame, rolling_window: int = 30) -> go.Figure:
    """
    Create a time series plot of daily returns with rolling mean and SD bands.

    Args:
        df: DataFrame with return data and calculated metrics
        rolling_window: Number of days used in rolling calculations

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Add daily returns
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Return"],
            mode="lines",
            name="Daily Return",
            line=dict(color="rgba(0, 0, 255, 0.3)"),
        )
    )

    # Add rolling mean
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[f"Mean_{rolling_window}"],
            mode="lines",
            name=f"{rolling_window}-Day Mean",
            line=dict(color="black"),
        )
    )

    # Add SD bands
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["+1SD"],
            mode="lines",
            name="+1 SD",
            line=dict(color="gold", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["-1SD"],
            mode="lines",
            name="-1 SD",
            line=dict(color="gold", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["+2SD"],
            mode="lines",
            name="+2 SD",
            line=dict(color="red", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["-2SD"],
            mode="lines",
            name="-2 SD",
            line=dict(color="red", dash="dash"),
        )
    )

    # Update layout
    fig.update_layout(
        template="none",
        xaxis_title="Date",
        yaxis_title="Daily Return",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=60, r=60, t=60, b=60),
        hovermode="x unified",
    )

    return fig


def create_distribution_plot(df: pd.DataFrame) -> go.Figure:
    """
    Create a histogram of returns with normal distribution overlay and SD markers.

    Args:
        df: DataFrame with return data

    Returns:
        Plotly Figure object
    """
    returns = df["Return"]
    fig = go.Figure()

    # Add histogram of returns
    fig.add_trace(
        go.Histogram(
            x=returns,
            opacity=0.6,
            name="Observed Return",
            marker_color="lightblue",
            histnorm="probability density",
        )
    )

    # Add normal distribution curve
    x_range = np.linspace(returns.min(), returns.max(), 100)
    y_range = norm.pdf(x_range, returns.mean(), returns.std())
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_range,
            mode="lines",
            name="Normal Distribution",
            line=dict(color="red"),
        )
    )

    # Add vertical lines for mean and standard deviations
    fig.add_vline(
        x=returns.mean(),
        line_dash="dash",
        line_color="black",
        annotation_text="Mean",
        annotation_position="top right",
    )

    # Add SD bands
    sd = returns.std()
    mean = returns.mean()

    for sd_value, color in [(1, "gold"), (2, "red")]:
        # Positive SD
        fig.add_vline(
            x=mean + sd_value * sd,
            line_dash="dash",
            line_color=color,
            annotation_text=f"+{sd_value} SD",
            annotation_position="top right",
        )

        # Negative SD
        fig.add_vline(
            x=mean - sd_value * sd,
            line_dash="dash",
            line_color=color,
            annotation_text=f"-{sd_value} SD",
            annotation_position="top right",
        )

    # Update layout
    fig.update_layout(
        template="none",
        xaxis_title="Daily Return",
        yaxis_title="Density",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        hovermode="x unified",
    )

    return fig


def create_updown_pie(up_days: int, down_days: int) -> go.Figure:
    """
    Create a pie chart of positive vs negative trading days.

    Args:
        up_days: Number of days with positive returns
        down_days: Number of days with negative returns

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(
        go.Pie(
            labels=["Positive Days", "Negative Days"],
            values=[up_days, down_days],
            marker_colors=["#66BB6A", "#EF5350"],
            hole=0.4,
            textinfo="percent+label",
            hoverinfo="label+percent+value",
            textfont=dict(color="white"),
            insidetextorientation="horizontal",
        )
    )

    fig.update_layout(
        title="📈 Ratio of Positive vs Negative Days",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


# ---- UI Components ----
def render_sidebar() -> Tuple[str, date, date, int]:
    """
    Render the sidebar with user input controls.

    Returns:
        Tuple of (ticker, start_date, end_date, rolling_window)
    """
    st.sidebar.header("Settings")

    # Ticker input
    default_ticker = "BTC-USD"
    ticker = st.sidebar.text_input(
        "Enter Asset Ticker (e.g., BTC-USD, NVDA, ^GSPC)", default_ticker
    )

    # Date range selector
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", pd.to_datetime("2014-09-17"))
    with col2:
        end_date = st.date_input("End Date", pd.to_datetime("today"))

    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        rolling_window = st.slider(
            "Rolling Window (Days)",
            min_value=5,
            max_value=90,
            value=30,
            help="Number of days used for calculating rolling statistics",
        )

    # Add information section
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        """This dashboard analyzes the daily returns of financial assets.
        Enter a valid Yahoo Finance ticker symbol to begin.

        **Features:**
        - Statistical analysis of daily returns
        - Return distribution visualization
        - Trading day performance metrics
        - Extreme movement detection
        """
    )

    return ticker, start_date, end_date, rolling_window


def render_summary_stats(stats: dict) -> None:
    """
    Render a table with summary statistics.

    Args:
        stats: Dictionary of calculated statistics
    """
    summary_df = pd.DataFrame(
        {
            "Value": [
                stats["mean_return"],
                stats["median_return"],
                stats["std_dev"],
                stats["max_return"],
                stats["min_return"],
                stats["one_sd_pos"],
                stats["one_sd_neg"],
                stats["two_sd_pos"],
                stats["two_sd_neg"],
                stats["skewness"],
                stats["kurtosis"],
            ]
        },
        index=[
            "Mean Return (%)",
            "Median Return (%)",
            "Standard Deviation (%)",
            "Max Return (%)",
            "Min Return (%)",
            "+1 SD (%)",
            "-1 SD (%)",
            "+2 SD (%)",
            "-2 SD (%)",
            "Skewness",
            "Kurtosis",
        ],
    )
    summary_df.index.name = "Metric"
    st.dataframe(summary_df, use_container_width=True)


def render_rolling_stats(latest_rolling: pd.Series, rolling_window: int) -> None:
    """
    Render a table with latest rolling statistics.

    Args:
        latest_rolling: Series with the latest rolling statistics
        rolling_window: Number of days used in rolling calculations
    """
    # Rename columns to include rolling window period
    renamed_series = latest_rolling.copy()
    renamed_series.index = pd.Index(
        [
            f"{rolling_window}-Day Mean (%)"
            if "Mean" in idx
            else f"{rolling_window}-Day SD (%)"
            if "SD_" in idx
            else idx
            for idx in renamed_series.index
        ]
    )

    rolling_df = renamed_series.to_frame(name="Latest Value (%)")
    rolling_df.index.name = "Metric"

    st.dataframe(rolling_df, use_container_width=True)


def render_trading_report(stats: dict, ticker: str) -> None:
    """
    Render the trading statistics report.

    Args:
        stats: Dictionary of calculated statistics
        ticker: Asset ticker symbol
    """
    col1, col2 = st.columns((1, 1))

    with col1:
        st.markdown(f"""
        - 📅 **Total Trading Days:** {stats["total_days"]} days
        - ✅ **Days Closed Positive:** {stats["up_days"]} days ({stats["up_days_pct"]:.2f}%)
        - 📉 **Days Closed Negative:** {stats["down_days"]} days ({stats["down_days_pct"]:.2f}%)
        - 📊 **Average Gain on Up Days:** {stats["mean_up"]:.2f}%
        - 📉 **Average Loss on Down Days:** {stats["mean_down"]:.2f}%

        ### 🔍 Extreme Price Movements
        - 🔺 **Strong Up Days (> +{stats["two_sd_pos"]:.2f}%):** {stats["extreme_up_days"]} days ({stats["extreme_up_pct"]:.2f}%)
        - 🔻 **Strong Down Days (< {stats["two_sd_neg"]:.2f}%):** {stats["extreme_down_days"]} days ({stats["extreme_down_pct"]:.2f}%)
        """)

    with col2:
        st.plotly_chart(
            create_updown_pie(stats["up_days"], stats["down_days"]),
            use_container_width=True,
        )

    # Financial interpretation
    st.markdown(f"""
    ### 📊 Daily Return Analysis
    - 🟢 **Mean Return:** {stats["mean_return"]:.2f}%
    → Average daily return over all trading days.
    - 🟢 **Median Return:** {stats["median_return"]:.2f}%
    → The midpoint of daily returns, meaning 50% of the time {ticker} closes above this value.
    - 🟠 **Standard Deviation:** {stats["std_dev"]:.2f}%
    → Measures price volatility, most returns fall within ±{stats["std_dev"]:.2f}%.
    - 🟥 **Max/Min Daily Returns:** {stats["max_return"]:.2f}% / {stats["min_return"]:.2f}%
    → Largest single-day gain and loss recorded in the period.

    ### 📊 Standard Deviation Ranges
    - 🔵 **±1 SD Range:** ({stats["one_sd_pos"]:.2f}%, {stats["one_sd_neg"]:.2f}%)
    → About 68% of daily moves should stay within this range.
    - 🔵 **±2 SD Range:** ({stats["two_sd_pos"]:.2f}%, {stats["two_sd_neg"]:.2f}%)
    → About 95% of daily moves should stay within this range.

    ### 📉 Distribution Insights
    - 🌀 **Skewness:** {stats["skewness"]:.2f}
    → {interpret_skewness(stats["skewness"])}
    - 🌀 **Kurtosis:** {stats["kurtosis"]:.2f}
    → {interpret_kurtosis(stats["kurtosis"])}
    """)


def interpret_skewness(skew_value: float) -> str:
    """Provide interpretation of skewness values."""
    if skew_value > 0.5:
        return "Strong positive skew indicates more frequent small losses but occasional large gains"
    elif skew_value > 0.1:
        return "Slightly positive skew indicates a modest tendency toward large positive moves"
    elif skew_value < -0.5:
        return "Strong negative skew indicates more frequent small gains but occasional large losses"
    elif skew_value < -0.1:
        return "Slightly negative skew indicates a modest tendency toward large negative moves"
    else:
        return "Near-zero skew indicates a relatively symmetric distribution of returns"


def interpret_kurtosis(kurt_value: float) -> str:
    """Provide interpretation of kurtosis values."""
    if kurt_value > 3:
        return "High kurtosis indicates more frequent extreme movements than a normal distribution"
    elif kurt_value > 1:
        return (
            "Moderate kurtosis indicates somewhat more extreme movements than expected"
        )
    elif kurt_value < -1:
        return "Negative kurtosis indicates fewer extreme movements than a normal distribution"
    else:
        return (
            "Kurtosis near zero indicates a distribution of returns similar to normal"
        )


# ---- Main Application ----
def main() -> None:
    """Main application entry point."""
    # Setup page
    setup_page()

    # Render sidebar and get inputs
    ticker, start_date, end_date, rolling_window = render_sidebar()

    # Fetch data
    asset_df = get_asset_data(ticker, start_date, end_date)

    if asset_df is not None and len(asset_df) > 0:
        # Calculate metrics
        asset_df, stats = calculate_metrics(asset_df, rolling_window)

        # Main dashboard
        st.title(f"📊 {ticker} Daily Return Analysis")
        st.caption(f"Using {rolling_window}-day rolling window")

        # First row: Time series and summary stats
        col1, col2 = st.columns((3, 2))

        with col1:
            st.subheader(
                f"📈 Daily Return with {rolling_window}-Day Rolling Mean and Standard Deviation Bands"
            )
            st.plotly_chart(
                create_returns_timeseries(asset_df, rolling_window),
                use_container_width=True,
            )

        with col2:
            st.subheader("📊 Daily Return Summary (%)")
            render_summary_stats(stats)

        # Second row: Distribution and rolling stats
        col1, col2 = st.columns((3, 2))

        with col1:
            st.subheader("📊 Return Distribution vs Normal Distribution")
            st.plotly_chart(
                create_distribution_plot(asset_df), use_container_width=True
            )

        with col2:
            st.subheader(f"📉 Latest {rolling_window}-Day Rolling Stats (%)")
            render_rolling_stats(stats["latest_rolling"], rolling_window)

        # Report section
        st.subheader("📊 Trading Statistics Report")
        st.subheader("📈 Daily Performance Summary")

        render_trading_report(stats, ticker)

        # Additional insights and analysis recommendations
        with st.expander("💡 Additional Analysis Insights and Recommendations"):
            st.markdown(f"""
            ### Sharpe Ratio Approximation
            The annualized Sharpe ratio (assuming zero risk-free rate) is approximately:
            **{(stats["mean_return"] * 252) / (stats["std_dev"] * np.sqrt(252)):.2f}**

            ### Distribution Analysis
            - The asset's returns are {"not " if abs(stats["skewness"]) < 0.5 and abs(stats["kurtosis"]) < 1 else ""}normally distributed.
            - The QQ-plot and normality tests would provide more definitive analysis.

            ### Risk Management Implications
            - For this asset, a 2-sigma daily move is {stats["two_sd_pos"]:.2f}%, which should happen only ~2.5% of the time.
            - Actual frequency of moves > 2-sigma: {stats["extreme_up_pct"] + stats["extreme_down_pct"]:.2f}% of trading days.

            ### Effect of Rolling Window Selection
            - Current rolling window: {rolling_window} days
            - Shorter windows ({rolling_window // 2} days) would be more responsive to recent price changes but noisier
            - Longer windows ({rolling_window * 2} days) would provide more stable readings but react slower to new trends

            ### Recommendations for Further Analysis
            1. **Regime Analysis**: Test for shifting volatility regimes
            2. **Autocorrelation Analysis**: Check for serial correlation in returns
            3. **Conditional Volatility Models**: GARCH modeling for volatility forecasting
            4. **Rolling Beta Calculation**: For stocks, relative to market index
            5. **Drawdown Analysis**: Duration and magnitude of peak-to-trough moves
            """)

    else:
        st.warning("Enter a valid ticker symbol to begin analysis")


if __name__ == "__main__":
    main()
