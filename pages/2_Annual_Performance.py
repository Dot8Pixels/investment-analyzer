"""
Annual Performance Analyzer 📊

This Streamlit dashboard visualizes the annual performance of financial assets
retrieved from Yahoo Finance. Users can input a ticker symbol and select specific
years to compare asset performance either as percentage returns or actual values.

Features:
- Interactive line charts with hover details.
- Sidebar controls for selecting ticker, number of years, and display mode.
- Option to toggle between % return and value.
- Summary table with year-end value and performance.
"""

from datetime import datetime
from typing import Any, cast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Constants
COLOR_PALETTE = [
    "#d62728",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
MONTH_ABBR = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
DEFAULT_TICKER = "AAPL"
CACHE_TTL = 3600  # Cache data for 1 hour

# Page configuration
st.set_page_config(page_title="Annual Performance Analyzer", layout="wide")


def main() -> None:
    st.title("📊 Annual Performance Analyzer")
    ticker_symbol, years_to_display, show_percent = get_user_inputs()
    st.header(f"Annual Performance for {ticker_symbol.upper()}")

    if ticker_symbol:
        process_and_display_data(ticker_symbol, years_to_display, show_percent)
    else:
        st.info("Please enter a ticker symbol to get started.")


def get_user_inputs() -> tuple[str, list[int], bool]:
    with st.sidebar:
        st.header("Settings")
        ticker_symbol = st.text_input("Enter Ticker Symbol", value=DEFAULT_TICKER)
        current_year = datetime.now().year
        num_years = st.slider("Number of Years to Display", 1, 10, 5)
        all_years = list(range(current_year, current_year - num_years, -1))
        years_to_display = st.multiselect(
            "Choose years to display on the chart", options=all_years, default=all_years
        )
        show_percent = st.checkbox("Show % Return (uncheck for Value)", value=True)
    return ticker_symbol, years_to_display, show_percent


@st.cache_data(ttl=CACHE_TTL)
def get_asset_data(ticker: str, years: list[int]) -> pd.DataFrame | None:
    try:
        if not years:
            st.warning("Please select at least one year to display")
            return None

        start_date = f"{min(years) - 1}-12-31"
        end_date = datetime.now().strftime("%Y-%m-%d")
        asset_df = yf.download(ticker, start=start_date, end=end_date)

        if asset_df is None or asset_df.empty:
            st.error(f"No data found for ticker '{ticker}'")
            return None

        if isinstance(asset_df.columns, pd.MultiIndex):
            asset_df.columns = asset_df.columns.get_level_values(0)

        return asset_df

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def process_annual_data(
    data: pd.DataFrame, years: list[int]
) -> dict[str, dict[int, Any]] | None:
    if data is None or not years:
        return None

    annual_data, annual_original, annual_dates = {}, {}, {}

    for year in years:
        year_data = data[cast(pd.DatetimeIndex, data.index).year == year][
            "Close"
        ].copy()

        if len(year_data) > 5:
            first_valid_idx = year_data.first_valid_index()

            if first_valid_idx is not None:
                first_value = year_data.loc[first_valid_idx]
                year_data = year_data.ffill().bfill()

                annual_original[year] = year_data.copy()
                annual_dates[year] = year_data.index

                norm_data = ((year_data / first_value) - 1) * 100
                common_year_index = [
                    pd.Timestamp(2000, d.month, d.day) for d in year_data.index
                ]
                annual_data[year] = pd.Series(norm_data.values, index=common_year_index)

    return {
        "normalized": annual_data,
        "original": annual_original,
        "dates": annual_dates,
    }


def create_performance_chart(
    annual_results: dict[str, dict[int, Any]],
    ticker_symbol: str,
    years_to_display: list[int],
    show_percent: bool,
) -> go.Figure:
    annual_data = annual_results["normalized"]
    annual_original = annual_results["original"]
    annual_dates = annual_results["dates"]

    fig = go.Figure()

    for i, year in enumerate(years_to_display):
        if year in annual_data:
            y_data = (
                annual_data[year].values
                if show_percent
                else annual_original[year].values
            )
            dates = cast(pd.DatetimeIndex, annual_data[year].index)
            hover_text = create_hover_text(
                actual_dates=annual_dates[year],
                original_series=annual_original[year],
                year_series=annual_data[year],
            )

            custom_data = [
                [d.strftime("%Y-%m-%d"), p, v]
                for d, p, v in zip(
                    annual_dates[year], annual_original[year], annual_data[year].values
                )
            ]

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=y_data,
                    mode="lines",
                    name=str(year),
                    line=dict(color=COLOR_PALETTE[i % len(COLOR_PALETTE)], width=2),
                    hoverinfo="text",
                    hovertext=hover_text if len(hover_text) == len(dates) else None,
                    customdata=custom_data,
                )
            )

    fig.update_layout(
        title=f"{ticker_symbol.upper()} - Annual {'Performance (%)' if show_percent else 'Value'}",
        xaxis=dict(
            title="Month",
            tickformat="%b",
            tickmode="array",
            tickvals=pd.date_range("2000-01-01", "2000-12-31", freq="MS"),
            ticktext=MONTH_ABBR,
        ),
        yaxis=dict(
            title="Performance (%)" if show_percent else "Value",
            ticksuffix="%" if show_percent else "",
        ),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_hover_text(
    actual_dates: pd.DatetimeIndex, original_series: pd.Series, year_series: pd.Series
) -> list[str]:
    hover_text = []
    for j, (_, perf_value) in enumerate(year_series.items()):
        if j < len(actual_dates):
            actual_date = actual_dates[j]
            value = original_series.iloc[j]
            hover_text.append(
                f"<b>Date</b>: {actual_date.strftime('%Y-%m-%d')}<br>"
                + f"<b>Day</b>: {actual_date.strftime('%A')}<br>"
                + f"<b>Value</b>: {value:.2f}<br>"
                + f"<b>Performance</b>: {perf_value:.2f}%"
            )
    return hover_text


def create_summary_table(
    annual_data: dict[int, pd.Series],
    annual_original: dict[int, pd.Series],
    years_to_display: list[int],
) -> pd.DataFrame | None:
    year_end_perf = {}
    year_end_value = {}

    for year in years_to_display:
        if year in annual_data:
            if not annual_data[year].empty:
                year_end_perf[year] = annual_data[year].iloc[-1]
                year_end_value[year] = annual_original[year].iloc[-1]

    if year_end_perf:
        return pd.DataFrame(
            {
                "Year": list(year_end_perf.keys()),
                "Year-End Value": [f"{p:.2f}" for p in year_end_value.values()],
                "Year Performance (%)": [f"{v:.2f}" for v in year_end_perf.values()],
            }
        ).set_index("Year")

    return None


def process_and_display_data(
    ticker_symbol: str, years_to_display: list[int], show_percent: bool
) -> None:
    with st.spinner(f"Fetching data for {ticker_symbol}..."):
        asset_data = get_asset_data(ticker_symbol, years_to_display)

        if asset_data is not None and years_to_display:
            annual_results = process_annual_data(asset_data, years_to_display)

            if annual_results and annual_results["normalized"]:
                fig = create_performance_chart(
                    annual_results, ticker_symbol, years_to_display, show_percent
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Performance Summary")
                summary_df = create_summary_table(
                    annual_results["normalized"],
                    annual_results["original"],
                    years_to_display,
                )
                if summary_df is not None:
                    st.table(summary_df)
            else:
                st.warning(
                    f"No complete annual data available for {ticker_symbol} in the selected years."
                )
        elif years_to_display:
            st.error(
                f"Failed to retrieve data for {ticker_symbol}. Please check the ticker symbol."
            )
        else:
            st.warning("Please select at least one year to display")


if __name__ == "__main__":
    main()
