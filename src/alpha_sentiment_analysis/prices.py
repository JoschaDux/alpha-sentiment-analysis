from __future__ import annotations

from datetime import timedelta

import pandas as pd
import yfinance as yf


def _nearest_price(prices: pd.Series, date: pd.Timestamp) -> float | None:
    future_prices = prices.loc[prices.index >= date]
    if future_prices.empty:
        return None
    return float(future_prices.iloc[0])


def forward_return(
    ticker: str,
    filing_date: str,
    horizon_days: int = 252,
    benchmark: str = "SPY",
) -> dict[str, float | str | None]:
    start = pd.Timestamp(filing_date) - timedelta(days=7)
    end = pd.Timestamp(filing_date) + timedelta(days=int(horizon_days * 1.8))
    symbols = [ticker, benchmark]
    prices = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(ticker)

    filing_ts = pd.Timestamp(filing_date)
    target_ts = filing_ts + pd.tseries.offsets.BDay(horizon_days)

    stock_start = _nearest_price(prices[ticker].dropna(), filing_ts)
    stock_end = _nearest_price(prices[ticker].dropna(), target_ts)
    bench_start = _nearest_price(prices[benchmark].dropna(), filing_ts)
    bench_end = _nearest_price(prices[benchmark].dropna(), target_ts)

    if None in {stock_start, stock_end, bench_start, bench_end}:
        return {
            "ticker": ticker,
            "filing_date": filing_date,
            "forward_return": None,
            "benchmark_return": None,
            "excess_return": None,
            "outperformed": None,
        }

    stock_return = stock_end / stock_start - 1
    benchmark_return = bench_end / bench_start - 1
    excess_return = stock_return - benchmark_return

    return {
        "ticker": ticker,
        "filing_date": filing_date,
        "forward_return": stock_return,
        "benchmark_return": benchmark_return,
        "excess_return": excess_return,
        "outperformed": int(excess_return > 0),
    }
