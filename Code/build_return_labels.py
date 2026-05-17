from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build forward-return and volatility labels for downloaded 10-K filings."
    )
    parser.add_argument(
        "--filings",
        default="data/interim/filings_metadata_sp500.csv",
        help="Filing metadata CSV produced by the EDGAR downloader.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/return_labels.csv",
        help="Output CSV for return labels.",
    )
    parser.add_argument(
        "--benchmark",
        default="SPY",
        help="Benchmark ticker used for market-adjusted returns.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[30, 90, 180, 252],
        help="Forward trading-day horizons to calculate.",
    )
    parser.add_argument(
        "--min-date-buffer-days",
        type=int,
        default=10,
        help="Calendar-day buffer before the first filing date for price download.",
    )
    return parser.parse_args()


def normalize_ticker_for_yfinance(ticker: str) -> str:
    return ticker.upper().replace(".", "-")


def extract_close_prices(downloaded: pd.DataFrame | pd.Series, ticker: str) -> pd.Series:
    if downloaded.empty:
        return pd.Series(dtype=float)

    if isinstance(downloaded, pd.Series):
        close = downloaded
    elif isinstance(downloaded.columns, pd.MultiIndex):
        close = downloaded["Close"][ticker]
    else:
        close = downloaded["Close"]

    close = close.dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.astype(float)


def download_close_prices(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError as error:
        raise ImportError(
            "The build_return_labels.py script requires yfinance. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from error

    downloaded = yf.download(
        ticker,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    return extract_close_prices(downloaded, ticker)


def first_trading_index_on_or_after(prices: pd.Series, date: pd.Timestamp) -> int | None:
    positions = np.flatnonzero(prices.index >= date)
    if len(positions) == 0:
        return None
    return int(positions[0])


def horizon_stats(prices: pd.Series, filing_date: pd.Timestamp, horizon: int) -> dict[str, float | None]:
    start_index = first_trading_index_on_or_after(prices, filing_date)
    if start_index is None:
        return {"return": None, "realized_volatility": None, "start_date": None, "end_date": None}

    end_index = start_index + horizon
    if end_index >= len(prices):
        return {
            "return": None,
            "realized_volatility": None,
            "start_date": prices.index[start_index].date().isoformat(),
            "end_date": None,
        }

    start_price = float(prices.iloc[start_index])
    end_price = float(prices.iloc[end_index])
    forward_return = end_price / start_price - 1

    window = prices.iloc[start_index : end_index + 1].pct_change().dropna()
    realized_volatility = float(window.std(ddof=0) * np.sqrt(252)) if len(window) else None

    return {
        "return": float(forward_return),
        "realized_volatility": realized_volatility,
        "start_date": prices.index[start_index].date().isoformat(),
        "end_date": prices.index[end_index].date().isoformat(),
    }


def build_labels_for_ticker(
    ticker: str,
    filings: pd.DataFrame,
    benchmark_prices: pd.Series,
    horizons: list[int],
    min_date_buffer_days: int,
) -> list[dict]:
    filing_dates = pd.to_datetime(filings["filing_date"])
    start = filing_dates.min() - timedelta(days=min_date_buffer_days)
    end = pd.Timestamp.today().normalize() + timedelta(days=1)

    yf_ticker = normalize_ticker_for_yfinance(ticker)
    stock_prices = download_close_prices(yf_ticker, start, end)

    rows = []
    for filing in filings.to_dict(orient="records"):
        filing_date = pd.Timestamp(filing["filing_date"])
        row = {
            "ticker": ticker,
            "accession_number": filing.get("accession_number"),
            "filing_date": filing["filing_date"],
            "price_ticker": yf_ticker,
            "label_status": "ok",
            "label_error": "",
        }

        if stock_prices.empty:
            row["label_status"] = "failed"
            row["label_error"] = f"No price data from yfinance for {yf_ticker}"
            rows.append(row)
            continue

        for horizon in horizons:
            stock = horizon_stats(stock_prices, filing_date, horizon)
            benchmark = horizon_stats(benchmark_prices, filing_date, horizon)

            stock_return = stock["return"]
            benchmark_return = benchmark["return"]
            excess_return = (
                stock_return - benchmark_return
                if stock_return is not None and benchmark_return is not None
                else None
            )

            row[f"price_start_date_{horizon}d"] = stock["start_date"]
            row[f"price_end_date_{horizon}d"] = stock["end_date"]
            row[f"forward_return_{horizon}d"] = stock_return
            row[f"benchmark_return_{horizon}d"] = benchmark_return
            row[f"excess_return_{horizon}d"] = excess_return
            row[f"outperformed_{horizon}d"] = (
                int(excess_return > 0) if excess_return is not None else None
            )
            row[f"realized_volatility_{horizon}d"] = stock["realized_volatility"]

        rows.append(row)

    return rows


def main() -> None:
    args = parse_args()
    filings = pd.read_csv(args.filings)

    if "download_status" in filings.columns:
        filings = filings.loc[filings["download_status"].fillna("downloaded") == "downloaded"]

    filings = filings.dropna(subset=["ticker", "filing_date", "accession_number"]).copy()
    filings["ticker"] = filings["ticker"].astype(str).str.upper()
    filings["filing_date"] = pd.to_datetime(filings["filing_date"]).dt.date.astype(str)

    if filings.empty:
        raise ValueError("No usable filing rows found in the metadata CSV.")

    first_date = pd.to_datetime(filings["filing_date"]).min() - timedelta(
        days=args.min_date_buffer_days
    )
    end_date = pd.Timestamp.today().normalize() + timedelta(days=1)
    benchmark_prices = download_close_prices(args.benchmark, first_date, end_date)
    if benchmark_prices.empty:
        raise ValueError(f"No benchmark price data found for {args.benchmark}.")

    rows = []
    grouped = filings.sort_values(["ticker", "filing_date"]).groupby("ticker", sort=True)
    for index, (ticker, ticker_filings) in enumerate(grouped, start=1):
        print(f"[{index}/{grouped.ngroups}] Building labels for {ticker}")
        try:
            rows.extend(
                build_labels_for_ticker(
                    ticker=ticker,
                    filings=ticker_filings,
                    benchmark_prices=benchmark_prices,
                    horizons=args.horizons,
                    min_date_buffer_days=args.min_date_buffer_days,
                )
            )
        except Exception as error:
            print(f"Failed {ticker}: {error}")
            for filing in ticker_filings.to_dict(orient="records"):
                rows.append(
                    {
                        "ticker": ticker,
                        "accession_number": filing.get("accession_number"),
                        "filing_date": filing["filing_date"],
                        "price_ticker": normalize_ticker_for_yfinance(ticker),
                        "label_status": "failed",
                        "label_error": str(error),
                    }
                )

    labels = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output_path, index=False)

    print(f"Saved {len(labels)} return-label rows to {output_path}")
    print(labels["label_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
