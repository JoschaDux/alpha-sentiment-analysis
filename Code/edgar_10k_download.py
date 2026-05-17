"""
Download recent 10-K filings from SEC EDGAR without an account or API key.

Before running, set a user agent so the SEC can identify your script:

    export SEC_USER_AGENT="Your Name your.email@example.com"

Then run

    python Code/simple_edgar_10k_download.py --tickers-file config/tickers_sample.txt --limit 10

Outputs:

    data/raw/filings/              downloaded 10-K HTML files
    data/interim/filings_metadata.csv
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests
from requests import HTTPError


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{document}"
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


def sec_session() -> requests.Session:
    user_agent = os.getenv("SEC_USER_AGENT")
    if not user_agent:
        raise RuntimeError(
            "Please set SEC_USER_AGENT first, for example:\n"
            'export SEC_USER_AGENT="Your Name your.email@example.com"'
        )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
    )
    return session


def sec_get(session: requests.Session, url: str, sleep_seconds: float, max_retries: int) -> requests.Response:
    for attempt in range(max_retries + 1):
        time.sleep(sleep_seconds)
        response = session.get(url, timeout=30)
        if response.status_code not in RETRY_STATUS_CODES:
            response.raise_for_status()
            return response

        if attempt == max_retries:
            response.raise_for_status()

        wait_seconds = sleep_seconds * (2 ** attempt) + 1
        print(
            f"SEC returned {response.status_code}. "
            f"Retrying in {wait_seconds:.1f}s: {url}"
        )
        time.sleep(wait_seconds)

    raise RuntimeError("Unreachable retry state.")


def sec_get_json(session: requests.Session, url: str, sleep_seconds: float, max_retries: int) -> dict:
    response = sec_get(session, url, sleep_seconds=sleep_seconds, max_retries=max_retries)
    return response.json()


def sec_get_bytes(session: requests.Session, url: str, sleep_seconds: float, max_retries: int) -> bytes:
    response = sec_get(session, url, sleep_seconds=sleep_seconds, max_retries=max_retries)
    return response.content


def load_ticker_map(session: requests.Session, sleep_seconds: float, max_retries: int) -> pd.DataFrame:
    data = sec_get_json(session, SEC_TICKERS_URL, sleep_seconds, max_retries)
    ticker_map = pd.DataFrame(data.values())
    ticker_map["ticker"] = ticker_map["ticker"].str.upper()
    ticker_map["cik_str"] = ticker_map["cik_str"].astype(int).astype(str).str.zfill(10)
    return ticker_map.rename(columns={"cik_str": "cik"})


def cik_for_ticker(ticker_map: pd.DataFrame, ticker: str) -> str:
    match = ticker_map.loc[ticker_map["ticker"] == ticker.upper()]
    if match.empty:
        raise ValueError(f"Could not find CIK for ticker: {ticker}")
    return str(match.iloc[0]["cik"])


def recent_10k_filings(
    session: requests.Session,
    ticker: str,
    cik: str,
    limit: int,
    sleep_seconds: float,
    max_retries: int,
) -> list[dict]:
    submissions_url = SEC_SUBMISSIONS_URL.format(cik=cik)
    submissions = sec_get_json(session, submissions_url, sleep_seconds, max_retries)
    recent = pd.DataFrame(submissions["filings"]["recent"])
    ten_ks = recent.loc[recent["form"].isin(["10-K", "10-K/A"])].head(limit)

    filings = []
    for row in ten_ks.to_dict(orient="records"):
        accession_number = row["accessionNumber"]
        accession_compact = accession_number.replace("-", "")
        document = row["primaryDocument"]
        filing_url = SEC_ARCHIVE_URL.format(
            cik_int=int(cik),
            accession=accession_compact,
            document=document,
        )
        filings.append(
            {
                "ticker": ticker.upper(),
                "cik": cik,
                "accession_number": accession_number,
                "filing_date": row["filingDate"],
                "report_date": row.get("reportDate"),
                "form": row["form"],
                "primary_document": document,
                "filing_url": filing_url,
            }
        )
    return filings


def download_filing(
    session: requests.Session,
    filing: dict,
    output_dir: Path,
    sleep_seconds: float,
    max_retries: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{filing['ticker']}_{filing['filing_date']}_"
        f"{filing['accession_number'].replace('-', '')}.html"
    )
    output_path = output_dir / filename

    if not output_path.exists():
        content = sec_get_bytes(session, filing["filing_url"], sleep_seconds, max_retries)
        output_path.write_bytes(content)

    filing["local_path"] = str(output_path)
    filing["download_status"] = "downloaded"
    filing["download_error"] = ""
    return filing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download 10-K filings from SEC EDGAR.")
    parser.add_argument("--tickers", nargs="+", help="Ticker symbols, e.g. AAPL MSFT.")
    parser.add_argument(
        "--tickers-file",
        help="Optional text file with one ticker per line. Lines beginning with # are ignored.",
    )
    parser.add_argument("--limit", type=int, default=3, help="Number of recent 10-Ks per ticker.")
    parser.add_argument("--output-dir", default="data/raw/filings")
    parser.add_argument("--metadata-output", default="data/interim/filings_metadata.csv")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.35,
        help="Pause between SEC requests. Increase this if SEC returns 429 or 503.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Number of retries for temporary SEC errors such as 429 or 503.",
    )
    args = parser.parse_args()
    if not args.tickers and not args.tickers_file:
        parser.error("Provide either --tickers or --tickers-file.")
    return args


def read_tickers(args: argparse.Namespace) -> list[str]:
    tickers = list(args.tickers or [])
    if args.tickers_file:
        for line in Path(args.tickers_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                tickers.append(line)
    return sorted({ticker.upper().replace(".", "-") for ticker in tickers})


def main() -> None:
    args = parse_args()
    session = sec_session()
    ticker_map = load_ticker_map(session, args.sleep_seconds, args.max_retries)
    output_dir = Path(args.output_dir)
    tickers = read_tickers(args)

    downloaded = []
    for index, ticker in enumerate(tickers, start=1):
        print(f"[{index}/{len(tickers)}] Processing {ticker}")
        try:
            cik = cik_for_ticker(ticker_map, ticker)
            filings = recent_10k_filings(
                session,
                ticker,
                cik,
                args.limit,
                args.sleep_seconds,
                args.max_retries,
            )
        except (ValueError, HTTPError, requests.RequestException) as error:
            print(f"Skipping {ticker}: {error}")
            downloaded.append(
                {
                    "ticker": ticker,
                    "download_status": "failed",
                    "download_error": str(error),
                }
            )
            continue

        for filing in filings:
            try:
                downloaded.append(
                    download_filing(
                        session,
                        filing,
                        output_dir,
                        args.sleep_seconds,
                        args.max_retries,
                    )
                )
            except (HTTPError, requests.RequestException) as error:
                print(f"Failed {filing['ticker']} {filing['filing_date']}: {error}")
                filing["local_path"] = ""
                filing["download_status"] = "failed"
                filing["download_error"] = str(error)
                downloaded.append(filing)

    metadata = pd.DataFrame(downloaded)
    metadata_output = Path(args.metadata_output)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_output, index=False)

    print(f"Downloaded or found {len(metadata)} filings.")
    print(f"Metadata saved to {metadata_output}")
    print(f"Filing files saved to {output_dir}")


if __name__ == "__main__":
    main()
