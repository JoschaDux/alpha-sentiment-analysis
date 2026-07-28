from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{document}"


@dataclass(frozen=True)
class Filing:
    ticker: str
    cik: str
    accession_number: str
    filing_date: str
    report_date: str | None
    form: str
    primary_document: str
    filing_url: str
    local_path: str | None = None


class EdgarClient:
    """Small SEC EDGAR client for company metadata and 10-K downloads."""

    def __init__(self, user_agent: str | None = None, sleep_seconds: float = 0.12) -> None:
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT")
        if not self.user_agent:
            raise ValueError(
                "Set SEC_USER_AGENT, for example: "
                "'Your Name your.email@example.com'. The SEC requires a user agent."
            )
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )

    def _get(self, url: str) -> requests.Response:
        time.sleep(self.sleep_seconds)
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response

    def ticker_map(self) -> pd.DataFrame:
        response = self._get(SEC_TICKERS_URL)
        rows = response.json().values()
        frame = pd.DataFrame(rows)
        frame["ticker"] = frame["ticker"].str.upper()
        frame["cik_str"] = frame["cik_str"].astype(int).astype(str).str.zfill(10)
        return frame.rename(columns={"cik_str": "cik"})

    def cik_for_ticker(self, ticker: str) -> str:
        tickers = self.ticker_map()
        match = tickers.loc[tickers["ticker"] == ticker.upper()]
        if match.empty:
            raise ValueError(f"No CIK found for ticker {ticker!r}.")
        return str(match.iloc[0]["cik"])

    def company_submissions(self, cik: str) -> dict:
        response = self._get(SEC_SUBMISSIONS_URL.format(cik=cik.zfill(10)))
        return response.json()

    def ten_k_filings(self, ticker: str, limit: int = 10) -> list[Filing]:
        cik = self.cik_for_ticker(ticker)
        submissions = self.company_submissions(cik)
        recent = submissions["filings"]["recent"]
        rows = pd.DataFrame(recent)
        rows = rows.loc[rows["form"].isin(["10-K", "10-K/A"])].head(limit)

        filings: list[Filing] = []
        for row in rows.to_dict(orient="records"):
            accession = row["accessionNumber"]
            accession_compact = accession.replace("-", "")
            document = row["primaryDocument"]
            filing_url = SEC_ARCHIVES_URL.format(
                cik_int=str(int(cik)),
                accession=accession_compact,
                document=document,
            )
            filings.append(
                Filing(
                    ticker=ticker.upper(),
                    cik=cik,
                    accession_number=accession,
                    filing_date=row["filingDate"],
                    report_date=row.get("reportDate"),
                    form=row["form"],
                    primary_document=document,
                    filing_url=filing_url,
                )
            )
        return filings

    def download_filing(self, filing: Filing, output_dir: Path) -> Filing:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{filing.ticker}_{filing.filing_date}_"
            f"{filing.accession_number.replace('-', '')}.html"
        )
        output_path = output_dir / filename
        if not output_path.exists():
            response = self._get(filing.filing_url)
            output_path.write_bytes(response.content)
        return Filing(**{**filing.__dict__, "local_path": str(output_path)})

    def download_10k_filings(
        self,
        tickers: Iterable[str],
        output_dir: Path,
        limit: int = 5,
    ) -> pd.DataFrame:
        records: list[dict] = []
        for ticker in tickers:
            for filing in self.ten_k_filings(ticker, limit=limit):
                downloaded = self.download_filing(filing, output_dir)
                records.append(downloaded.__dict__)
        return pd.DataFrame(records)
