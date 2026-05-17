from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")

NEGATIVE_WORDS = {
    "adverse",
    "decline",
    "decrease",
    "default",
    "deteriorate",
    "difficult",
    "failure",
    "loss",
    "losses",
    "negative",
    "risk",
    "risks",
    "uncertain",
    "weakness",
}

POSITIVE_WORDS = {
    "benefit",
    "efficient",
    "favorable",
    "gain",
    "gains",
    "growth",
    "improve",
    "improved",
    "opportunity",
    "positive",
    "profit",
    "strong",
}

UNCERTAINTY_WORDS = {
    "approximately",
    "contingent",
    "depend",
    "depends",
    "fluctuate",
    "may",
    "might",
    "possible",
    "uncertain",
    "uncertainty",
    "variable",
    "whether",
}

LITIGIOUS_WORDS = {
    "claim",
    "claims",
    "complaint",
    "court",
    "legal",
    "litigation",
    "plaintiff",
    "regulatory",
    "settlement",
    "sue",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract first-pass textual features from downloaded 10-K filings."
    )
    parser.add_argument(
        "--filings",
        default="data/interim/filings_metadata_sp500.csv",
        help="Filing metadata CSV produced by the EDGAR downloader.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/text_features.csv",
        help="Output CSV for text features.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=1000,
        help="Mark filings with fewer words than this as suspicious.",
    )
    return parser.parse_args()


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    text = soup.get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


def read_filing_text(path: Path) -> str:
    raw = path.read_text(errors="ignore")
    lower_start = raw[:5000].lower()
    if "<html" in lower_start or "<document" in lower_start or "<sec-document" in lower_start:
        return html_to_text(raw)
    return re.sub(r"\s+", " ", raw).strip()


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def word_share(counter: Counter[str], vocabulary: set[str], total: int) -> float:
    if total == 0:
        return 0.0
    return sum(counter[word] for word in vocabulary) / total


def count_sentences(text: str) -> int:
    sentence_marks = text.count(".") + text.count("?") + text.count("!")
    return max(sentence_marks, 1)


def text_features(text: str) -> dict[str, float]:
    tokens = tokenize(text)
    counter = Counter(tokens)
    total = len(tokens)
    unique = len(counter)
    sentence_count = count_sentences(text)
    avg_word_length = sum(len(token) for token in tokens) / total if total else 0.0

    return {
        "word_count": float(total),
        "unique_word_count": float(unique),
        "type_token_ratio": unique / total if total else 0.0,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": total / sentence_count if total else 0.0,
        "log_word_count": math.log1p(total),
        "negative_share": word_share(counter, NEGATIVE_WORDS, total),
        "positive_share": word_share(counter, POSITIVE_WORDS, total),
        "uncertainty_share": word_share(counter, UNCERTAINTY_WORDS, total),
        "litigious_share": word_share(counter, LITIGIOUS_WORDS, total),
    }


def add_change_features(features: pd.DataFrame) -> pd.DataFrame:
    ordered = features.sort_values(["ticker", "filing_date"]).copy()
    change_columns = [
        "negative_share",
        "positive_share",
        "uncertainty_share",
        "litigious_share",
        "log_word_count",
        "word_count",
    ]
    for column in change_columns:
        if column in ordered.columns:
            ordered[f"{column}_yoy_change"] = ordered.groupby("ticker")[column].diff()
    return ordered


def resolve_local_path(local_path: str, project_root: Path) -> Path:
    path = Path(str(local_path))
    if path.is_absolute():
        return path
    return project_root / path


def build_feature_row(filing: dict, project_root: Path, min_words: int) -> dict:
    row = {
        "ticker": filing.get("ticker"),
        "cik": filing.get("cik"),
        "accession_number": filing.get("accession_number"),
        "filing_date": filing.get("filing_date"),
        "report_date": filing.get("report_date"),
        "form": filing.get("form"),
        "local_path": filing.get("local_path"),
        "feature_status": "ok",
        "feature_error": "",
    }

    path = resolve_local_path(filing.get("local_path", ""), project_root)
    if not path.exists():
        row["feature_status"] = "failed"
        row["feature_error"] = f"Local filing path does not exist: {path}"
        return row

    text = read_filing_text(path)
    features = text_features(text)
    row.update(features)

    if features["word_count"] < min_words:
        row["feature_status"] = "suspicious"
        row["feature_error"] = f"Very short extracted text: {features['word_count']:.0f} words"

    return row


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    filings = pd.read_csv(args.filings)

    if "download_status" in filings.columns:
        filings = filings.loc[filings["download_status"].fillna("downloaded") == "downloaded"]

    filings = filings.dropna(subset=["ticker", "filing_date", "accession_number", "local_path"])
    if filings.empty:
        raise ValueError("No usable downloaded filing rows found in the metadata CSV.")

    rows = []
    for index, filing in enumerate(filings.to_dict(orient="records"), start=1):
        ticker = filing.get("ticker")
        filing_date = filing.get("filing_date")
        print(f"[{index}/{len(filings)}] Extracting text features for {ticker} {filing_date}")
        try:
            rows.append(build_feature_row(filing, project_root, args.min_words))
        except Exception as error:
            rows.append(
                {
                    "ticker": filing.get("ticker"),
                    "cik": filing.get("cik"),
                    "accession_number": filing.get("accession_number"),
                    "filing_date": filing.get("filing_date"),
                    "report_date": filing.get("report_date"),
                    "form": filing.get("form"),
                    "local_path": filing.get("local_path"),
                    "feature_status": "failed",
                    "feature_error": str(error),
                }
            )

    features = add_change_features(pd.DataFrame(rows))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)

    print(f"Saved {len(features)} text-feature rows to {output_path}")
    print(features["feature_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
