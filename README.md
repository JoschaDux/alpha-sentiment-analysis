# Alpha Sentiment Analysis

This project tests whether textual information in SEC Form 10-K filings is
related to future stock returns and realized volatility.

The pipeline downloads annual reports from SEC EDGAR, extracts interpretable
textual features, joins them to future stock-price outcomes from `yfinance`, and
trains baseline models with time-based evaluation.

For the full methodology, equations, pipeline explanation, and first results,
see the project write-up:

[alpha_sentiment_analysis.pdf](alpha_sentiment_analysis.pdf)

## Research Question

Can textual features extracted from annual 10-K reports predict future
market-adjusted returns or future realized volatility after the filing date?

This is a research project, not investment advice. The first results are
interpreted conservatively and are meant to establish a reproducible baseline.

## Current Status

Implemented:

- SEC EDGAR 10-K downloader
- S&P 500 ticker universe file
- return-label construction with `yfinance`
- text feature extraction from downloaded filings
- model-dataset merge
- baseline model training
- PDF methodology report

First baseline models:

- logistic regression for 252-day outperformance
- random forest classifier for 252-day outperformance
- ridge regression for 252-day excess return
- random forest regression for 252-day excess return
- ridge regression for 252-day realized volatility
- random forest regression for 252-day realized volatility

## First Results

Using a time split where filings before 2023 are used for training and filings
from 2023 onward are used for testing:

| Task | Model | Test Metric | Value |
|---|---:|---:|---:|
| Outperformance | Logistic regression | ROC-AUC | 0.514 |
| Outperformance | Random forest | ROC-AUC | 0.491 |
| Excess return | Ridge regression | MAE | 0.289 |
| Excess return | Random forest | MAE | 0.292 |
| Realized volatility | Ridge regression | MAE | 0.104 |
| Realized volatility | Random forest | MAE | 0.101 |

These first-pass dictionary and document-length features do not yet show a
strong predictive signal for one-year excess returns. The project therefore
serves as a transparent baseline for richer feature engineering, such as
section-level extraction, stronger financial dictionaries, and embeddings.

## Repository Structure

```text
alpha_sentiment_analysis/
  Code/
    edgar_10k_download.py        # download 10-K filings from SEC EDGAR
    build_return_labels.py       # calculate returns and realized volatility
    build_text_features.py       # extract textual features from filings
    build_model_dataset.py       # merge text features and return labels
    train_baseline_models.py     # train baseline models and save results
  config/
    tickers_sp500.txt            # S&P 500 ticker universe
    tickers_sample.txt           # smaller test universe
  data/
    raw/filings/                 # downloaded 10-K files
    interim/                     # filing metadata
    processed/                   # return labels, text features, model dataset
  reports/
    model_results/               # model metrics and predictions
    figures/                     # optional figures
  alpha_sentiment_analysis.tex   # LaTeX source
  alpha_sentiment_analysis.pdf   # project write-up
  requirements.txt
```

## Reproduce The Pipeline

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set a SEC user agent:

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
```

Download 10-K filings:

```bash
python Code/edgar_10k_download.py \
  --tickers-file config/tickers_sp500.txt \
  --limit 10 \
  --metadata-output data/interim/filings_metadata_sp500.csv \
  --sleep-seconds 0.75 \
  --max-retries 8
```

Build return labels:

```bash
python Code/build_return_labels.py \
  --filings data/interim/filings_metadata_sp500.csv \
  --output data/processed/return_labels.csv \
  --benchmark SPY \
  --horizons 30 90 180 252
```

Build text features:

```bash
python Code/build_text_features.py \
  --filings data/interim/filings_metadata_sp500.csv \
  --output data/processed/text_features.csv
```

Merge the modeling dataset:

```bash
python Code/build_model_dataset.py \
  --text-features data/processed/text_features.csv \
  --return-labels data/processed/return_labels.csv \
  --output data/processed/model_dataset.csv
```

Train baseline models:

```bash
python Code/train_baseline_models.py \
  --dataset data/processed/model_dataset.csv \
  --output-dir reports/model_results \
  --test-year 2023 \
  --horizon 252
```

## Notes

- Data files are not intended to be committed if they become large.
- SEC EDGAR does not require an account, but scripts should identify themselves
  with a user agent.
- The modeling split is time-based to reduce look-ahead bias.
- The current results are preliminary and should be treated as a baseline.
