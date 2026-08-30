# Tweets Sentiment Analysis

This folder contains the production flow for collecting, cleaning, and scoring tweet data.

## Flow

1. Capture raw X posts from the notebook: `tweet_capture.ipynb`
2. Clean and normalize them with `clean_tweets.py`
3. Build sentiment/trading signals with `trading_signals.py`

The notebook is kept separate because it is interactive and meant for extraction. The Python files are the reusable.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Use it

Clean the raw tweet export:

```bash
python clean_tweets.py
```

Or point to a specific file:

```bash
python clean_tweets.py --input raw/tweets.csv --output tweets_clean.parquet
```

Generate trading signals:

```bash
python trading_signals.py 
```

The default interval is `4h` so the latest 2–3 days of data do not collapse into too few buckets.

## Output files

- `tweets_clean.parquet`: cleaned tweet dataset
- `tweet_signals.parquet`: per-tweet signal data
- `trading_signals.parquet`: aggregated time-bucket signal data
- `trading_signals.png`: signal chart

## Notes

- Keep the notebook as the collection layer.
- Use the scripts for repeatable runs
