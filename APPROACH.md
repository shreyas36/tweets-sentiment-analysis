# Approach

We fetch raw tweet data in the notebook, then clean it into a consistent table, score each tweet for sentiment and engagement, and aggregate the results into time buckets before plotting the final chart.

1. Fetch data
   - The notebook collects raw X posts and keeps the extraction logic separate from the reusable code.
   - Raw fields include text, timestamps, handles, and engagement metrics.

2. Clean data
   - `clean_tweets.py` normalizes timestamps, removes duplicates, cleans text, and extracts mentions/hashtags.
   - The output is a clean parquet dataset used for downstream analysis.

3. Build signals
   - `trading_signals.py` scores each tweet using sentiment, engagement, and spam-weight adjustments.
   - It combines these into a composite signal and classifies tweets as bullish, bearish, or neutral.

4. Aggregate and plot
   - Results are grouped into time buckets to smooth short-term noise.
   - The script calculates tweet counts, confidence bands, and volume z-scores.
   - A final chart shows sentiment versus activity over time.

This keeps collection interactive, cleaning repeatable, and the final chart easy to interpret.
