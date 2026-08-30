from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "tweets_clean.parquet"
SIGNALS_PATH = PROJECT_DIR / "tweet_signals.parquet"
DAILY_PATH = PROJECT_DIR / "daily_trading_signals.parquet"
PLOT_PATH = PROJECT_DIR / "daily_trading_signals.png"
VECTORIZER_PATH = PROJECT_DIR / "tfidf_vectorizer.joblib"
DEFAULT_INTERVAL = "4h"
MIN_RELIABLE_COUNT = 5  # buckets with fewer tweets than this are flagged unreliable, not dropped

# ---------------------------------------------------------------------------
# Sentiment lexicon
# ---------------------------------------------------------------------------
BULLISH_LEXICON: dict[str, float] = {
    "bullish": 1.0, "buy": 0.8, "bought": 0.6, "breakout": 1.0, "support": 0.5,
    "upside": 0.8, "gain": 0.6, "gains": 0.6, "long": 0.5, "call": 0.5, "calls": 0.5,
    "rally": 0.9, "surge": 1.1, "surging": 1.1, "moon": 1.2, "mooning": 1.2,
    "outperform": 0.9, "upgrade": 0.9, "upgraded": 0.9, "accumulate": 0.7,
}
BEARISH_LEXICON: dict[str, float] = {
    "bearish": 1.0, "sell": 0.8, "sold": 0.6, "breakdown": 1.0, "resistance": 0.5,
    "downside": 0.8, "loss": 0.6, "losses": 0.6, "short": 0.5, "put": 0.5, "puts": 0.5,
    "crash": 1.2, "crashing": 1.2, "dump": 1.0, "dumping": 1.0, "plunge": 1.1,
    "underperform": 0.9, "downgrade": 0.9, "downgraded": 0.9, "offload": 0.7,
}
NEGATION_TOKENS = frozenset("not no never n't without hardly barely rarely".split())
INTENSIFIERS: dict[str, float] = {
    "very": 1.5, "extremely": 2.0, "highly": 1.5, "super": 1.5,
    "slightly": 0.5, "somewhat": 0.6, "barely": 0.4,
}
NEGATION_WINDOW = 3 
_SENTIMENT_TOKEN_RE = re.compile(r"[a-zA-Z']+|n't")


def _tokenize_for_sentiment(text: str) -> list[str]:
    return [token.casefold() for token in _SENTIMENT_TOKEN_RE.findall(text)]


def _lexical_score(text: str) -> float:
    tokens = _tokenize_for_sentiment(text)
    bullish_total = 0.0
    bearish_total = 0.0
    for index, token in enumerate(tokens):
        weight = BULLISH_LEXICON.get(token)
        polarity = 1
        if weight is None:
            weight = BEARISH_LEXICON.get(token)
            polarity = -1
        if weight is None:
            continue

        window = tokens[max(0, index - NEGATION_WINDOW):index]
        if any(w in NEGATION_TOKENS for w in window):
            polarity *= -1
        if window and window[-1] in INTENSIFIERS:
            weight *= INTENSIFIERS[window[-1]]

        if polarity > 0:
            bullish_total += weight
        else:
            bearish_total += weight

    denominator = bullish_total + bearish_total
    if denominator == 0:
        return 0.0
    return (bullish_total - bearish_total) / denominator


# ---------------------------------------------------------------------------
# Bot / spam heuristics
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+")
_MENTION_HASHTAG_RE = re.compile(r"[@#]\w+")


def _content_fingerprint(text: str) -> str:
    """Normalize text for near-duplicate detection: strip URLs/mentions/hashtags/punctuation."""
    stripped = _URL_RE.sub("", text)
    stripped = _MENTION_HASHTAG_RE.sub("", stripped)
    stripped = re.sub(r"[^a-z0-9\s]", "", stripped.casefold())
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return hashlib.sha1(stripped.encode("utf-8")).hexdigest()


def compute_spam_weight(
    tweets: pd.DataFrame,
    duplicate_window: str = "1h",
    duplicate_account_threshold: int = 4,
    frequency_window: str = "1h",
    frequency_post_threshold: int = 8,
) -> pd.Series:

    frame = tweets[["handle", "created_at", "text"]].copy()
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
    frame["fingerprint"] = frame["text"].fillna("").map(_content_fingerprint)
    frame["bucket"] = frame["created_at"].dt.floor(duplicate_window)

    # Heuristic 1: distinct-account fan-out of near-identical text within a bucket.
    fanout = (
        frame.groupby(["bucket", "fingerprint"])["handle"]
        .transform(lambda handles: handles.nunique())
    )
    duplicate_weight = np.where(
        fanout > duplicate_account_threshold,
        (duplicate_account_threshold / fanout.clip(lower=1)).clip(lower=0.2),
        1.0,
    )

    # Heuristic 2: unusually high posting frequency for a single account in a short window.
    freq_bucket = frame["created_at"].dt.floor(frequency_window)
    post_counts = frame.assign(_fb=freq_bucket).groupby(["handle", "_fb"])["handle"].transform("size")
    frequency_weight = np.where(
        post_counts > frequency_post_threshold,
        (frequency_post_threshold / post_counts.clip(lower=1)).clip(lower=0.2),
        1.0,
    )

    combined = pd.Series(duplicate_weight, index=frame.index) * pd.Series(frequency_weight, index=frame.index)
    
    return combined.astype("float32")


# ---------------------------------------------------------------------------
# Engagement
# ---------------------------------------------------------------------------
def _engagement_score(frame: pd.DataFrame, spam_weight: pd.Series) -> pd.Series:
    engagement = frame[["likes", "retweets", "replies", "quotes", "bookmarks", "views"]].fillna(0).astype(float)
    weighted = engagement["likes"] + 2 * engagement["retweets"] + engagement["replies"] + engagement["quotes"] + engagement["bookmarks"]
    weighted += np.sqrt(engagement["views"].clip(lower=0))
    weighted *= spam_weight.to_numpy()  # suspicious activity contributes less engagement "weight"
    return np.log1p(weighted)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _bootstrap_interval(values: np.ndarray, seed: int, iterations: int = 500) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values.mean()) if len(values) else 0.0
        return value, value
    if np.allclose(values, values[0]):
        # scipy's BCa bootstrap is undefined for zero-variance samples.
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    try:
        result = stats.bootstrap(
            (values,), np.mean, n_resamples=iterations, method="BCa",
            random_state=rng, confidence_level=0.95,
        )
        return float(result.confidence_interval.low), float(result.confidence_interval.high)
    except Exception:
        # BCa can fail to converge on pathological samples (e.g. near-constant with one
        # outlier); fall back to the plain percentile method rather than crashing the run.
        samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        return float(low), float(high)


def _sample_for_plot(frame: pd.DataFrame, limit: int = 50_000) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    return frame.sample(limit, random_state=42)


TOKEN_PATTERN = r"(?u)\$[A-Za-z]{1,6}\b|[@#]\w+|\b\w\w+\b"


def build_signals(
    tweets: pd.DataFrame,
    max_features: int = 20_000,
    interval: str = DEFAULT_INTERVAL,
    min_reliable_count: int = MIN_RELIABLE_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if tweets.empty:
        raise ValueError("tweets_clean.parquet contains no rows")
    texts = tweets["text"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents=None,
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
        token_pattern=TOKEN_PATTERN,
    )
    tfidf = vectorizer.fit_transform(texts)
    # joblib.dump(vectorizer, VECTORIZER_PATH)  # persist so it can be reused/inspected later
    vocabulary = np.asarray(vectorizer.get_feature_names_out())
    top_indices = np.asarray(tfidf.argmax(axis=1)).ravel()

    spam_weight = compute_spam_weight(tweets)

    signals = tweets[["tweet_id", "created_at", "handle", "search_hashtag", "text"]].copy()
    signals["tfidf_peak_term"] = vocabulary[top_indices]
    signals["tfidf_peak_weight"] = tfidf.max(axis=1).toarray().ravel().astype("float32")
    signals["spam_weight"] = spam_weight.to_numpy()

    signals["text_sentiment"] = texts.map(_lexical_score).astype("float32")

    signals["engagement_score"] = _engagement_score(tweets, spam_weight).astype("float32")
    engagement_scale = signals["engagement_score"] / max(1.0, float(signals["engagement_score"].quantile(0.95)))
    signals["composite_signal"] = (0.65 * signals["text_sentiment"] + 0.35 * engagement_scale.clip(upper=1)).astype("float32")
    signals["signal_confidence"] = (
        signals["tfidf_peak_weight"] * np.sqrt(1 + signals["engagement_score"]) * signals["spam_weight"]
    ).clip(upper=1).astype("float32")

    lower_cut, upper_cut = signals["composite_signal"].quantile([1 / 3, 2 / 3])
    signals["signal_direction"] = np.select(
        [signals["composite_signal"] > upper_cut, signals["composite_signal"] < lower_cut],
        ["bullish", "bearish"],
        default="neutral",
    )

    signals["created_at"] = pd.to_datetime(signals["created_at"], utc=True, errors="coerce")
    signals = signals.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)

    aggregate_rows: list[dict[str, object]] = []
    for bucket_start, group in signals.groupby(signals["created_at"].dt.floor(interval), sort=True):
        values = group["composite_signal"].to_numpy(dtype=float)
        seed = int(hashlib.sha256(str(bucket_start).encode()).hexdigest()[:8], 16)
        low, high = _bootstrap_interval(values, seed=seed)
        aggregate_rows.append({
            "date": bucket_start,
            "tweet_count": len(group),
            "signal_mean": values.mean(),
            "signal_lower_95": low,
            "signal_upper_95": high,
            "bullish_share": float((group["signal_direction"] == "bullish").mean()),
            "bearish_share": float((group["signal_direction"] == "bearish").mean()),
            "avg_spam_weight": float(group["spam_weight"].mean()),
        })
    daily = pd.DataFrame(aggregate_rows)
    daily["reliable"] = daily["tweet_count"] >= min_reliable_count


    rolling_window = max(3, min(20, len(daily) // 5 or 3))
    rolling_mean = daily["tweet_count"].rolling(rolling_window, min_periods=2).mean()
    rolling_std = daily["tweet_count"].rolling(rolling_window, min_periods=2).std()
    daily["volume_zscore"] = ((daily["tweet_count"] - rolling_mean) / rolling_std.replace(0, np.nan)).fillna(0.0)

    return signals, daily


def plot_daily_signals(daily: pd.DataFrame, output_path: Path) -> None:
    sampled = _sample_for_plot(daily)
    figure, (signal_axis, volume_axis) = plt.subplots(
        2, 1, figsize=(12, 7), constrained_layout=True, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    signal_axis.plot(sampled["date"], sampled["signal_mean"], color="#0b7285", linewidth=1.5, label="Composite signal")
    signal_axis.fill_between(
        sampled["date"], sampled["signal_lower_95"], sampled["signal_upper_95"],
        color="#74c0fc", alpha=0.35, label="95% BCa bootstrap interval",
    )
    unreliable = sampled[~sampled["reliable"]]
    if not unreliable.empty:
        signal_axis.scatter(
            unreliable["date"], unreliable["signal_mean"], marker="x", color="#e8590c",
            s=25, label=f"Low tweet count (<{MIN_RELIABLE_COUNT})", zorder=3,
        )
    signal_axis.axhline(0, color="#495057", linewidth=0.8)
    signal_axis.set(ylabel="Signal", title=f"{DEFAULT_INTERVAL.upper()} tweet trading signal")
    signal_axis.legend(frameon=False, loc="upper left")

    volume_axis.bar(sampled["date"], sampled["tweet_count"], color="#868e96", width=0.03, label="Tweet count")
    volume_axis.set(xlabel="Time", ylabel="Tweets")
    volume_twin = volume_axis.twinx()
    volume_twin.plot(sampled["date"], sampled["volume_zscore"], color="#f08c00", linewidth=1.2, label="Volume z-score")
    volume_twin.set_ylabel("Volume z-score")

    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError("tweets_clean.parquet is missing; run clean_tweets.py first")
    tweets = pd.read_parquet(
        INPUT_PATH,
        columns=["tweet_id", "created_at", "handle", "search_hashtag", "text",
                 "likes", "retweets", "replies", "quotes", "bookmarks", "views"],
    )
    signals, daily = build_signals(tweets, interval=DEFAULT_INTERVAL)
    signals.to_parquet(SIGNALS_PATH, index=False, engine="pyarrow", compression="zstd")
    daily.to_parquet(DAILY_PATH, index=False, engine="pyarrow", compression="zstd")
    plot_daily_signals(daily, PLOT_PATH)
    unreliable_buckets = int((~daily["reliable"]).sum())
    print(
        f"Processed {len(signals):,} tweets into {SIGNALS_PATH.name} and {len(daily):,} "
        f"{DEFAULT_INTERVAL} aggregates ({unreliable_buckets} flagged low-confidence)."
    )


if __name__ == "__main__":
    main()