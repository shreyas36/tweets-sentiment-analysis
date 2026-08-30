from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trading_signals import build_signals


def test_build_signals_uses_shorter_buckets_for_recent_data() -> None:
    timestamps = pd.to_datetime(
        [
            "2026-08-30 09:15:00+00:00",
            "2026-08-30 10:45:00+00:00",
            "2026-08-30 11:30:00+00:00",
            "2026-08-31 00:20:00+00:00",
        ],
        utc=True,
    )
    tweets = pd.DataFrame(
        {
            "tweet_id": [1, 2, 3, 4],
            "created_at": timestamps,
            "handle": ["a", "a", "b", "c"],
            "search_hashtag": ["#nifty", "#nifty", "#nifty", "#nifty"],
            "text": [
                "bullish breakout support",
                "bullish buy gains",
                "bearish sell losses",
                "breakout support long",
            ],
            "likes": [10, 20, 15, 8],
            "retweets": [1, 0, 2, 1],
            "replies": [0, 0, 1, 0],
            "quotes": [0, 0, 0, 1],
            "bookmarks": [0, 0, 0, 0],
            "views": [1000, 2000, 1500, 500],
        }
    )

    _, aggregates = build_signals(tweets, interval="1h")

    assert len(aggregates) >= 4
    assert aggregates["date"].dt.floor("h").nunique() == len(aggregates)
