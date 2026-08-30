from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clean_tweets import normalize_tweets


def test_normalize_tweets_deduplicates_and_preserves_fields() -> None:
    rows = [
        {
            "tweet_id": "1",
            "username": "alpha",
            "handle": "alpha",
            "created_at": "2026-08-30 09:15:00+00:00",
            "text": "Bullish breakout support #nifty @trader",
            "likes": 10,
            "retweets": 1,
            "replies": 2,
            "quotes": 0,
            "bookmarks": 0,
            "views": 200,
            "mentions": ["trader"],
            "hashtags": ["nifty"],
            "search_hashtag": "#nifty",
            "tweet_url": "https://x.com/alpha/status/1",
        },
        {
            "tweet_id": "1",
            "username": "alpha",
            "handle": "alpha",
            "created_at": "2026-08-30 09:15:00+00:00",
            "text": "Bullish breakout support #nifty @trader",
            "likes": 10,
            "retweets": 1,
            "replies": 2,
            "quotes": 0,
            "bookmarks": 0,
            "views": 200,
            "mentions": ["trader"],
            "hashtags": ["nifty"],
            "search_hashtag": "#nifty",
            "tweet_url": "https://x.com/alpha/status/1",
        },
    ]

    cleaned = normalize_tweets(pd.DataFrame(rows))

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["tweet_id"] == "1"
    assert cleaned.iloc[0]["handle"] == "alpha"
    assert cleaned.iloc[0]["hashtags"] == ["nifty"]
