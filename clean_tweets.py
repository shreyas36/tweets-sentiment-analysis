from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "tweet_id",
    "username",
    "handle",
    "created_at",
    "text",
    "likes",
    "retweets",
    "replies",
    "quotes",
    "bookmarks",
    "views",
    "mentions",
    "hashtags",
    "search_hashtag",
    "tweet_url",
    "dedup_key",
]

COLUMN_ALIASES = {
    "content": "text",
    "timestamp": "created_at",
    "timestamp_utc": "created_at",
    "url": "tweet_url",
    "tweet_url": "tweet_url",
}

TEXT_COLUMNS = {"tweet_id", "username", "handle", "text", "search_hashtag", "tweet_url"}
METRIC_COLUMNS = {"likes", "retweets", "replies", "quotes", "bookmarks", "views"}
TOKEN_PATTERN = re.compile(r"(?<!\w)[@#]([\w\u200c\u200d]+)", re.UNICODE)


def normalize_text(value: object) -> str:
    """Normalize Unicode without dropping Indian-language characters."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = text.replace("\ufeff", "").replace("\u00a0", " ")
    text = "".join(character for character in text if character in "\n\r\t" or not unicodedata.category(character).startswith("C"))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_scalar(value: object) -> str:
    return normalize_text(value)


def _ordered_unique(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = normalize_text(value).lstrip("@#")
        if cleaned and cleaned.casefold() not in seen:
            result.append(cleaned)
            seen.add(cleaned.casefold())
    return result


def extract_tokens(text: str, prefix: str) -> list[str]:
    return _ordered_unique(match.group(1) for match in TOKEN_PATTERN.finditer(text) if match.group(0).startswith(prefix))


def _as_list(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return _ordered_unique(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return _ordered_unique(parsed)
        except json.JSONDecodeError:
            pass
        return _ordered_unique(value.split(","))
    return _ordered_unique([value])


def _metric(value: object) -> int | None:
    if value is None or pd.isna(value) or str(value).strip().casefold() in {"", "n/a", "na", "none"}:
        return None
    try:
        return max(0, int(float(str(value).replace(",", "").strip())))
    except (TypeError, ValueError):
        return None


def _record_from_mapping(record: Mapping[str, object]) -> dict[str, object]:
    flattened = dict(record)
    metrics = flattened.pop("engagement_metrics", {})
    if isinstance(metrics, Mapping):
        for name in METRIC_COLUMNS:
            flattened.setdefault(name, metrics.get(name))
    for source, target in COLUMN_ALIASES.items():
        if target not in flattened and source in flattened:
            flattened[target] = flattened[source]

    text = normalize_text(flattened.get("text", ""))
    username = _clean_scalar(flattened.get("username"))
    handle = _clean_scalar(flattened.get("handle")) or username
    tweet_id = _clean_scalar(flattened.get("tweet_id"))
    created_at = pd.to_datetime(flattened.get("created_at"), utc=True, errors="coerce")
    mentions = _as_list(flattened.get("mentions")) or extract_tokens(text, "@")
    hashtags = _as_list(flattened.get("hashtags")) or extract_tokens(text, "#")
    tweet_url = _clean_scalar(flattened.get("tweet_url"))
    if not tweet_url and tweet_id and handle:
        tweet_url = f"https://x.com/{handle}/status/{tweet_id}"

    identity = tweet_id or f"{handle.casefold()}|{created_at.isoformat()}|{text.casefold()}"
    dedup_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()

    return {
        "tweet_id": tweet_id or None,
        "username": username or None,
        "handle": handle or None,
        "created_at": created_at,
        "text": text or None,
        **{name: _metric(flattened.get(name)) for name in METRIC_COLUMNS},
        "mentions": mentions,
        "hashtags": hashtags,
        "search_hashtag": _clean_scalar(flattened.get("search_hashtag")) or None,
        "tweet_url": tweet_url or None,
        "dedup_key": dedup_key,
    }


def normalize_tweets(records: pd.DataFrame | Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Return a typed dataframe with stable columns and duplicate identities removed."""
    source = records.to_dict(orient="records") if isinstance(records, pd.DataFrame) else records
    normalized = pd.DataFrame([_record_from_mapping(record) for record in source])
    if normalized.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    normalized = normalized.drop_duplicates(subset="dedup_key", keep="last")
    normalized["created_at"] = pd.to_datetime(normalized["created_at"], utc=True, errors="coerce")
    for column in METRIC_COLUMNS:
        normalized[column] = pd.array(normalized[column], dtype="Int64")
    for column in TEXT_COLUMNS:
        normalized[column] = normalized[column].astype("string")
    return normalized[OUTPUT_COLUMNS].sort_values("created_at", na_position="last").reset_index(drop=True)


def load_records(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix == ".json":
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            if all(isinstance(value, list) for value in payload.values()):
                return pd.DataFrame(payload)
            return pd.DataFrame([payload])
        raise ValueError("JSON input must be a list or dictionary")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def clean_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    source = load_records(input_path)
    cleaned = normalize_tweets(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd")
    return len(source), len(cleaned)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    default_inputs = [
        project_dir / "tweets.csv",
        project_dir / "india_stock_tweets.csv",
        project_dir / "raw" / "tweets.csv",
    ]

    parser = argparse.ArgumentParser(description="Normalize tweet exports into a deduplicated Parquet dataset.")
    parser.add_argument("--input", type=Path, action="append", dest="inputs", default=[], help="Input CSV/JSON/Parquet file. Repeat for multiple files.")
    parser.add_argument("--output", type=Path, default=project_dir / "tweets_clean.parquet", help="Destination Parquet file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    inputs = args.inputs or [path for path in [
        project_dir / "tweets.csv",
        project_dir / "india_stock_tweets.csv",
        project_dir / "raw" / "tweets.csv",
    ] if path.exists()]
    if not inputs:
        raise FileNotFoundError("No input tweet file was found in the project directory.")

    frames = [load_records(path) for path in inputs]
    source = pd.concat(frames, ignore_index=True, sort=False)
    cleaned = normalize_tweets(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(args.output, index=False, engine="pyarrow", compression="zstd")
    print(f"Read {len(source):,} rows; wrote {len(cleaned):,} unique tweets to {args.output.name}")


if __name__ == "__main__":
    main()
