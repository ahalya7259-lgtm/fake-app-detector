"""Feature engineering for the ML model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.utils import (
    CATEGORY_ALLOWED_PERMISSIONS,
    DANGEROUS_PERMISSIONS,
    SPAM_WORDS,
    load_popular_apps,
    normalize_text,
    package_tokens,
)

FEATURE_COLUMNS = [
    "name_similarity_max",
    "dev_mismatch",
    "dangerous_perm_count",
    "rating",
    "log_reviews",
    "log_downloads",
    "app_age_days",
    "downloads_per_day",
    "spam_word_count",
    "package_match",
    "rating_high_reviews_low",
]


def _max_name_similarity(app_name: str, popular_names: list[str]) -> float:
    name_n = normalize_text(app_name)
    if not name_n or not popular_names:
        return 0.0
    return float(max(fuzz.token_set_ratio(name_n, normalize_text(n)) for n in popular_names))


def _dev_mismatch_flag(app_name: str, developer: str, popular: pd.DataFrame) -> int:
    name_n = normalize_text(app_name)
    dev_n = normalize_text(developer)
    for _, row in popular.iterrows():
        sim = fuzz.token_set_ratio(name_n, normalize_text(row["app_name"]))
        if sim >= 85:
            official = normalize_text(row["developer"])
            if official not in dev_n and dev_n not in official:
                return 1
    return 0


def _dangerous_count(category: str, permissions: list[str]) -> int:
    allowed = CATEGORY_ALLOWED_PERMISSIONS.get(category, set())
    count = 0
    for p in permissions or []:
        p_up = str(p).upper().replace(" ", "_")
        if p_up in DANGEROUS_PERMISSIONS and p_up not in allowed:
            count += 1
    return count


def _spam_count(description: str) -> int:
    text = normalize_text(description)
    return sum(1 for w in SPAM_WORDS if w in text)


def _package_match_flag(app_name: str, developer: str, package_name: str) -> int:
    tokens = package_tokens(package_name)
    if not tokens:
        return 0
    name_n = normalize_text(app_name).replace(" ", "")
    dev_n = normalize_text(developer).replace(" ", "")
    return int(any(t in name_n or t in dev_n or name_n in t or dev_n in t for t in tokens))


def row_to_features(app: dict, popular: pd.DataFrame | None = None) -> dict:
    """Convert one app dict into a numeric feature dict."""
    if popular is None:
        popular = load_popular_apps()
    popular_names = popular["app_name"].tolist()

    downloads = max(0, int(app.get("downloads", 0) or 0))
    reviews = max(0, int(app.get("reviews", 0) or 0))
    age = max(1, int(app.get("app_age_days", 365) or 365))
    rating = float(app.get("rating", 0) or 0)

    return {
        "name_similarity_max": _max_name_similarity(app.get("app_name", ""), popular_names),
        "dev_mismatch": _dev_mismatch_flag(
            app.get("app_name", ""), app.get("developer", ""), popular
        ),
        "dangerous_perm_count": _dangerous_count(
            app.get("category", "Other"), app.get("permissions") or []
        ),
        "rating": rating,
        "log_reviews": float(np.log1p(reviews)),
        "log_downloads": float(np.log1p(downloads)),
        "app_age_days": float(age),
        "downloads_per_day": float(downloads) / age,
        "spam_word_count": _spam_count(app.get("description", "")),
        "package_match": _package_match_flag(
            app.get("app_name", ""),
            app.get("developer", ""),
            app.get("package_name", ""),
        ),
        "rating_high_reviews_low": int(rating >= 4.5 and reviews < 30),
    }


def apps_to_dataframe(apps: list[dict]) -> pd.DataFrame:
    popular = load_popular_apps()
    rows = [row_to_features(a, popular) for a in apps]
    return pd.DataFrame(rows)[FEATURE_COLUMNS]
