"""Rule-based risk scoring for suspicious / fake Android apps."""

from __future__ import annotations

from rapidfuzz import fuzz

from src.utils import (
    CATEGORY_ALLOWED_PERMISSIONS,
    DANGEROUS_PERMISSIONS,
    SPAM_WORDS,
    load_popular_apps,
    normalize_text,
    package_tokens,
)


def _name_impersonation(app_name: str, developer: str, popular: list[dict]) -> tuple[int, str | None]:
    """High similarity to a known app but different developer."""
    name_n = normalize_text(app_name)
    dev_n = normalize_text(developer)
    best_score = 0
    best_match = None
    for row in popular:
        score = fuzz.token_set_ratio(name_n, normalize_text(row["app_name"]))
        if score > best_score:
            best_score = score
            best_match = row
    if best_match and best_score >= 85:
        official_dev = normalize_text(best_match["developer"])
        if official_dev not in dev_n and dev_n not in official_dev:
            points = 35 if best_score >= 92 else 25
            return points, (
                f"App name is very similar to '{best_match['app_name']}' "
                f"(similarity {best_score}%) but developer is '{developer}', "
                f"not '{best_match['developer']}'."
            )
    return 0, None


def _dangerous_permissions(category: str, permissions: list[str]) -> tuple[int, str | None]:
    allowed = CATEGORY_ALLOWED_PERMISSIONS.get(category, set())
    risky = []
    for p in permissions:
        p_up = p.upper().replace(" ", "_")
        if p_up in DANGEROUS_PERMISSIONS and p_up not in allowed:
            if p_up in {"READ_SMS", "SEND_SMS", "RECEIVE_SMS", "READ_CONTACTS", "READ_CALL_LOG"}:
                risky.append(p_up)
            elif category in {"Tools", "Games", "Other", "Entertainment"}:
                risky.append(p_up)
    if not risky:
        return 0, None
    points = min(30, 10 + 5 * len(risky))
    return points, (
        f"Requests sensitive permissions unusual for category '{category}': "
        + ", ".join(risky)
        + "."
    )


def _rating_review_mismatch(rating: float, reviews: int) -> tuple[int, str | None]:
    if rating >= 4.7 and reviews < 20:
        return 20, f"Very high rating ({rating}) with only {reviews} reviews."
    if rating >= 4.5 and reviews < 10:
        return 25, f"Near-perfect rating ({rating}) with very few reviews ({reviews})."
    if rating >= 4.8 and reviews < 50:
        return 15, f"Extremely high rating ({rating}) with relatively few reviews ({reviews})."
    return 0, None


def _download_age_mismatch(downloads: int, age_days: int) -> tuple[int, str | None]:
    if age_days <= 7 and downloads >= 100_000:
        return 25, f"App is only {age_days} days old but claims {downloads:,} downloads."
    if age_days <= 30 and downloads >= 1_000_000:
        return 30, f"App is only {age_days} days old but claims {downloads:,} downloads."
    if age_days <= 14 and downloads >= 50_000:
        return 18, f"Very new app ({age_days} days) with high download count ({downloads:,})."
    return 0, None


def _spam_description(description: str) -> tuple[int, str | None]:
    text = normalize_text(description)
    hits = [w for w in SPAM_WORDS if w in text]
    if not hits:
        if description.count("!") >= 4 or description.count("$$$") >= 1:
            return 10, "Description uses excessive punctuation or money symbols often seen in spam."
        return 0, None
    points = min(25, 10 + 5 * len(hits))
    return points, f"Description contains suspicious phrases: {', '.join(hits)}."


def _package_mismatch(app_name: str, developer: str, package_name: str) -> tuple[int, str | None]:
    tokens = package_tokens(package_name)
    if not tokens:
        return 5, "Package name could not be parsed into meaningful tokens."
    name_n = normalize_text(app_name).replace(" ", "")
    dev_n = normalize_text(developer).replace(" ", "")
    matched = any(t in name_n or t in dev_n or name_n in t or dev_n in t for t in tokens)
    if not matched:
        return 15, (
            f"Package '{package_name}' does not clearly match app name or developer brand."
        )
    return 0, None


def compute_rule_score(app: dict) -> dict:
    """
    Compute rule-based risk score (0–100) and list of reasons.

    app keys: app_name, developer, package_name, category, downloads,
              rating, reviews, permissions (list), description, app_age_days
    """
    popular_df = load_popular_apps()
    popular = popular_df.to_dict(orient="records")

    total = 0
    reasons: list[str] = []

    checks = [
        _name_impersonation(app.get("app_name", ""), app.get("developer", ""), popular),
        _dangerous_permissions(app.get("category", "Other"), app.get("permissions") or []),
        _rating_review_mismatch(float(app.get("rating", 0)), int(app.get("reviews", 0))),
        _download_age_mismatch(int(app.get("downloads", 0)), int(app.get("app_age_days", 365))),
        _spam_description(app.get("description", "")),
        _package_mismatch(
            app.get("app_name", ""),
            app.get("developer", ""),
            app.get("package_name", ""),
        ),
    ]

    for points, reason in checks:
        if points and reason:
            total += points
            reasons.append(reason)

    total = min(100, total)
    return {
        "rule_score": total,
        "reasons": reasons,
    }
