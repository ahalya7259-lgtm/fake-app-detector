"""Shared helpers for validation and loading resources."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DANGEROUS_PERMISSIONS = [
    "READ_SMS",
    "SEND_SMS",
    "RECEIVE_SMS",
    "READ_CONTACTS",
    "WRITE_CONTACTS",
    "ACCESS_FINE_LOCATION",
    "ACCESS_COARSE_LOCATION",
    "RECORD_AUDIO",
    "CAMERA",
    "READ_CALL_LOG",
    "WRITE_CALL_LOG",
    "PROCESS_OUTGOING_CALLS",
    "GET_ACCOUNTS",
    "READ_EXTERNAL_STORAGE",
    "WRITE_EXTERNAL_STORAGE",
    "REQUEST_INSTALL_PACKAGES",
    "SYSTEM_ALERT_WINDOW",
    "ACCESSIBILITY_SERVICE",
]

CATEGORY_ALLOWED_PERMISSIONS = {
    "Communication": {"READ_SMS", "SEND_SMS", "RECEIVE_SMS", "READ_CONTACTS", "CAMERA", "RECORD_AUDIO"},
    "Social": {"CAMERA", "RECORD_AUDIO", "READ_CONTACTS", "ACCESS_FINE_LOCATION"},
    "Maps & Navigation": {"ACCESS_FINE_LOCATION", "ACCESS_COARSE_LOCATION", "CAMERA"},
    "Photography": {"CAMERA", "RECORD_AUDIO", "READ_EXTERNAL_STORAGE", "WRITE_EXTERNAL_STORAGE"},
    "Finance": {"CAMERA", "GET_ACCOUNTS"},
    "Health & Fitness": {"ACCESS_FINE_LOCATION", "BODY_SENSORS"},
    "Tools": set(),
    "Productivity": {"GET_ACCOUNTS", "READ_EXTERNAL_STORAGE"},
    "Entertainment": {"RECORD_AUDIO"},
    "Games": set(),
    "Shopping": {"ACCESS_FINE_LOCATION", "CAMERA"},
    "Other": set(),
}

SPAM_WORDS = [
    "free money",
    "hack",
    "unlimited coins",
    "unlimited money",
    "cheat",
    "generator",
    "crack",
    "mod apk",
    "free gift",
    "earn cash",
    "get rich",
    "no verification",
    "guaranteed money",
]


def load_popular_apps() -> pd.DataFrame:
    path = DATA_DIR / "popular_apps.csv"
    return pd.read_csv(path)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def package_tokens(package_name: str) -> list[str]:
    """Split package name into meaningful tokens (ignore com/org/net)."""
    if not package_name:
        return []
    parts = re.split(r"[._]", package_name.lower())
    skip = {"com", "org", "net", "android", "app", "mobile", "io"}
    return [p for p in parts if p and p not in skip and len(p) > 1]


def validate_app_input(data: dict) -> list[str]:
    """Return a list of validation error messages (empty if OK)."""
    errors = []
    if not str(data.get("app_name", "")).strip():
        errors.append("App name is required.")
    if not str(data.get("developer", "")).strip():
        errors.append("Developer name is required.")
    pkg = str(data.get("package_name", "")).strip()
    if not pkg:
        errors.append("Package name is required.")
    elif not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*(\.[a-zA-Z][a-zA-Z0-9_]*)+$", pkg):
        errors.append("Package name should look like com.example.app.")
    try:
        downloads = int(data.get("downloads", 0))
        if downloads < 0:
            errors.append("Downloads cannot be negative.")
    except (TypeError, ValueError):
        errors.append("Downloads must be a whole number.")
    try:
        rating = float(data.get("rating", 0))
        if rating < 0 or rating > 5:
            errors.append("Rating must be between 0 and 5.")
    except (TypeError, ValueError):
        errors.append("Rating must be a number.")
    try:
        reviews = int(data.get("reviews", 0))
        if reviews < 0:
            errors.append("Number of reviews cannot be negative.")
    except (TypeError, ValueError):
        errors.append("Number of reviews must be a whole number.")
    try:
        age = int(data.get("app_age_days", 0))
        if age < 0:
            errors.append("App age (days) cannot be negative.")
    except (TypeError, ValueError):
        errors.append("App age must be a whole number of days.")
    return errors


def label_from_score(score: float) -> str:
    if score < 35:
        return "Genuine"
    if score < 65:
        return "Suspicious"
    return "Likely Fake"
