"""
Generate a synthetic dataset of genuine and fake Android app profiles
for training the Fake App Detector model.

This data is SYNTHETIC — not from the real Play Store.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import DATA_DIR, load_popular_apps

CATEGORIES = [
    "Tools", "Games", "Social", "Communication", "Finance",
    "Photography", "Productivity", "Entertainment",
    "Maps & Navigation", "Shopping", "Other",
]

GENUINE_DESCRIPTIONS = [
    "Simple and reliable utility for everyday use. Privacy focused.",
    "Official app from the developer. Regular updates and support.",
    "Organize your tasks and notes with a clean interface.",
    "Stream music offline. Subscription optional.",
    "Navigate with offline maps and real-time traffic.",
    "Secure messaging with end-to-end encryption.",
    "Edit photos with filters and basic tools.",
    "Track fitness goals and daily activity.",
]

FAKE_DESCRIPTIONS = [
    "Get free money now!!! Unlimited coins generator hack no verification.",
    "WhatsApp plus MOD APK free gift earn cash guaranteed money.",
    "Hack any account crack password free money transfer.",
    "Unlimited money cheat for games. Free gift cards inside!!!",
    "Earn cash fast get rich scheme no verification required $$$",
]


def _genuine_row(rng: random.Random, popular: pd.DataFrame) -> dict:
    if rng.random() < 0.4:
        row = popular.sample(1, random_state=rng.randint(0, 10_000)).iloc[0]
        name = row["app_name"]
        developer = row["developer"]
        pkg = row["package_hint"]
    else:
        name = rng.choice(
            ["NotePad Pro", "Weather Now", "Focus Timer", "Budget Lite", "Photo Grid", "FitTrack"]
        )
        developer = rng.choice(
            ["Bright Soft Labs", "Northwind Apps", "Cedar Mobile", "Pixel Forge Inc"]
        )
        pkg = f"com.{developer.split()[0].lower()}.{name.replace(' ', '').lower()}"

    category = rng.choice(CATEGORIES)
    age = rng.randint(60, 2000)
    downloads = int(10 ** rng.uniform(2.5, 7.5))
    reviews = max(50, int(downloads * rng.uniform(0.01, 0.08)))
    rating = round(rng.uniform(3.5, 4.8), 1)
    perms = ["INTERNET", "ACCESS_NETWORK_STATE"]
    if category == "Photography":
        perms += ["CAMERA", "READ_EXTERNAL_STORAGE"]
    if category == "Maps & Navigation":
        perms += ["ACCESS_FINE_LOCATION"]
    if category == "Communication":
        perms += ["CAMERA", "RECORD_AUDIO"]

    return {
        "app_name": name,
        "developer": developer,
        "package_name": pkg,
        "category": category,
        "downloads": downloads,
        "rating": rating,
        "reviews": reviews,
        "permissions": "|".join(perms),
        "description": rng.choice(GENUINE_DESCRIPTIONS),
        "app_age_days": age,
        "is_fake": 0,
    }


def _fake_row(rng: random.Random, popular: pd.DataFrame) -> dict:
    row = popular.sample(1, random_state=rng.randint(0, 10_000)).iloc[0]
    if rng.random() < 0.7:
        name = row["app_name"] + rng.choice([" Pro", " Plus", " 2024", " Free", ""])
        if rng.random() < 0.5:
            name = name.replace(" ", "")
    else:
        name = rng.choice(["FlashLight Ultra", "Cleaner Master", "Battery Saver Max"])

    developer = rng.choice(
        ["Unknown Dev", "apk-world", "free-tools-ltd", "xInstaller", "mod-hub"]
    )
    pkg = f"com.{rng.choice(['temp', 'xyz', 'appstore', 'download'])}.tool{rng.randint(1, 999)}"

    category = rng.choice(["Tools", "Games", "Other", "Entertainment"])
    age = rng.randint(1, 40)
    downloads = int(10 ** rng.uniform(4.5, 6.5))
    reviews = rng.randint(0, 25)
    rating = round(rng.uniform(4.6, 5.0), 1)
    perms = [
        "INTERNET", "READ_SMS", "SEND_SMS", "READ_CONTACTS",
        "ACCESS_FINE_LOCATION", "SYSTEM_ALERT_WINDOW",
    ]
    if rng.random() < 0.5:
        perms.append("READ_CALL_LOG")

    return {
        "app_name": name,
        "developer": developer,
        "package_name": pkg,
        "category": category,
        "downloads": downloads,
        "rating": rating,
        "reviews": reviews,
        "permissions": "|".join(perms),
        "description": rng.choice(FAKE_DESCRIPTIONS),
        "app_age_days": age,
        "is_fake": 1,
    }


def generate(n_genuine: int = 800, n_fake: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    np.random.seed(seed)
    popular = load_popular_apps()
    rows = [_genuine_row(rng, popular) for _ in range(n_genuine)]
    rows += [_fake_row(rng, popular) for _ in range(n_fake)]
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def main():
    df = generate()
    out = DATA_DIR / "synthetic_apps.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows → {out}")
    print(df["is_fake"].value_counts().to_dict())


if __name__ == "__main__":
    main()
