"""Tests for feature building and model helpers."""

from pathlib import Path

import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, apps_to_dataframe, row_to_features
from src.utils import DATA_DIR, label_from_score


def test_row_to_features_keys():
    app = {
        "app_name": "Test App",
        "developer": "Test Dev",
        "package_name": "com.testdev.testapp",
        "category": "Tools",
        "downloads": 1000,
        "rating": 4.0,
        "reviews": 50,
        "permissions": ["INTERNET"],
        "description": "Hello world",
        "app_age_days": 100,
    }
    feats = row_to_features(app)
    assert set(FEATURE_COLUMNS).issubset(set(feats.keys()))
    assert feats["package_match"] == 1


def test_label_from_score():
    assert label_from_score(10) == "Genuine"
    assert label_from_score(50) == "Suspicious"
    assert label_from_score(80) == "Likely Fake"


def test_apps_to_dataframe_shape():
    apps = [
        {
            "app_name": "A",
            "developer": "D",
            "package_name": "com.d.a",
            "category": "Tools",
            "downloads": 10,
            "rating": 3.0,
            "reviews": 5,
            "permissions": [],
            "description": "",
            "app_age_days": 10,
        }
    ]
    df = apps_to_dataframe(apps)
    assert list(df.columns) == FEATURE_COLUMNS
    assert len(df) == 1


@pytest.mark.skipif(
    not (DATA_DIR / "synthetic_apps.csv").exists(),
    reason="synthetic data not generated yet",
)
def test_synthetic_csv_has_label():
    df = pd.read_csv(DATA_DIR / "synthetic_apps.csv")
    assert "is_fake" in df.columns
    assert set(df["is_fake"].unique()).issubset({0, 1})
