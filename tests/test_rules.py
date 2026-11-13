"""Tests for rule-based scoring."""

from src.rules import compute_rule_score


def test_impersonation_raises_score():
    app = {
        "app_name": "WhatsApp",
        "developer": "apk-world",
        "package_name": "com.xyz.fakewa",
        "category": "Communication",
        "downloads": 1000,
        "rating": 4.0,
        "reviews": 100,
        "permissions": ["INTERNET"],
        "description": "Chat app",
        "app_age_days": 100,
    }
    result = compute_rule_score(app)
    assert result["rule_score"] > 0
    assert any("similar" in r.lower() or "whatsapp" in r.lower() for r in result["reasons"])


def test_spam_description():
    app = {
        "app_name": "Tool",
        "developer": "Bright Soft Labs",
        "package_name": "com.bright.tool",
        "category": "Tools",
        "downloads": 1000,
        "rating": 4.0,
        "reviews": 200,
        "permissions": ["INTERNET"],
        "description": "Get free money now and hack accounts with unlimited coins",
        "app_age_days": 200,
    }
    result = compute_rule_score(app)
    assert result["rule_score"] >= 10
    assert any("suspicious phrases" in r.lower() or "free money" in r.lower() for r in result["reasons"])


def test_high_rating_few_reviews():
    app = {
        "app_name": "My Utility",
        "developer": "Cedar Mobile",
        "package_name": "com.cedar.myutility",
        "category": "Tools",
        "downloads": 500,
        "rating": 4.9,
        "reviews": 5,
        "permissions": ["INTERNET"],
        "description": "A simple utility.",
        "app_age_days": 100,
    }
    result = compute_rule_score(app)
    assert result["rule_score"] >= 15
    assert any("rating" in r.lower() for r in result["reasons"])


def test_clean_app_low_score():
    app = {
        "app_name": "NotePad Pro",
        "developer": "Bright Soft Labs",
        "package_name": "com.bright.notepadpro",
        "category": "Productivity",
        "downloads": 50000,
        "rating": 4.2,
        "reviews": 1200,
        "permissions": ["INTERNET", "ACCESS_NETWORK_STATE"],
        "description": "Organize your tasks and notes with a clean interface.",
        "app_age_days": 500,
    }
    result = compute_rule_score(app)
    assert result["rule_score"] < 35
