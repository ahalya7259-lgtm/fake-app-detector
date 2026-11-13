"""Train, load, and predict with the fake-app ML model."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS, apps_to_dataframe, row_to_features
from src.utils import MODELS_DIR, load_popular_apps

MODEL_PATH = MODELS_DIR / "fake_app_model.joblib"
META_PATH = MODELS_DIR / "model_meta.joblib"


def train_from_csv(
    csv_path: Path,
    model_type: str = "random_forest",
    random_state: int = 42,
) -> dict:
    """Train on synthetic CSV with columns matching feature inputs + is_fake."""
    df = pd.read_csv(csv_path)

    apps = []
    labels = []
    for _, row in df.iterrows():
        perms = str(row.get("permissions", "") or "")
        perm_list = [p.strip() for p in perms.split("|") if p.strip()]
        apps.append(
            {
                "app_name": row["app_name"],
                "developer": row["developer"],
                "package_name": row["package_name"],
                "category": row["category"],
                "downloads": int(row["downloads"]),
                "rating": float(row["rating"]),
                "reviews": int(row["reviews"]),
                "permissions": perm_list,
                "description": row.get("description", ""),
                "app_age_days": int(row["app_age_days"]),
            }
        )
        labels.append(int(row["is_fake"]))

    X = apps_to_dataframe(apps)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    if model_type == "logistic":
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
                ),
            ]
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds, digits=3)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    meta = {
        "model_type": model_type,
        "roc_auc": float(auc),
        "feature_columns": FEATURE_COLUMNS,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    joblib.dump(meta, META_PATH)

    print(f"Model type: {model_type}")
    print(f"ROC-AUC: {auc:.4f}")
    print(report)
    print(f"Saved → {MODEL_PATH}")
    return meta


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run data generation and training (see README)."
        )
    return joblib.load(MODEL_PATH)


def predict_proba_fake(app: dict) -> float:
    """Return probability that the app is fake (0–1)."""
    model = load_model()
    popular = load_popular_apps()
    feats = row_to_features(app, popular)
    X = pd.DataFrame([feats])[FEATURE_COLUMNS]
    return float(model.predict_proba(X)[0, 1])


def combined_assessment(app: dict, rule_result: dict) -> dict:
    """Combine rule score (0–100) and ML probability into a final label."""
    from src.utils import label_from_score

    rule_score = float(rule_result.get("rule_score", 0))
    reasons = list(rule_result.get("reasons") or [])

    try:
        ml_prob = predict_proba_fake(app)
        ml_score = ml_prob * 100
        ml_ok = True
    except FileNotFoundError:
        ml_prob = None
        ml_score = rule_score
        ml_ok = False
        reasons.append("ML model not loaded; result is rule-based only.")

    final_score = 0.55 * rule_score + 0.45 * ml_score
    final_score = float(min(100, max(0, final_score)))
    label = label_from_score(final_score)

    if ml_ok and ml_prob is not None and ml_prob >= 0.7:
        reasons.append(f"ML model estimates {ml_prob:.0%} probability of being fake.")

    return {
        "rule_score": rule_score,
        "ml_probability": ml_prob,
        "final_score": round(final_score, 1),
        "label": label,
        "reasons": reasons,
    }
