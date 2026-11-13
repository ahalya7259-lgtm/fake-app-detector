"""
Fake App Detector – Streamlit UI
Run: streamlit run app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import combined_assessment
from src.rules import compute_rule_score
from src.utils import DANGEROUS_PERMISSIONS, validate_app_input

st.set_page_config(
    page_title="Fake App Detector",
    page_icon="🛡️",
    layout="wide",
)

CATEGORIES = [
    "Tools",
    "Games",
    "Social",
    "Communication",
    "Finance",
    "Photography",
    "Productivity",
    "Entertainment",
    "Maps & Navigation",
    "Shopping",
    "Health & Fitness",
    "Other",
]

PERM_OPTIONS = sorted(
    set(DANGEROUS_PERMISSIONS)
    | {
        "INTERNET",
        "ACCESS_NETWORK_STATE",
        "BODY_SENSORS",
        "REQUEST_INSTALL_PACKAGES",
        "ACCESSIBILITY_SERVICE",
    }
)


def assess(app: dict) -> dict:
    rules = compute_rule_score(app)
    return combined_assessment(app, rules)


def risk_color(label: str) -> str:
    return {
        "Genuine": "🟢",
        "Suspicious": "🟡",
        "Likely Fake": "🔴",
    }.get(label, "⚪")


st.title("🛡️ Fake App Detector")
st.caption(
    "Learning project: rule-based checks + a model trained on **synthetic** data. "
    "Not a real security product — do not rely on it for production decisions."
)

tab_single, tab_batch = st.tabs(["Single app", "Batch CSV"])

with tab_single:
    st.subheader("Enter app details")
    c1, c2 = st.columns(2)
    with c1:
        app_name = st.text_input("App name", placeholder="e.g. WhatsApp Plus")
        developer = st.text_input("Developer name", placeholder="e.g. Meta Platforms")
        package_name = st.text_input("Package name", placeholder="com.example.app")
        category = st.selectbox("Category", CATEGORIES)
        downloads = st.number_input("Number of downloads", min_value=0, value=1000, step=100)
    with c2:
        rating = st.slider("Rating", 0.0, 5.0, 4.0, 0.1)
        reviews = st.number_input("Number of reviews", min_value=0, value=50, step=1)
        app_age_days = st.number_input("App age (days since release)", min_value=0, value=90, step=1)
        permissions = st.multiselect("Permissions requested", PERM_OPTIONS, default=["INTERNET"])
    description = st.text_area("App description", height=100, placeholder="Short store description…")

    if st.button("Analyze", type="primary"):
        app = {
            "app_name": app_name,
            "developer": developer,
            "package_name": package_name,
            "category": category,
            "downloads": downloads,
            "rating": rating,
            "reviews": reviews,
            "permissions": permissions,
            "description": description,
            "app_age_days": app_age_days,
        }
        errors = validate_app_input(app)
        if errors:
            for e in errors:
                st.error(e)
        else:
            with st.spinner("Running rules and model…"):
                result = assess(app)
            st.markdown("---")
            st.markdown(f"### {risk_color(result['label'])} **{result['label']}**")
            st.progress(min(100, int(result["final_score"])) / 100.0)
            m1, m2, m3 = st.columns(3)
            m1.metric("Final risk score", f"{result['final_score']:.0f} / 100")
            m2.metric("Rule score", f"{result['rule_score']:.0f}")
            ml = result.get("ml_probability")
            m3.metric("ML fake probability", f"{ml:.0%}" if ml is not None else "N/A")

            st.subheader("Why?")
            if result["reasons"]:
                for r in result["reasons"]:
                    st.markdown(f"- {r}")
            else:
                st.success("No strong risk signals from the rule set.")

with tab_batch:
    st.subheader("Upload a CSV of apps")
    st.markdown(
        "Expected columns: `app_name`, `developer`, `package_name`, `category`, "
        "`downloads`, `rating`, `reviews`, `permissions` (pipe-separated), "
        "`description`, `app_age_days`."
    )
    st.download_button(
        "Download sample CSV",
        data=(ROOT / "data" / "sample_input.csv").read_text(encoding="utf-8"),
        file_name="sample_input.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            df = None
        if df is not None:
            required = [
                "app_name",
                "developer",
                "package_name",
                "category",
                "downloads",
                "rating",
                "reviews",
                "description",
                "app_age_days",
            ]
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                rows_out = []
                progress = st.progress(0)
                for i, row in df.iterrows():
                    perms = str(row.get("permissions", "") or "")
                    perm_list = [p.strip() for p in perms.split("|") if p.strip()]
                    app = {
                        "app_name": row["app_name"],
                        "developer": row["developer"],
                        "package_name": row["package_name"],
                        "category": row["category"],
                        "downloads": int(row["downloads"]),
                        "rating": float(row["rating"]),
                        "reviews": int(row["reviews"]),
                        "permissions": perm_list,
                        "description": str(row.get("description", "")),
                        "app_age_days": int(row["app_age_days"]),
                    }
                    res = assess(app)
                    rows_out.append(
                        {
                            "app_name": app["app_name"],
                            "developer": app["developer"],
                            "label": res["label"],
                            "final_score": res["final_score"],
                            "rule_score": res["rule_score"],
                            "ml_probability": res.get("ml_probability"),
                            "reasons": " | ".join(res["reasons"]),
                        }
                    )
                    progress.progress((i + 1) / len(df))
                out_df = pd.DataFrame(rows_out)
                st.dataframe(out_df, use_container_width=True)
                buf = io.StringIO()
                out_df.to_csv(buf, index=False)
                st.download_button(
                    "Download results CSV",
                    data=buf.getvalue(),
                    file_name="fake_app_results.csv",
                    mime="text/csv",
                )

st.markdown("---")
st.caption("Fake App Detector · synthetic training data · educational use only")
