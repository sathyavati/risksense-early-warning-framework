"""RiskSense demo: one framework, two domains.

Loads the two trained models exported from the Colab notebook
(fraud.joblib and student.joblib) and scores a single case.
"""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="RiskSense", layout="centered")


@st.cache_resource
def load(name):
    return joblib.load(name)


# ---------------------------------------------------------------- domain configs
HIGH_RISK = ["shopping_net", "misc_net", "grocery_pos"]

FRAUD_RULES = [
    ("R1 amount > 5x usual AND new device", lambda d: (d.amt_ratio > 5) & (d.new_device == 1), 0.9),
    ("R2 amount > 5x usual AND night", lambda d: (d.amt_ratio > 5) & (d.is_night == 1), 0.8),
    ("R3 new device AND login-to-buy < 30s", lambda d: (d.new_device == 1) & (d.login_gap_s < 30), 0.6),
    ("R4 night AND high-risk category AND amount > 200",
     lambda d: (d.is_night == 1) & d.category.isin(HIGH_RISK) & (d.amt > 200), 0.6),
    ("R5 10+ transactions in 24h", lambda d: d.txn_24h >= 10, 0.4),
    ("R6 first-time category AND amount > 3x usual", lambda d: (d.new_category == 1) & (d.amt_ratio > 3), 0.4),
]
FRAUD_TIERS = [(0, "Allow"), (0.3, "Notify customer"), (0.5, "Request OTP"), (0.8, "Block and investigate")]

STUDENT_RULES = [
    ("S1 first grade < 8 AND past failure", lambda d: (d.G1 < 8) & (d.failures >= 1), 0.9),
    ("S2 first grade < 10 AND 8+ absences", lambda d: (d.G1 < 10) & (d.absences >= 8), 0.8),
    ("S3 two or more past failures", lambda d: d.failures >= 2, 0.6),
    ("S4 first grade < 10 AND no plan for higher education", lambda d: (d.G1 < 10) & (d.higher == 0), 0.6),
    ("S5 first grade < 10", lambda d: d.G1 < 10, 0.4),
    ("S6 lowest study time AND 6+ absences", lambda d: (d.studytime == 1) & (d.absences >= 6), 0.4),
]
STUDENT_TIERS = [(0, "No action"), (0.3, "Monitor"), (0.5, "Advisor check-in"), (0.8, "Intensive support")]

LABELS = {
    "amt": "transaction amount", "amt_ratio": "amount vs customer's usual", "dist_km": "distance to merchant",
    "is_night": "night-time", "hour": "hour of day", "txn_24h": "transactions in last 24h",
    "secs_since_last": "time since last transaction", "new_category": "first purchase in category",
    "city_pop": "city population", "new_device": "new device", "login_gap_s": "login-to-purchase gap",
    "category": "merchant category",
    "G1": "first-period grade", "failures": "past failures", "absences": "absences",
    "studytime": "study time", "higher": "higher-education plans", "parent_edu": "parental education",
}


# ---------------------------------------------------------------- generic scoring (same for both domains)
def score_frame(bundle, rules, tiers, df):
    """Score every row of df: rule score, ML score, final score and action tier."""
    X = np.asarray(bundle["pre"].transform(df[bundle["num"] + bundle["cat"]]), dtype=float)
    ml = bundle["model"].predict_proba(X)[:, 1]
    rule = np.zeros(len(df))
    fired = [[] for _ in range(len(df))]
    for name, cond, level in rules:
        hit = np.asarray(cond(df), dtype=bool)
        rule = np.where(hit, np.maximum(rule, level), rule)
        for i in np.flatnonzero(hit):
            fired[i].append((name, level))
    final = np.maximum(rule, ml)
    tier_index = np.searchsorted([cut for cut, _ in tiers], final, side="right") - 1
    return X, ml, rule, final, tier_index, fired


def score(bundle, rules, tiers, row, shown):
    """Score one case and explain it."""
    X, ml, rule, final, tier_index, fired = score_frame(bundle, rules, tiers, pd.DataFrame([row]))
    ml, model = float(ml[0]), bundle["model"]

    # per-feature contributions (SHAP values)
    names = [c.split("__")[1] for c in bundle["pre"].get_feature_names_out()]
    if bundle["kind"] == "xgb":
        contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
    else:
        contrib = model.coef_[0] * (X[0] - bundle["bg_mean"])

    # explain only with inputs the viewer can see and change on this page
    reasons = []
    if ml >= 0.3:
        for i in np.argsort(-contrib):
            # stop at three reasons, and skip effects under 5% of the strongest one
            if contrib[i] <= 0 or contrib[i] < 0.05 * contrib.max() or len(reasons) == 3:
                break
            name = names[i]
            if name in shown:
                reasons.append(LABELS.get(name, name))
            else:
                for c in bundle["cat"]:
                    # a one-hot column counts only when it is the category actually selected
                    if c in shown and name == f"{c}_{row[c]}":
                        reasons.append(f"{LABELS.get(c, c)}: {row[c]}")

    return {"ml": ml, "rule": float(rule[0]), "final": float(final[0]), "fired": fired[0],
            "tier": tiers[tier_index[0]][1], "tier_index": int(tier_index[0]), "reasons": reasons}


def show(result):
    box = [st.success, st.info, st.warning, st.error][result["tier_index"]]
    box(f"Decision: {result['tier']}")
    a, b, c = st.columns(3)
    a.metric("Rule score", f"{result['rule']:.2f}")
    b.metric("ML score", f"{result['ml']:.2f}")
    c.metric("Final score", f"{result['final']:.2f}")
    st.markdown("**Rules fired**")
    if result["fired"]:
        for name, level in result["fired"]:
            st.markdown(f"- {name} (risk level {level})")
    else:
        st.markdown("- none")
    st.markdown("**Top model reasons**")
    if result["reasons"]:
        st.markdown("- " + "\n- ".join(result["reasons"]))
    elif result["ml"] < 0.3:
        st.markdown("- none: the model sees low risk")
    else:
        st.markdown("- none among the inputs shown")


def fraud_features(fb, inp):
    """Turn raw transaction fields into the model's feature columns."""
    df = pd.DataFrame({c: fb["defaults"][c] for c in fb["num"] + fb["cat"]}, index=inp.index)
    df["amt"] = inp["amount"]
    df["amt_ratio"] = inp["amount"] / inp["usual_amount"]
    df["hour"] = inp["hour"]
    df["is_night"] = ((inp["hour"] >= 22) | (inp["hour"] <= 3)).astype(int)
    df["category"] = inp["category"].astype(str)
    df["new_device"] = inp["new_device"]
    df["login_gap_s"] = inp["login_gap_s"]
    df["txn_24h"] = inp["txn_24h"]
    df["new_category"] = inp["new_category"]
    return df


def to_flag(col):
    return col.astype(str).str.strip().str.lower().isin(["1", "1.0", "true", "yes", "y"]).astype(int)


USUAL_AMOUNT = 70.0   # average normal transaction in the training data
FRAUD_SHOWN = {"amt", "amt_ratio", "hour", "is_night", "category", "new_device", "login_gap_s", "txn_24h", "new_category"}

# ---------------------------------------------------------------- page
st.title("RiskSense")
st.caption("One early-warning framework, two domains. The scoring code is shared; "
           "only the rules, features and action tiers differ.")

fraud_tab, student_tab, batch_tab, cost_tab, results_tab = st.tabs(
    ["Fraud check", "Student risk", "Batch upload", "Cost analysis", "Results"])

fb = load("fraud.joblib")
sb = load("student.joblib")
cats = fb["categories"]["category"]

with fraud_tab:
    amount = st.number_input("Transaction amount", min_value=1.0, value=45.0, key="amt_fraud")
    with st.expander("More details about this transaction"):
        left, right = st.columns(2)
        usual = left.number_input("Customer's usual amount", min_value=1.0, value=USUAL_AMOUNT, key="usual_fraud")
        cat = right.selectbox("Merchant category", cats, index=cats.index("grocery_pos") if "grocery_pos" in cats else 0,
                              key="cat_fraud")
        hour = left.slider("Hour of day", 0, 23, 14, key="hour_fraud")
        n24 = right.slider("Transactions in last 24 hours", 0, 30, 3, key="n24_fraud")
        gap = left.slider("Seconds from login to purchase", 1, 600, 150, key="gap_fraud")
        dev = right.checkbox("New device", False, key="dev_fraud")
        newcat = right.checkbox("First purchase in this category", False, key="newcat_fraud")

    inp = pd.DataFrame([{"amount": amount, "usual_amount": usual, "hour": hour, "category": cat,
                         "new_device": int(dev), "login_gap_s": gap, "txn_24h": n24, "new_category": int(newcat)}])
    row = fraud_features(fb, inp).iloc[0].to_dict()
    st.divider()
    show(score(fb, FRAUD_RULES, FRAUD_TIERS, row, FRAUD_SHOWN))
    st.caption("Model: XGBoost. Device and login fields were simulated in training. "
               "Inputs not shown here use typical values from the training data, and reasons are drawn from the inputs shown.")

with student_tab:
    st.markdown("Enter a student's details. The decision updates as you change them.")
    left, right = st.columns(2)
    g1 = left.slider("First-period grade (out of 20)", 0, 20, 14, key="g1_student")
    fail = right.slider("Past failures", 0, 3, 0, key="fail_student")
    absn = left.slider("Absences", 0, 30, 2, key="abs_student")
    study = right.slider("Study time band (1 = lowest)", 1, 4, 3, key="study_student")
    higher = left.checkbox("Plans higher education", True, key="higher_student")

    row = dict(sb["defaults"])
    row.update(G1=g1, failures=fail, absences=absn, studytime=study, higher=int(higher))
    st.divider()
    show(score(sb, STUDENT_RULES, STUDENT_TIERS, row, {"G1", "failures", "absences", "studytime", "higher"}))
    st.caption("Model: Logistic Regression. A flag only triggers an offer of support. "
               "Inputs not shown here use typical values from the training data, and reasons are drawn from the inputs shown.")

with batch_tab:
    st.markdown("Upload a CSV of transactions and score them all at once. Only the `amount` column is required.")
    template = pd.DataFrame({
        "amount": [45, 120, 250, 900, 30, 640],
        "usual_amount": [70, 90, 70, 70, 40, 80],
        "hour": [14, 10, 23, 2, 19, 1],
        "category": ["grocery_pos", "gas_transport", "shopping_net", "shopping_net", "food_dining", "misc_net"],
        "new_device": [0, 0, 0, 1, 0, 1],
        "login_gap_s": [150, 200, 90, 12, 180, 20],
        "txn_24h": [3, 2, 4, 6, 1, 11],
        "new_category": [0, 0, 0, 1, 0, 0]})
    st.download_button("Download a sample CSV", template.to_csv(index=False), "sample_transactions.csv", "text/csv")
    up = st.file_uploader("Transactions CSV", type="csv")
    if up is not None:
        try:
            raw = pd.read_csv(up)
            raw.columns = [c.strip().lower() for c in raw.columns]
            if "amount" not in raw.columns:
                st.error("The file needs a column named `amount`.")
            else:
                fill = {"usual_amount": USUAL_AMOUNT, "hour": 12, "category": fb["defaults"]["category"],
                        "new_device": 0, "login_gap_s": fb["defaults"]["login_gap_s"],
                        "txn_24h": fb["defaults"]["txn_24h"], "new_category": 0}
                missing = [c for c in fill if c not in raw.columns]
                inp = raw.copy()
                for c in missing:
                    inp[c] = fill[c]
                for c in ["amount", "usual_amount", "hour", "login_gap_s", "txn_24h"]:
                    inp[c] = pd.to_numeric(inp[c], errors="coerce")
                bad = inp[["amount", "usual_amount", "hour", "login_gap_s", "txn_24h"]].isna().any(axis=1) | (inp["usual_amount"] <= 0)
                inp["new_device"], inp["new_category"] = to_flag(inp["new_device"]), to_flag(inp["new_category"])
                inp = inp[~bad].reset_index(drop=True)
                if bad.any():
                    st.warning(f"Skipped {int(bad.sum())} row(s) with missing or non-numeric values.")
                if missing:
                    st.info("Columns not in the file were filled with typical values: " + ", ".join(missing))
                if len(inp):
                    _, ml, rule, final, tier_index, fired = score_frame(fb, FRAUD_RULES, FRAUD_TIERS, fraud_features(fb, inp))
                    out = inp[["amount", "hour", "category"]].copy()
                    out["rule score"], out["ML score"], out["final score"] = rule.round(2), ml.round(2), final.round(2)
                    out["decision"] = [FRAUD_TIERS[i][1] for i in tier_index]
                    out["rules fired"] = [", ".join(n.split()[0] for n, _ in f) or "none" for f in fired]
                    counts = out["decision"].value_counts()
                    cols = st.columns(4)
                    for col, (_, label) in zip(cols, FRAUD_TIERS):
                        col.metric(label, int(counts.get(label, 0)))
                    st.dataframe(out.sort_values("final score", ascending=False), hide_index=True)
                    st.download_button("Download scored file", out.to_csv(index=False), "scored_transactions.csv", "text/csv")
        except Exception as err:
            st.error(f"Could not read that file: {err}")

with cost_tab:
    st.markdown("Which alert threshold costs the bank least? It depends on what a missed fraud and a false alarm each cost.")
    if not os.path.exists("fraud_scores.csv"):
        st.info("Add `fraud_scores.csv` to the repository to enable this tab.")
    else:
        fs = pd.read_csv("fraud_scores.csv")
        left, right = st.columns(2)
        c_miss = left.slider("Cost of one missed fraud", 50, 2000, 500, step=50, key="c_miss")
        c_alarm = right.slider("Cost of one false alarm", 1, 100, 10, key="c_alarm")
        which = st.radio("Score used", ["ML only", "Hybrid (rules + ML)"], horizontal=True, key="cost_score")
        s, y = fs["ml" if which == "ML only" else "hybrid"].to_numpy(), fs["y"].to_numpy()
        grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
        missed = np.array([((s < t) & (y == 1)).sum() for t in grid])
        alarms = np.array([((s >= t) & (y == 0)).sum() for t in grid])
        cost = missed * c_miss + alarms * c_alarm
        best = int(cost.argmin())
        at_half = int(np.flatnonzero(grid == 0.5)[0])
        a, b, c = st.columns(3)
        a.metric("Cheapest threshold", f"{grid[best]:.2f}")
        b.metric("Total cost there", f"{cost[best]:,}")
        c.metric("Total cost at 0.50", f"{cost[at_half]:,}")
        st.markdown(f"At {grid[best]:.2f}: **{missed[best]}** frauds missed and **{alarms[best]:,}** false alarms. "
                    f"At 0.50: **{missed[at_half]}** missed and **{alarms[at_half]:,}** false alarms.")
        st.line_chart(pd.DataFrame({"Alert threshold": grid, "Total cost": cost}), x="Alert threshold", y="Total cost")
        st.caption(f"Computed on the held-out test set: {len(fs):,} transactions, {int(y.sum())} fraud. "
                   "Costs are in the same currency units as the transaction amounts and are assumptions you set.")

with results_tab:
    st.markdown("Results from the project notebook, on held-out test data.")
    if os.path.exists("results_all.csv"):
        st.dataframe(pd.read_csv("results_all.csv"), hide_index=True)
    figures = [
        ("results_comparison.png", "Rules only, ML only and hybrid in each domain"),
        ("shap_fraud.png", "Fraud: which features drive the risk score"),
        ("shap_student.png", "Student risk: which features drive the risk score"),
        ("fairness_student.png", "Student fairness by school, before and after mitigation"),
        ("eda_fraud.png", "Fraud rate by hour and by merchant category"),
        ("eda_student.png", "At-risk rate by past failures, and absences by status"),
    ]
    found = [(f, cap) for f, cap in figures if os.path.exists(f)]
    for f, cap in found:
        st.subheader(cap)
        st.image(f)
    if not found and not os.path.exists("results_all.csv"):
        st.info("Add the result charts (PNG files) and `results_all.csv` to the repository to show them here.")
