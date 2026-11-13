"""RiskSense demo: one framework, two domains.

Loads the two trained models exported from the Colab notebook
(fraud.joblib and student.joblib) and scores a single case.
"""
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
    "G1": "first-period grade", "failures": "past failures", "absences": "absences",
    "studytime": "study time", "higher": "plans higher education", "parent_edu": "parental education",
}


# ---------------------------------------------------------------- generic scoring (same for both domains)
def score(bundle, rules, tiers, row):
    df = pd.DataFrame([row])
    X = np.asarray(bundle["pre"].transform(df[bundle["num"] + bundle["cat"]]), dtype=float)
    model = bundle["model"]

    ml = float(model.predict_proba(X)[0, 1])
    fired = [(name, level) for name, cond, level in rules if bool(cond(df).iloc[0])]
    rule = max([level for _, level in fired], default=0.0)
    final = max(rule, ml)
    tier_index = max(i for i, (cut, _) in enumerate(tiers) if final >= cut)

    # per-feature contributions (SHAP values)
    names = [c.split("__")[1] for c in bundle["pre"].get_feature_names_out()]
    if bundle["kind"] == "xgb":
        contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
    else:
        contrib = model.coef_[0] * (X[0] - bundle["bg_mean"])
    order = np.argsort(-contrib)[:3]
    reasons = [LABELS.get(names[i], names[i]) for i in order if contrib[i] > 0]

    return {"ml": ml, "rule": rule, "final": final, "fired": fired,
            "tier": tiers[tier_index][1], "tier_index": tier_index, "reasons": reasons}


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
    st.markdown("- " + "\n- ".join(result["reasons"]) if result["reasons"] else "- nothing raising risk")


# ---------------------------------------------------------------- page
st.title("RiskSense")
st.caption("One early-warning framework, two domains. The scoring code is shared; "
           "only the rules, features and action tiers differ.")

fraud_tab, student_tab = st.tabs(["Credit card fraud", "Student academic risk"])

with fraud_tab:
    fb = load("fraud.joblib")
    presets = {
        "Normal purchase": dict(amt=45.0, usual=70.0, hour=14, cat="grocery_pos", dev=False, gap=150, n24=3, newcat=False),
        "Borderline": dict(amt=250.0, usual=70.0, hour=23, cat="shopping_net", dev=False, gap=90, n24=4, newcat=False),
        "Clear high risk": dict(amt=900.0, usual=70.0, hour=2, cat="shopping_net", dev=True, gap=12, n24=6, newcat=True),
    }
    pname = st.selectbox("Load an example", list(presets), key="fraud_preset")
    p = presets[pname]
    cats = fb["categories"]["category"]

    left, right = st.columns(2)
    amt = left.number_input("Transaction amount", min_value=1.0, value=p["amt"], key=f"amt_{pname}")
    usual = right.number_input("Customer's usual amount", min_value=1.0, value=p["usual"], key=f"usual_{pname}")
    hour = left.slider("Hour of day", 0, 23, p["hour"], key=f"hour_{pname}")
    cat = right.selectbox("Merchant category", cats, index=cats.index(p["cat"]), key=f"cat_{pname}")
    gap = left.slider("Seconds from login to purchase", 1, 600, p["gap"], key=f"gap_{pname}")
    n24 = right.slider("Transactions in last 24 hours", 0, 30, p["n24"], key=f"n24_{pname}")
    dev = left.checkbox("New device", p["dev"], key=f"dev_{pname}")
    newcat = right.checkbox("First purchase in this category", p["newcat"], key=f"newcat_{pname}")

    row = dict(fb["defaults"])
    row.update(amt=amt, amt_ratio=amt / usual, hour=hour, is_night=int(hour >= 22 or hour <= 3),
               category=cat, new_device=int(dev), login_gap_s=gap, txn_24h=n24, new_category=int(newcat))
    st.divider()
    show(score(fb, FRAUD_RULES, FRAUD_TIERS, row))
    st.caption("Model: XGBoost. Device and login fields were simulated in training. "
               "Inputs not shown here use typical values from the training data.")

with student_tab:
    sb = load("student.joblib")
    presets = {
        "Doing well": dict(g1=14, fail=0, absn=2, study=3, higher=True),
        "Borderline": dict(g1=9, fail=0, absn=4, study=2, higher=True),
        "Clear high risk": dict(g1=7, fail=2, absn=10, study=1, higher=False),
    }
    pname = st.selectbox("Load an example", list(presets), key="student_preset")
    p = presets[pname]

    left, right = st.columns(2)
    g1 = left.slider("First-period grade (out of 20)", 0, 20, p["g1"], key=f"g1_{pname}")
    fail = right.slider("Past failures", 0, 3, p["fail"], key=f"fail_{pname}")
    absn = left.slider("Absences", 0, 30, p["absn"], key=f"abs_{pname}")
    study = right.slider("Study time band (1 = lowest)", 1, 4, p["study"], key=f"study_{pname}")
    higher = left.checkbox("Plans higher education", p["higher"], key=f"higher_{pname}")

    row = dict(sb["defaults"])
    row.update(G1=g1, failures=fail, absences=absn, studytime=study, higher=int(higher))
    st.divider()
    show(score(sb, STUDENT_RULES, STUDENT_TIERS, row))
    st.caption("Model: Logistic Regression. A flag only triggers an offer of support. "
               "Inputs not shown here use typical values from the training data.")
