"""
Demo: Surfactant formulation selection for oil-contaminated soil remediation.
Run:     streamlit run app.py
Data:    soil_remediation_AIDME.csv (next to app.py or upload in sidebar)

P0-fixes relative to the original version:
  P0.1  Explicit feature selection rules (ID-like, constants, >95% NaN are dropped)
  P0.2  Uncertainty intervals (quantile XGBoost q10/q90, fallback to residual std)
  P0.3  Joint surfactant feasibility constraint (Mahalanobis on HLB, M, CMC within surf type)
  P0.4  OOD detector (LocalOutlierFactor) — warning on out-of-distribution inputs
"""
import io
import itertools
import os
import re

import altair as alt
import numpy as np
import optuna
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import LocalOutlierFactor

optuna.logging.set_verbosity(optuna.logging.WARNING)

st.set_page_config(page_title="Surfactant Selection", page_icon="🧪", layout="wide")

TARGET = "oil_removal"
DEFAULT_PATHS = ["soil_remediation_AIDME.csv", "data/soil_remediation_AIDME.csv"]

CONTROLLABLE = [
    "surfactant_mgL", "hlb", "mol_weight_gmol", "cmc_mgL",
    "agitation_rpm", "wash_time_h", "temperature_c", "liquid_soil_ratio",
]
SURF_PROPS = ["hlb", "mol_weight_gmol", "cmc_mgL"]

LABELS = {
    "surfactant_mgL": ("Surfactant Concentration", "mg/L"),
    "hlb": ("HLB", ""),
    "mol_weight_gmol": ("Surfactant Molar Mass", "g/mol"),
    "cmc_mgL": ("CMC", "mg/L"),
    "agitation_rpm": ("Agitation Speed", "rpm"),
    "wash_time_h": ("Washing Time", "h"),
    "temperature_c": ("Temperature", "°C"),
    "liquid_soil_ratio": ("Liquid/Soil Ratio", "L/kg"),
}
CAT_LABELS = {
    "surf_nonionic": "Nonionic",
    "surf_anionic": "Anionic",
    "surf_cationic": "Cationic",
    "surf_biosurfactant": "Biosurfactant",
    "surf_mixed": "Mixed",
    "method_batch": "Batch Washing",
    "method_column": "Column Washing",
}


def label(f):
    return LABELS.get(f, (f.replace("_", " ").title(), ""))[0]


def unit(f):
    return LABELS.get(f, ("", ""))[1]


def cat_label(c):
    return CAT_LABELS.get(c, c)


# ---------------------------------------------------------------------
# P0.1: Feature Selection Rules
# ---------------------------------------------------------------------
ID_LIKE_REGEX = re.compile(
    r"^(id|idx|index|row|row_?num|sample|sample_?id|patient_?id|subject_?id|unnamed.*)$",
    re.IGNORECASE,
)
MANUAL_EXCLUDE: set[str] = set()  # Add columns here for manual exclusion


def is_id_like(col: str) -> bool:
    return bool(ID_LIKE_REGEX.match(col)) or col in MANUAL_EXCLUDE


def detect_features(df: pd.DataFrame, target: str = TARGET) -> dict:
    """Returns dict with keep (features for model) and dropped (reason for dropping)."""
    candidate = df.drop(columns=[target], errors="ignore").select_dtypes(include=[np.number])
    keep, dropped = [], {}
    for col in candidate.columns:
        if is_id_like(col):
            dropped[col] = "ID-like column"
        elif candidate[col].nunique(dropna=True) <= 1:
            dropped[col] = "Constant"
        elif candidate[col].isna().mean() > 0.95:
            dropped[col] = ">95% NaN"
        else:
            keep.append(col)
    return {"keep": keep, "dropped": dropped, "all_numeric": list(candidate.columns)}


# ---------------------------------------------------------------------
# P0.3: Joint Feasibility of Surfactant Properties (Mahalanobis per surf-type)
# ---------------------------------------------------------------------
class SurfactantFeasibility:
    """Checks whether the triplet (HLB, M, CMC) is realistic for a specific surfactant type."""

    def __init__(self, X_train: pd.DataFrame, min_samples: int = 10, q: float = 0.99):
        self.stats: dict[str, dict] = {}
        surf_cols = [c for c in X_train.columns if c.startswith("surf_")]
        for s in surf_cols:
            if s not in X_train.columns:
                continue
            sub = X_train.loc[X_train[s] == 1, SURF_PROPS].dropna()
            if len(sub) < min_samples:
                continue
            mu = sub.mean().values
            cov = np.cov(sub.values.T)
            if cov.ndim < 2:
                cov = np.atleast_2d(cov)
            cov = cov + np.eye(len(SURF_PROPS)) * 1e-6
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                continue
            dists = np.array([
                np.sqrt((v - mu) @ inv_cov @ (v - mu)) for v in sub.values
            ])
            self.stats[s] = {
                "mu": mu,
                "inv_cov": inv_cov,
                "n": len(sub),
                "threshold": float(np.quantile(dists, q)),
            }

    def distance(self, s: str, props: np.ndarray) -> float | None:
        st_ = self.stats.get(s)
        if st_ is None:
            return None
        d = props - st_["mu"]
        return float(np.sqrt(d @ st_["inv_cov"] @ d))

    def is_feasible(self, s: str, props: np.ndarray) -> bool:
        st_ = self.stats.get(s)
        if st_ is None:
            return True  # Insufficient data — do not block
        return self.distance(s, props) <= st_["threshold"]


# ---------------------------------------------------------------------
# P0.4: OOD Detector
# ---------------------------------------------------------------------
class OODDetector:
    def __init__(self, X_train: pd.DataFrame, contamination: float = 0.05):
        self.cols = list(X_train.columns)
        self.lof = LocalOutlierFactor(
            n_neighbors=min(20, max(5, len(X_train) // 10)),
            novelty=True,
            contamination=contamination,
        )
        self.lof.fit(X_train[self.cols].values)

    def score(self, row: pd.Series) -> float:
        return float(self.lof.score_samples(row[self.cols].values.reshape(1, -1))[0])

    def is_ood(self, row: pd.Series) -> bool:
        try:
            return self.lof.predict(row[self.cols].values.reshape(1, -1))[0] == -1
        except Exception:
            return False


# ---------------------------------------------------------------------
# Data & Model
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_df(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    return df[(df[TARGET] >= 0) & (df[TARGET] <= 100)].reset_index(drop=True)


FAST_PARAMS = dict(
    n_estimators=800, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
)


def _make_quantile_model(params: dict, alpha: float):
    """Attempts to create a quantile XGBoost model. Returns None if unsupported."""
    try:
        m = xgb.XGBRegressor(
            **params,
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        return m
    except (TypeError, xgb.core.XGBoostError):
        return None


@st.cache_resource(show_spinner=False)
def train_model(raw: bytes, n_tuning_trials: int):
    df = load_df(raw)

    # P0.1 — Feature Selection
    feat = detect_features(df, TARGET)
    cols = feat["keep"]
    X = df[cols]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if n_tuning_trials > 0:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        def objective(trial):
            p = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            }
            m = xgb.XGBRegressor(**p, tree_method="hist", random_state=42, n_jobs=-1)
            return cross_val_score(m, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1).mean()

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_tuning_trials)
        params = study.best_params
    else:
        params = FAST_PARAMS

    # Point Model
    model = xgb.XGBRegressor(**params, tree_method="hist", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # P0.2 — Quantile Models for Uncertainty Intervals
    q10 = _make_quantile_model(params, 0.1)
    q90 = _make_quantile_model(params, 0.9)
    if q10 is not None and q90 is not None:
        try:
            q10.fit(X_train, y_train)
            q90.fit(X_train, y_train)
        except Exception:
            q10 = q90 = None
    if q10 is None or q90 is None:
        q10 = q90 = None

    residual_std = float(np.std(y_train - model.predict(X_train)))
    has_quantile = q10 is not None

    # Feature Categorization
    ctrl = [f for f in CONTROLLABLE if f in cols]
    surf = [c for c in cols if c.startswith("surf_")]
    meth = [c for c in cols if c.startswith("method_")]
    rest = [c for c in cols if c not in ctrl + surf + meth]
    binary = {c for c in rest if set(X[c].dropna().unique()) <= {0, 1}}

    # P0.3 / P0.4
    feasibility = SurfactantFeasibility(X_train, min_samples=10)
    ood = OODDetector(X_train, contamination=0.05)

    return dict(
        model=model, q10=q10, q90=q90,
        has_quantile=has_quantile, residual_std=residual_std,
        X=X, X_train=X_train, X_test=X_test,
        r2_train=model.score(X_train, y_train),
        r2_test=model.score(X_test, y_test),
        params=params, cols=cols,
        ctrl=ctrl, surf=surf, meth=meth, rest=rest, binary=binary,
        medians=X_train.median(),
        importance=pd.Series(model.feature_importances_, index=X_train.columns)
        .sort_values(ascending=False),
        feasibility=feasibility,
        ood=ood,
        dropped_features=feat["dropped"],
    )


# ---------------------------------------------------------------------
# Input State Management
# ---------------------------------------------------------------------
def apply_row(B, row):
    for c in B["rest"]:
        v = row.get(c, B["medians"].get(c, 0.0))
        if pd.isna(v):
            v = B["medians"].get(c, 0.0)
        st.session_state[f"in_{c}"] = bool(round(v)) if c in B["binary"] else float(v)
    for f in B["ctrl"]:
        v = row.get(f, B["medians"].get(f, 0.0))
        if pd.isna(v):
            v = B["medians"].get(f, 0.0)
        st.session_state[f"cur_{f}"] = float(v)
    if B["surf"]:
        st.session_state["cur_surf"] = row[B["surf"]].astype(float).idxmax()
    if B["meth"]:
        st.session_state["cur_meth"] = row[B["meth"]].astype(float).idxmax()


def on_preset(B):
    choice = st.session_state["preset"]
    if choice.startswith("—"):
        row = B["medians"].copy()
        if B["surf"]:
            top = B["X_train"][B["surf"]].sum().idxmax()
            row[B["surf"]] = 0.0
            row[top] = 1.0
        if B["meth"]:
            top = B["X_train"][B["meth"]].sum().idxmax()
            row[B["meth"]] = 0.0
            row[top] = 1.0
    else:
        row = B["X_test"].iloc[int(choice.split("#")[1])]
    apply_row(B, row)
    st.session_state.pop("result", None)


# ---------------------------------------------------------------------
# Calculation Logic
# ---------------------------------------------------------------------
def build_row(B, soil, ctrl_vals, s, m):
    row = pd.Series(0.0, index=B["cols"])
    for c, v in soil.items():
        row[c] = v
    for f, v in ctrl_vals.items():
        row[f] = v
    for c in B["surf"]:
        row[c] = 1.0 if c == s else 0.0
    for c in B["meth"]:
        row[c] = 1.0 if c == m else 0.0
    return row


def predict_row(B, row):
    return float(B["model"].predict(pd.DataFrame([row], columns=B["cols"]))[0])


def predict_interval(B, row):
    """P0.2 — Uncertainty interval calculation for a single point."""
    Xdf = pd.DataFrame([row], columns=B["cols"])
    p = float(B["model"].predict(Xdf)[0])
    if B["has_quantile"]:
        lo = float(B["q10"].predict(Xdf)[0])
        hi = float(B["q90"].predict(Xdf)[0])
        lo, hi = sorted([lo, hi])
    else:
        lo = p - 1.28 * B["residual_std"]
        hi = p + 1.28 * B["residual_std"]
    lo = max(0.0, min(lo, p))
    hi = min(100.0, max(hi, p))
    return lo, p, hi


def bounds_for(B, f, s, lims, realistic):
    lo, hi = lims[f]
    if realistic and s is not None and f in SURF_PROPS:
        sub = B["X_train"].loc[B["X_train"][s] == 1, f]
        if len(sub) >= 5:
            lo2, hi2 = float(sub.min()), float(sub.max())
            lo, hi = max(lo, lo2), min(hi, hi2)
            if lo > hi:
                lo, hi = lo2, hi2
    return lo, hi


def optimize_one(B, soil, s, m, lims, realistic, n_trials):
    """Returns (best_vals, best_eff, relaxed_feasibility)."""
    bounds = {f: bounds_for(B, f, s, lims, realistic) for f in B["ctrl"]}

    def _suggest(trial):
        vals = {}
        for f, (lo, hi) in bounds.items():
            vals[f] = lo if hi - lo < 1e-12 else trial.suggest_float(f, lo, hi)
        return vals

    def objective(trial):
        vals = _suggest(trial)
        # P0.3: Filter out unphysical surfactant property combinations
        if realistic and s is not None and all(f in vals for f in SURF_PROPS):
            props = np.array([vals[f] for f in SURF_PROPS])
            if not B["feasibility"].is_feasible(s, props):
                raise optuna.TrialPruned()
        return predict_row(B, build_row(B, soil, vals, s, m))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    relaxed = False

    if not completed:
        # Fallback: Relax joint constraint
        relaxed = True

        def objective_relaxed(trial):
            vals = _suggest(trial)
            return predict_row(B, build_row(B, soil, vals, s, m))

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective_relaxed, n_trials=n_trials)

    best = {f: study.best_params.get(f, lo) for f, (lo, hi) in bounds.items()}
    return best, float(study.best_value), relaxed


# ---------------------------------------------------------------------
# Sidebar: Data & Model Settings
# ---------------------------------------------------------------------
st.sidebar.header("⚙️ Data & Model")
uploaded = st.sidebar.file_uploader("Upload CSV Dataset (Optional)", type="csv")
raw = None
if uploaded is not None:
    raw = uploaded.getvalue()
else:
    for p in DEFAULT_PATHS:
        if os.path.exists(p):
            with open(p, "rb") as fh:
                raw = fh.read()
            break
if raw is None:
    st.title("🧪 Surfactant Formulation Selection")
    st.info("Please place `soil_remediation_AIDME.csv` in the root directory or upload a file via the sidebar.")
    st.stop()

tune = st.sidebar.slider(
    "Optuna Hyperparameter Tuning Trials", 0, 50, 0,
    help="0 = fast default parameters (starts in seconds). 30 = full tuning (~1–2 min training time).",
)
with st.spinner("Training model…"):
    B = train_model(raw, tune)

if "inited" not in st.session_state:
    st.session_state["preset"] = "— Dataset Median —"
    on_preset(B)
    st.session_state["inited"] = True

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("🧪 Surfactant Formulation Selection for Soil Remediation")
st.caption(
    "Input soil characteristics and contamination parameters — the model (XGBoost + Optuna) "
    "will select the optimal surfactant type, concentration, and washing parameters to maximize predicted oil removal efficiency."
)
tab_calc, tab_model = st.tabs(["🎯 Recipe Optimization", "📊 Model Information"])

with tab_calc:
    col_in, col_out = st.columns([1, 1.35], gap="large")

    with col_in:
        st.subheader("1. Soil & Contamination Profile")
        st.selectbox(
            "Quick fill using preset example",
            ["— Dataset Median —"] + [f"Test Sample #{i}" for i in range(10)],
            key="preset", on_change=on_preset, args=(B,),
            help="Useful for demo: fills inputs with real test dataset samples.",
        )
        soil = {}
        cols2 = st.columns(2)
        for i, c in enumerate(B["rest"]):
            with cols2[i % 2]:
                if c in B["binary"]:
                    soil[c] = float(st.checkbox(label(c), key=f"in_{c}"))
                else:
                    lo, hi = float(B["X"][c].min()), float(B["X"][c].max())
                    if hi - lo < 1e-12:
                        soil[c] = lo
                        st.caption(f"{label(c)}: {lo:g} (constant)")
                    else:
                        if f"in_{c}" not in st.session_state:
                            st.session_state[f"in_{c}"] = (lo + hi) / 2
                        st.session_state[f"in_{c}"] = float(
                            np.clip(st.session_state[f"in_{c}"], lo, hi)
                        )
                        soil[c] = st.slider(
                            label(c) + (f", {unit(c)}" if unit(c) else ""),
                            lo, hi, step=(hi - lo) / 200, key=f"in_{c}",
                        )

        st.subheader("2. Optimization Targets")
        surf_opts = ["Auto (find best)"] + [cat_label(c) for c in B["surf"]]
        meth_opts = ["Auto (find best)"] + [cat_label(c) for c in B["meth"]]
        surf_sel = st.selectbox("Surfactant Type", surf_opts) if B["surf"] else None
        meth_sel = st.selectbox("Washing Method", meth_opts) if B["meth"] else None

        realistic = st.radio(
            "Surfactant Properties (HLB, M, CMC)",
            ["Realistic: bounded by chosen surfactant type",
             "Unconstrained: any values from entire dataset"],
            help="Realistic mode applies a Mahalanobis filter (P0.3) to reject "
                 "physically unviable combinations of (HLB, M, CMC) for the selected surfactant type.",
        ).startswith("Realistic")

        with st.expander("Parameter Search Ranges (Optimization Bounds)"):
            lims = {}
            for f in B["ctrl"]:
                lo, hi = float(B["X_train"][f].min()), float(B["X_train"][f].max())
                if hi - lo < 1e-12:
                    lims[f] = (lo, hi)
                    continue
                lims[f] = st.slider(
                    f"{label(f)}" + (f", {unit(f)}" if unit(f) else ""),
                    lo, hi, (lo, hi), step=(hi - lo) / 200, key=f"lim_{f}",
                )

        with st.expander("Current Baseline Recipe (Before / After Comparison)"):
            cur = {}
            c1, c2 = st.columns(2)
            for i, f in enumerate(B["ctrl"]):
                lo, hi = float(B["X"][f].min()), float(B["X"][f].max())
                with (c1 if i % 2 == 0 else c2):
                    if f"cur_{f}" not in st.session_state:
                        st.session_state[f"cur_{f}"] = (lo + hi) / 2
                    st.session_state[f"cur_{f}"] = float(
                        np.clip(st.session_state[f"cur_{f}"], lo, hi)
                    )
                    cur[f] = st.number_input(
                        label(f), min_value=lo, max_value=hi,
                        step=max((hi - lo) / 100, 1e-6), format="%.3f",
                        key=f"cur_{f}",
                    )
            cur_s = cur_m = None
            if B["surf"]:
                cur_s = st.selectbox(
                    "Surfactant Type (Current)", B["surf"], format_func=cat_label, key="cur_surf"
                )
            if B["meth"]:
                cur_m = st.selectbox(
                    "Method (Current)", B["meth"], format_func=cat_label, key="cur_meth"
                )

        n_trials = st.slider(
            "Search Depth (Optuna Trials per Combination)",
            30, 400, 120, step=10,
            help="Higher values give higher precision but take longer. 100–150 trials is optimal for demo purposes.",
        )
        go = st.button("🚀 Calculate Recipe", type="primary", use_container_width=True)

    # ------------------------- CALCULATION -------------------------
    if go:
        s_list = B["surf"] if (not B["surf"] or surf_sel == surf_opts[0]) else [
            c for c in B["surf"] if cat_label(c) == surf_sel
        ]
        m_list = B["meth"] if (not B["meth"] or meth_sel == meth_opts[0]) else [
            c for c in B["meth"] if cat_label(c) == meth_sel
        ]
        combos = list(itertools.product(s_list or [None], m_list or [None]))
        results = []
        bar = col_out.progress(0.0, text="Optimizing…")
        for i, (s, m) in enumerate(combos):
            vals, eff, relaxed = optimize_one(B, soil, s, m, lims, realistic, n_trials)
            results.append(dict(s=s, m=m, eff=eff, vals=vals, relaxed=relaxed))
            bar.progress((i + 1) / len(combos), text=f"Optimizing… {i + 1}/{len(combos)}")
        bar.empty()
        results.sort(key=lambda r: r["eff"], reverse=True)

        cur_row = build_row(B, soil, cur, cur_s, cur_m)
        base_lo, base_eff, base_hi = predict_interval(B, cur_row)

        # OOD Check (P0.4) — Evaluate recommended configuration
        best_for_ood = results[0]
        best_row = build_row(B, soil, best_for_ood["vals"],
                             best_for_ood["s"], best_for_ood["m"])
        ood_flag = B["ood"].is_ood(best_row)

        st.session_state["result"] = dict(
            results=results, base=base_eff,
            base_interval=(base_lo, base_hi),
            soil=soil, cur=cur, cur_s=cur_s, cur_m=cur_m,
            ood_flag=ood_flag,
        )

    # ------------------------- OUTPUT -------------------------
    with col_out:
        st.subheader("3. Optimization Results")
        R = st.session_state.get("result")
        if not R:
            st.info("Configure parameters on the left and click **«Calculate Recipe»**.")
        else:
            best = R["results"][0]
            gain = best["eff"] - R["base"]

            # P0.2 — Interval for recommended recipe
            best_row = build_row(B, R["soil"], best["vals"], best["s"], best["m"])
            best_lo, best_p, best_hi = predict_interval(B, best_row)

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Current Baseline",
                f"{R['base']:.1f}%",
                help=f"Confidence Interval: {R['base_interval'][0]:.1f}–{R['base_interval'][1]:.1f}%",
            )
            m2.metric(
                "Recommended Recipe",
                f"{best_p:.1f}%",
                f"{gain:+.1f} p.p.",
                help=f"Confidence Interval (P0.2): {best_lo:.1f}–{best_hi:.1f}%",
            )
            m3.metric(
                "Surfactant / Method",
                cat_label(best["s"]) if best["s"] else "—",
                cat_label(best["m"]) if best["m"] else None,
                delta_color="off",
            )

            st.caption(
                f"📏 Predicted Oil Removal: **{best_p:.1f}%** "
                f"(90% Confidence Interval: {best_lo:.1f}–{best_hi:.1f}%)"
            )

            # P0.4 — OOD Warning
            if R.get("ood_flag"):
                st.warning(
                    "Input parameters are Out-of-Distribution (P0.4). "
                    "The model is extrapolating — recommendation reliability is lower. "
                    "Laboratory validation is strictly recommended.",
                    icon="🔍",
                )

            # P0.3 — Relaxed Constraints Warning
            if best.get("relaxed"):
                st.warning(
                    "No physically viable combination of surfactant properties was found "
                    "within the specified ranges (P0.3). The joint feasibility constraint was "
                    "automatically relaxed — recommended properties may be unphysical.",
                    icon="🧪",
                )

            rows = []
            for f in B["ctrl"]:
                rows.append({
                    "Parameter": label(f), "Unit": unit(f),
                    "Current Baseline": R["cur"][f], "Recommended": best["vals"][f],
                    "Δ": best["vals"][f] - R["cur"][f],
                })
            tbl = pd.DataFrame(rows)
            st.dataframe(
                tbl, hide_index=True, use_container_width=True,
                column_config={
                    "Current Baseline": st.column_config.NumberColumn(format="%.3f"),
                    "Recommended": st.column_config.NumberColumn(format="%.3f"),
                    "Δ": st.column_config.NumberColumn(format="%+.3f"),
                },
            )
            st.download_button(
                "⬇️ Download Recipe (CSV)",
                tbl.to_csv(index=False).encode("utf-8-sig"),
                "recipe.csv", "text/csv",
            )

            if len(R["results"]) > 1:
                st.markdown("**Comparison of Surfactant × Method Combinations**")
                rank = pd.DataFrame({
                    "Combination": [
                        " + ".join(x for x in (
                            cat_label(r["s"]) if r["s"] else "",
                            cat_label(r["m"]) if r["m"] else "",
                        ) if x)
                        for r in R["results"]
                    ],
                    "Efficiency, %": [r["eff"] for r in R["results"]],
                })
                ch = alt.Chart(rank).mark_bar().encode(
                    x=alt.X("Efficiency, %:Q", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("Combination:N", sort="-x", title=None),
                    tooltip=["Combination", alt.Tooltip("Efficiency, %:Q", format=".1f")],
                ).properties(height=max(120, 32 * len(rank)))
                st.altair_chart(ch, use_container_width=True)

            st.markdown("**Sensitivity Analysis: Individual Parameter Variations**")
            f_sel = st.selectbox("Parameter", B["ctrl"], format_func=label, key="sens_f")
            lo, hi = lims[f_sel]
            if hi - lo > 1e-12:
                grid = np.linspace(lo, hi, 60)
                base_row = build_row(B, R["soil"], best["vals"], best["s"], best["m"])
                G = pd.DataFrame([base_row] * len(grid), columns=B["cols"])
                G[f_sel] = grid
                curve = pd.DataFrame({
                    label(f_sel): grid,
                    "Oil Removal Efficiency, %": B["model"].predict(G),
                })
                line = alt.Chart(curve).mark_line().encode(
                    x=alt.X(f"{label(f_sel)}:Q"),
                    y=alt.Y("Oil Removal Efficiency, %:Q", scale=alt.Scale(zero=False)),
                )
                pt = alt.Chart(pd.DataFrame({
                    label(f_sel): [best["vals"][f_sel]],
                    "Oil Removal Efficiency, %": [best["eff"]],
                })).mark_point(size=120, filled=True, color="red").encode(
                    x=f"{label(f_sel)}:Q", y="Oil Removal Efficiency, %:Q",
                )
                st.altair_chart(line + pt, use_container_width=True)
                st.caption("Red marker indicates the recommended optimal point.")

            st.warning(
                "This output is a model estimation based on literature and experimental datasets. "
                "Recommendations must be validated in a laboratory environment before deployment.",
                icon="⚠️",
            )

# ---------------------------------------------------------------------
# Model Information Tab
# ---------------------------------------------------------------------
with tab_model:
    a, b, c, d = st.columns(4)
    a.metric("Total Records", f"{len(B['X']):,}")
    b.metric("Features Used", len(B["cols"]))
    c.metric("R² Train Score", f"{B['r2_train']:.3f}")
    d.metric("R² Test Score", f"{B['r2_test']:.3f}")

    # P0.2 — Quantile status
    if B["has_quantile"]:
        st.success("✅ Uncertainty Intervals Active (Quantile XGBoost q10/q90).")
    else:
        st.info(
            f"ℹ️ Quantile XGBoost is unsupported in your installed version — intervals estimated "
            f"using residual std={B['residual_std']:.2f}. Upgrade to `xgboost>=2.0` "
            f"for native quantile models."
        )

    # P0.3 — Surfactant Constraints Status
    n_feas = len(B["feasibility"].stats)
    if n_feas > 0:
        st.success(
            f"✅ Joint Surfactant Constraints Active: "
            f"{n_feas} surfactant categories with Mahalanobis filter (P0.3)."
        )
        with st.expander("Mahalanobis Thresholds by Surfactant Type"):
            feas_df = pd.DataFrame([
                {"Surfactant Type": cat_label(s),
                 "Training Samples (N)": v["n"],
                 "Threshold (99%)": round(v["threshold"], 3)}
                for s, v in B["feasibility"].stats.items()
            ])
            st.dataframe(feas_df, hide_index=True, use_container_width=True)
    else:
        st.info(
            "ℹ️ Mahalanobis filter inactive: insufficient samples per category "
            "(<10). Joint feasibility constraint disabled."
        )

    # P0.1 — Dropped Features
    with st.expander("P0.1 — Feature Selection Summary"):
        st.markdown(
            "Only features passing strict rules were included: "
            "numeric, non-ID-like, non-constant, and <95% NaN values."
        )
        if B["dropped_features"]:
            st.markdown("**Dropped Columns:**")
            st.dataframe(
                pd.DataFrame([
                    {"Column": k, "Reason": v}
                    for k, v in B["dropped_features"].items()
                ]),
                hide_index=True, use_container_width=True,
            )
        else:
            st.caption("No columns were dropped.")
        st.caption(
            "To manually exclude additional columns, append them to "
            "`MANUAL_EXCLUDE` at the beginning of `app.py`."
        )

    st.markdown("**Feature Importance (Top 15)**")
    imp = B["importance"].head(15).reset_index()
    imp.columns = ["Feature", "Importance"]
    imp["Feature"] = imp["Feature"].map(
        lambda x: label(x) if x in LABELS else cat_label(x)
    )
    st.altair_chart(
        alt.Chart(imp).mark_bar().encode(
            x="Importance:Q",
            y=alt.Y("Feature:N", sort="-x", title=None),
        ).properties(height=380),
        use_container_width=True,
    )
    with st.expander("XGBoost Hyperparameters"):
        st.json(B["params"])