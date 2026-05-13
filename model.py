"""
model.py
Loads data, builds features, trains ML models, generates visualizations.
Returns the best trained model for use by the chatbot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_absolute_error, r2_score
from sklearn.preprocessing   import LabelEncoder

PLOTS_DIR = "plots"
DATA_PATH = os.path.join("data", "karachi_loadshedding.csv")

FEATURES = [
    "area_enc", "season_enc", "severity_enc", "day_enc",
    "month", "temperature_c", "demand_index",
    "feeder_overload", "is_holiday", "is_weekend",
    "is_peak_month", "week_num",
]
TARGET = "outage_hours"


# ── load ──────────────────────────────────────────────────────

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Dataset not found. Run:  python generate_dataset.py"
        )
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Dataset loaded  →  {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ── preprocess ────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.drop_duplicates(inplace=True)

    le_area     = LabelEncoder()
    le_season   = LabelEncoder()
    le_severity = LabelEncoder()
    le_day      = LabelEncoder()

    df["area_enc"]     = le_area.fit_transform(df["area"])
    df["season_enc"]   = le_season.fit_transform(df["season"])
    df["severity_enc"] = le_severity.fit_transform(df["severity"])
    df["day_enc"]      = le_day.fit_transform(df["day_of_week"])

    df["week_num"]      = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]    = df["day_of_week"].isin(["Saturday","Sunday"]).astype(int)
    df["is_peak_month"] = df["month"].isin([5,6,7,8]).astype(int)

    encoders = {
        "area": le_area, "season": le_season,
        "severity": le_severity, "day": le_day,
    }
    print(f"✅ Preprocessing done  →  {df.shape[1]} features")
    return df, encoders


# ── visualizations ────────────────────────────────────────────

def make_plots(df: pd.DataFrame):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    saved = []

    # 1 — avg outage by area
    fig, ax = plt.subplots(figsize=(13, 5))
    area_avg = df.groupby("area")["outage_hours"].mean().sort_values(ascending=False)
    bars = ax.bar(area_avg.index, area_avg.values,
                  color=sns.color_palette("RdYlGn_r", len(area_avg)))
    ax.set_title("Average Daily Outage Hours by Area", fontsize=14, fontweight="bold")
    ax.set_xlabel("Area")
    ax.set_ylabel("Avg Hours / Day")
    plt.xticks(rotation=45, ha="right")
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
                f"{b.get_height():.1f}h", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "1_outage_by_area.png")
    plt.savefig(p, dpi=150); plt.close(); saved.append(p)

    # 2 — by season
    fig, ax = plt.subplots(figsize=(7, 4))
    sa = df.groupby("season")["outage_hours"].mean().sort_values(ascending=False)
    sns.barplot(x=sa.index, y=sa.values,
                palette=["#e74c3c","#f39c12","#27ae60","#3498db"], ax=ax)
    ax.set_title("Average Outage Hours by Season", fontsize=13, fontweight="bold")
    ax.set_xlabel("Season"); ax.set_ylabel("Avg Hours/Day")
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "2_by_season.png")
    plt.savefig(p, dpi=150); plt.close(); saved.append(p)

    # 3 — monthly trend
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly = df.groupby("month")["outage_hours"].mean()
    ax.plot(monthly.index, monthly.values, marker="o",
            color="#e74c3c", linewidth=2, markersize=6)
    ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="#e74c3c")
    ax.set_title("Monthly Load Shedding Trend", fontsize=13, fontweight="bold")
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_ylabel("Avg Outage Hours"); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "3_monthly_trend.png")
    plt.savefig(p, dpi=150); plt.close(); saved.append(p)

    # 4 — histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["outage_hours"], bins=30, kde=True, color="#3498db", ax=ax)
    ax.set_title("Distribution of Daily Outage Hours", fontsize=13, fontweight="bold")
    ax.set_xlabel("Outage Hours"); ax.set_ylabel("Frequency")
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "4_distribution.png")
    plt.savefig(p, dpi=150); plt.close(); saved.append(p)

    # 5 — heatmap area × month
    pivot = df.pivot_table(values="outage_hours",
                           index="area", columns="month", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".1f",
                linewidths=0.4, ax=ax, cbar_kws={"label":"Avg Hours/Day"})
    ax.set_title("Outage Heatmap — Area × Month", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "5_heatmap.png")
    plt.savefig(p, dpi=150); plt.close(); saved.append(p)

    # 6 — temp vs outage
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(df["temperature_c"], df["outage_hours"],
                    c=df["demand_index"], cmap="RdYlGn_r", alpha=0.3, s=8)
    plt.colorbar(sc, ax=ax, label="Demand Index")
    ax.set_title("Temperature vs Outage Hours", fontsize=13, fontweight="bold")
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Outage Hours")
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "6_temp_vs_outage.png")
    plt.savefig(p, dpi=150); plt.close(); saved.append(p)

    print(f"✅ {len(saved)} plots saved → {PLOTS_DIR}/")
    return saved


# ── train ─────────────────────────────────────────────────────

def train(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "Linear Regression" : LinearRegression(),
        "Random Forest"     : RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "Gradient Boosting" : GradientBoostingRegressor(n_estimators=150, random_state=42),
    }

    print(f"\n{'Model':<25} {'MAE':>8}  {'R²':>8}")
    print("─" * 46)

    best_model, best_r2, best_name = None, -999, ""

    for name, mdl in candidates.items():
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)
        print(f" {name:<24} {mae:>7.2f}h  {r2:>7.3f}")
        if r2 > best_r2:
            best_r2, best_model, best_name = r2, mdl, name

    print(f"\n✅ Best model : {best_name}  (R² = {best_r2:.4f})")

    # feature importance plot
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(9, 4))
        fi.plot(kind="bar", color="#2ecc71", ax=ax)
        ax.set_title(f"Feature Importance — {best_name}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Importance")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "7_feature_importance.png"), dpi=150)
        plt.close()

    return best_model, best_name, best_r2


# ── public helper: predict one row ────────────────────────────

AREA_SEVERITY = {
    "Gulshan-e-Iqbal":"medium","North Nazimabad":"medium","Korangi":"high",
    "Landhi":"high","Defence (DHA)":"low","Clifton":"low","SITE Area":"medium",
    "Orangi Town":"high","Malir":"high","Saddar":"medium","Gulberg":"medium",
    "Liaquatabad":"high","Federal B Area":"medium","Nazimabad":"medium","Baldia Town":"high",
}

SEASONS_LOOKUP = {
    1:"winter",2:"winter",3:"spring",4:"spring",
    5:"summer",6:"summer",7:"summer",8:"summer",
    9:"autumn",10:"autumn",11:"winter",12:"winter",
}

AREA_ENC_MAP     = {a:i for i,a in enumerate(sorted(AREA_SEVERITY.keys()))}
SEASON_ENC_MAP   = {"autumn":0,"spring":1,"summer":2,"winter":3}
SEVERITY_ENC_MAP = {"high":0,"low":1,"medium":2}
DAY_ENC_MAP      = {"Friday":0,"Monday":1,"Saturday":2,"Sunday":3,
                    "Thursday":4,"Tuesday":5,"Wednesday":6}
TEMP_DEFAULTS    = {"winter":20,"spring":28,"summer":38,"autumn":30}
SEASON_MULT      = {"summer":1.6,"spring":1.1,"autumn":0.9,"winter":0.7}


def predict_outage(model, area: str, month: int,
                   day_name: str = "Monday",
                   temperature: float = None,
                   is_holiday: int = 0) -> float:
    season   = SEASONS_LOOKUP[month]
    severity = AREA_SEVERITY.get(area, "medium")
    temp     = temperature or TEMP_DEFAULTS[season]
    s_mult   = SEASON_MULT[season]

    is_weekend    = 1 if day_name in ["Saturday","Sunday"] else 0
    is_peak_month = 1 if month in [5,6,7,8] else 0
    demand        = round(min(10, (temp/38)*10*s_mult), 1)
    feeder        = 1 if severity=="high" and is_peak_month else 0
    week_num      = (month-1)*4+2

    X = [[
        AREA_ENC_MAP.get(area, 0),
        SEASON_ENC_MAP[season],
        SEVERITY_ENC_MAP[severity],
        DAY_ENC_MAP.get(day_name, 1),
        month, temp, demand,
        feeder, is_holiday, is_weekend,
        is_peak_month, week_num,
    ]]
    return round(max(0, model.predict(X)[0]), 1)


# ── run standalone ────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    df, encoders = preprocess(df)
    make_plots(df)
    best_model, best_name, best_r2 = train(df)
    print(f"\nDone. Model ready — R² = {best_r2:.4f}")

    import joblib
    joblib.dump(best_model, "random_forest_model.pkl")
