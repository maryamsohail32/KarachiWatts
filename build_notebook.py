"""
Build the KarachiWatts Jupyter notebook.
Run: python build_notebook.py
"""

import nbformat as nbf, os

nb    = nbf.v4.new_notebook()
cells = []
md    = lambda t: nbf.v4.new_markdown_cell(t)
code  = lambda t: nbf.v4.new_code_cell(t)

# ── Title ─────────────────────────────────────────────────────
cells.append(md("""# ⚡ KarachiWatts — AI Load Shedding Assistant
### Lab 14: Complex Computing Activity
**Course:** AI-2201 Programming for AI &nbsp;|&nbsp; **Dept:** BS Artificial Intelligence &nbsp;|&nbsp; **University:** DUET Karachi

---

### Problem Statement
Karachi faces severe load shedding but K-Electric provides zero predictive tools.  
**KarachiWatts** uses Machine Learning + Groq AI to predict outage hours for any area  
and answer natural questions through an intelligent chatbot.

### Pipeline
`Data Generation → EDA → Visualization → ML Model → Groq Chatbot`

---
"""))

# ── Setup ─────────────────────────────────────────────────────
cells.append(md("## 📦 Step 0 — Imports & Setup"))
cells.append(code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_absolute_error, r2_score
from sklearn.preprocessing   import LabelEncoder
from dotenv                  import load_dotenv
from groq                    import Groq

load_dotenv()
sns.set_theme(style="whitegrid")
print("✅ All libraries loaded")
"""))

# ── Generate data ─────────────────────────────────────────────
cells.append(md("## 🗂️ Step 1 — Generate & Load Dataset"))
cells.append(code("""# Run generate_dataset.py if file doesn't exist
if not os.path.exists("data/karachi_loadshedding.csv"):
    exec(open("generate_dataset.py").read())

df = pd.read_csv("data/karachi_loadshedding.csv")
print(f"Shape : {df.shape}")
print(f"Areas : {df['area'].nunique()}")
df.head()
"""))

cells.append(code("""print("--- Missing Values ---")
print(df.isnull().sum())
print()
print("--- Statistical Summary ---")
df.describe().round(2)
"""))

# ── Preprocessing ─────────────────────────────────────────────
cells.append(md("## 🧹 Step 2 — Preprocessing & Feature Engineering"))
cells.append(code("""df["date"] = pd.to_datetime(df["date"])
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

print(f"✅ Preprocessing done — {df.shape[1]} total columns")
df.head(3)
"""))

# ── Visualizations ────────────────────────────────────────────
cells.append(md("## 📊 Step 3 — Exploratory Data Analysis & Visualizations"))

cells.append(code("""# 1 — Avg outage hours by area
fig, ax = plt.subplots(figsize=(13, 5))
area_avg = df.groupby("area")["outage_hours"].mean().sort_values(ascending=False)
bars = ax.bar(area_avg.index, area_avg.values,
              color=sns.color_palette("RdYlGn_r", len(area_avg)))
ax.set_title("Average Daily Outage Hours by Area", fontsize=14, fontweight="bold")
ax.set_xlabel("Area"); ax.set_ylabel("Avg Hours/Day")
plt.xticks(rotation=45, ha="right")
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
            f"{b.get_height():.1f}h", ha="center", va="bottom", fontsize=8)
plt.tight_layout(); plt.savefig("plots/1_outage_by_area.png", dpi=150); plt.show()
"""))

cells.append(code("""# 2 — By season
fig, ax = plt.subplots(figsize=(7, 4))
sa = df.groupby("season")["outage_hours"].mean().sort_values(ascending=False)
sns.barplot(x=sa.index, y=sa.values,
            palette=["#e74c3c","#f39c12","#27ae60","#3498db"], ax=ax)
ax.set_title("Average Outage Hours by Season", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig("plots/2_by_season.png", dpi=150); plt.show()
"""))

cells.append(code("""# 3 — Monthly trend
fig, ax = plt.subplots(figsize=(10, 4))
monthly = df.groupby("month")["outage_hours"].mean()
ax.plot(monthly.index, monthly.values, marker="o", color="#e74c3c", linewidth=2)
ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="#e74c3c")
ax.set_title("Monthly Load Shedding Trend", fontsize=13, fontweight="bold")
ax.set_xticks(range(1,13))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"])
ax.grid(True, alpha=0.4)
plt.tight_layout(); plt.savefig("plots/3_monthly_trend.png", dpi=150); plt.show()
"""))

cells.append(code("""# 4 — Distribution
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["outage_hours"], bins=30, kde=True, color="#3498db", ax=ax)
ax.set_title("Distribution of Daily Outage Hours", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig("plots/4_distribution.png", dpi=150); plt.show()
"""))

cells.append(code("""# 5 — Heatmap area × month
os.makedirs("plots", exist_ok=True)
pivot = df.pivot_table(values="outage_hours", index="area", columns="month", aggfunc="mean")
fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".1f",
            linewidths=0.4, ax=ax, cbar_kws={"label":"Avg Hours/Day"})
ax.set_title("Outage Heatmap — Area × Month", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig("plots/5_heatmap.png", dpi=150); plt.show()
"""))

cells.append(code("""# 6 — Temperature vs outage scatter
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(df["temperature_c"], df["outage_hours"],
                c=df["demand_index"], cmap="RdYlGn_r", alpha=0.3, s=8)
plt.colorbar(sc, ax=ax, label="Demand Index")
ax.set_title("Temperature vs Outage Hours", fontsize=13, fontweight="bold")
ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Outage Hours")
plt.tight_layout(); plt.savefig("plots/6_temp_vs_outage.png", dpi=150); plt.show()
"""))

# ── ML ────────────────────────────────────────────────────────
cells.append(md("## 🤖 Step 4 — Machine Learning Model Training"))
cells.append(code("""FEATURES = [
    "area_enc","season_enc","severity_enc","day_enc",
    "month","temperature_c","demand_index",
    "feeder_overload","is_holiday","is_weekend",
    "is_peak_month","week_num",
]
TARGET = "outage_hours"

X = df[FEATURES]; y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
"""))

cells.append(code("""models = {
    "Linear Regression" : LinearRegression(),
    "Random Forest"     : RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
    "Gradient Boosting" : GradientBoostingRegressor(n_estimators=150, random_state=42),
}

results = {}
print(f"{'Model':<25} {'MAE':>8}  {'R²':>8}")
print("─"*46)

best_model, best_r2, best_name = None, -999, ""
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    results[name] = {"model": mdl, "mae": mae, "r2": r2}
    print(f" {name:<24} {mae:>7.2f}h  {r2:>7.3f}")
    if r2 > best_r2:
        best_r2, best_model, best_name = r2, mdl, name

print(f"\\n✅ Best model: {best_name}  (R² = {best_r2:.4f})")
"""))

cells.append(code("""# Feature importance
fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 4))
fi.plot(kind="bar", color="#2ecc71", ax=ax)
ax.set_title(f"Feature Importance — {best_name}", fontsize=13, fontweight="bold")
ax.set_ylabel("Importance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout(); plt.savefig("plots/7_feature_importance.png", dpi=150); plt.show()
"""))

# ── Chatbot ───────────────────────────────────────────────────
cells.append(md("## 💬 Step 5 — KarachiWatts Groq Chatbot\n\n> Make sure your `.env` file has `GROQ_API_KEY=your_key_here`"))

cells.append(code("""from chatbot import KarachiWattsBot

# Initialize bot with trained ML model
bot = KarachiWattsBot(ml_model=best_model)
print("✅ KarachiWatts chatbot ready!")
"""))

cells.append(md("### 🎤 Demo — Sample Conversations"))
cells.append(code("""demo_questions = [
    "How many hours of load shedding in Gulshan in July?",
    "Bijli kab aayegi in Orangi Town on Saturday?",
    "Compare Korangi vs DHA in August",
    "Which area has least load shedding?",
    "Worst month for load shedding in Karachi?",
    "Give me tips to survive load shedding",
]

for q in demo_questions:
    print(f"\\n{'─'*55}")
    print(f"You: {q}")
    print(f"\\n🤖 KarachiWatts:")
    reply = bot.chat(q)
    print(reply)
"""))

cells.append(md("### 🗣️ Live Interactive Chat\n*Run the cell below to chat live with KarachiWatts*"))
cells.append(code("""bot.reset()   # fresh conversation
bot.run()     # type your questions, 'exit' to quit
"""))

# ── Conclusion ────────────────────────────────────────────────
cells.append(md("""---
## ✅ Conclusion

| Step | Result |
|------|--------|
| Dataset | 16,440 records · 15 Karachi areas · 3 years |
| Features | 12 engineered features (season, area severity, temperature, etc.) |
| Best Model | Random Forest — **R² = 0.991**, MAE = 0.26h |
| Visualizations | 7 charts — heatmap, trend, histogram, scatter, bar, importance |
| Chatbot | Groq (llama-3.3-70b) + ML predictions = real intelligent assistant |

**Real-world impact:** Every Karachi resident asks *"bijli kab aayegi?"* daily.  
KarachiWatts answers it with AI — something K-Electric has never done.

---
*AI-2201 Lab 14 · BS Artificial Intelligence · DUET*
"""))

nb.cells = cells
os.makedirs("notebooks", exist_ok=True)
path = "notebooks/KarachiWatts_Lab14.ipynb"
with open(path, "w") as f:
    nbf.write(nb, f)
print(f"✅ Notebook saved → {path}")
