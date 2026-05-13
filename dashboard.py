"""
dashboard.py
KarachiWatts — Streamlit Dashboard
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
from groq import Groq

# ── page config (MUST be first streamlit call) ────────────────
st.set_page_config(
    page_title="KarachiWatts ⚡",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ── load model (cached so it only trains once) ────────────────
@st.cache_resource(show_spinner="⚡ Training ML model...")
def load_model_and_data():
    from model import load_data, preprocess, train
    df        = load_data()
    df, encs  = preprocess(df)
    model, name, r2 = train(df)
    return model, df, r2

# ── helpers (imported from model.py) ─────────────────────────
from model import (
    predict_outage, AREA_SEVERITY,
    SEASONS_LOOKUP, TEMP_DEFAULTS,
)

AREAS = sorted(AREA_SEVERITY.keys())

MONTHS = {
    1:"January", 2:"February", 3:"March",    4:"April",
    5:"May",     6:"June",     7:"July",      8:"August",
    9:"September",10:"October",11:"November",12:"December",
}

DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

def risk_label(h):
    if h <= 3:    return ("🟢 LOW",       "#27ae60")
    elif h <= 6:  return ("🟡 MODERATE",  "#f39c12")
    elif h <= 10: return ("🔴 HIGH",      "#e74c3c")
    else:         return ("🔴 VERY HIGH", "#c0392b")

def tip(h):
    if h <= 3:
        return "✅ Minimal disruption expected. Normal routine is fine."
    elif h <= 6:
        return "⚠️ Charge devices early. Avoid heavy appliances 6–10pm."
    elif h <= 10:
        return "🔋 Keep UPS charged. Plan around evening peak hours."
    else:
        return "🚨 Severe outages expected. Charge everything by 8am, store water."

# ── custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0f1117; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #16213e);
        border: 1px solid #2d3561;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-number {
        font-size: 52px;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 13px;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 8px 24px;
        border-radius: 99px;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-top: 8px;
    }

    /* Section headers */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #ccd6f6;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #2d3561;
    }

    /* Tip box */
    .tip-box {
        background: #1a1f2e;
        border-left: 4px solid #64ffda;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 14px;
        color: #ccd6f6;
        margin-top: 12px;
    }

    /* Chat messages */
    .chat-user {
        background: #1e3a5f;
        border-radius: 12px 12px 4px 12px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #ccd6f6;
        font-size: 14px;
    }
    .chat-bot {
        background: #1a2744;
        border: 1px solid #2d3561;
        border-radius: 4px 12px 12px 12px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #ccd6f6;
        font-size: 14px;
    }

    /* Sidebar */
    .css-1d391kg { background-color: #0d1117; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚡ KarachiWatts")
    st.markdown("*AI Load Shedding Assistant*")
    st.markdown("---")

    st.markdown("### 🗺️ Select Your Area")
    selected_area = st.selectbox("Area", AREAS, index=AREAS.index("Gulshan-e-Iqbal"))

    st.markdown("### 📅 Select Month")
    selected_month_name = st.selectbox("Month", list(MONTHS.values()), index=6)
    selected_month = [k for k,v in MONTHS.items() if v==selected_month_name][0]

    st.markdown("### 📆 Select Day")
    selected_day = st.selectbox("Day", DAYS)

    st.markdown("### 🌡️ Temperature (optional)")
    use_custom_temp = st.checkbox("Set custom temperature")
    if use_custom_temp:
        temperature = st.slider("°C", 15, 48,
                                TEMP_DEFAULTS[SEASONS_LOOKUP[selected_month]])
    else:
        temperature = None

    st.markdown("---")
    st.markdown("### 📊 Navigation")
    page = st.radio("Go to", ["🏠 Dashboard", "📈 Analytics", "💬 Chatbot"])

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#4a5568;'>CREATED BY :MARYAMS SOHAIL AHMED </div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════

model, df, model_r2 = load_model_and_data()

# current prediction
pred_hours = predict_outage(model, selected_area, selected_month,
                             selected_day, temperature)
rlabel, rcolor = risk_label(pred_hours)
current_tip    = tip(pred_hours)
severity       = AREA_SEVERITY.get(selected_area, "medium").upper()
season         = SEASONS_LOOKUP[selected_month].capitalize()


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════

if page == "🏠 Dashboard":

    # header
    st.markdown(f"""
    <h1 style='font-size:36px;font-weight:800;color:#ccd6f6;margin-bottom:4px;'>
        ⚡ KarachiWatts
    </h1>
    <p style='color:#8892b0;font-size:15px;margin-bottom:32px;'>
        AI-Powered Load Shedding Predictor &nbsp;·&nbsp; {selected_area} &nbsp;·&nbsp;
        {selected_month_name} &nbsp;·&nbsp; {selected_day}
    </p>
    """, unsafe_allow_html=True)

    # ── top metric cards ──────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-number' style='color:#64ffda;'>{pred_hours}h</div>
            <div class='metric-label'>Predicted Outage Today</div>
            <div class='risk-badge' style='background:{rcolor}22;color:{rcolor};'>
                {rlabel}
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        # avg for this area
        area_avg = round(df[df["area"]==selected_area]["outage_hours"].mean(), 1)
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-number' style='color:#f39c12;'>{area_avg}h</div>
            <div class='metric-label'>Area Annual Average</div>
            <div style='font-size:12px;color:#8892b0;margin-top:8px;'>per day</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        # worst month for this area
        area_monthly = df[df["area"]==selected_area].groupby("month")["outage_hours"].mean()
        worst_month  = MONTHS[area_monthly.idxmax()]
        worst_hours  = round(area_monthly.max(), 1)
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-number' style='color:#e74c3c;font-size:32px;
                margin-top:8px;'>{worst_month}</div>
            <div class='metric-label'>Worst Month</div>
            <div style='font-size:12px;color:#8892b0;margin-top:8px;'>{worst_hours}h avg</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-number' style='color:#bd93f9;font-size:28px;
                margin-top:8px;'>{severity}</div>
            <div class='metric-label'>Zone Severity</div>
            <div style='font-size:12px;color:#8892b0;margin-top:8px;'>
                Model R² = {model_r2:.3f}
            </div>
        </div>""", unsafe_allow_html=True)

    # tip box
    st.markdown(f"<div class='tip-box'>💡 <b>Tip:</b> {current_tip}</div>",
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── charts row ────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("<div class='section-header'>📈 Monthly Trend — " +
                    selected_area + "</div>", unsafe_allow_html=True)

        area_df  = df[df["area"] == selected_area]
        monthly  = area_df.groupby("month")["outage_hours"].mean()

        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#1a1f2e")
        ax.set_facecolor("#1a1f2e")
        ax.plot(monthly.index, monthly.values, color="#64ffda",
                linewidth=2.5, marker="o", markersize=5)
        ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="#64ffda")
        ax.axvline(x=selected_month, color=rcolor, linestyle="--",
                   alpha=0.8, linewidth=1.5, label=f"Selected: {selected_month_name}")
        ax.set_xticks(range(1,13))
        ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"],
                           color="#8892b0", fontsize=10)
        ax.tick_params(colors="#8892b0")
        ax.spines[:].set_color("#2d3561")
        ax.set_ylabel("Avg Hours/Day", color="#8892b0", fontsize=10)
        ax.legend(facecolor="#1a1f2e", edgecolor="#2d3561",
                  labelcolor="#ccd6f6", fontsize=9)
        ax.grid(True, alpha=0.15, color="#2d3561")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown("<div class='section-header'>🏙️ All Areas — " +
                    selected_month_name + "</div>", unsafe_allow_html=True)

        month_df   = df[df["month"] == selected_month]
        area_hours = month_df.groupby("area")["outage_hours"].mean().sort_values()
        colors_bar = ["#27ae60" if a == selected_area else
                      ("#e74c3c" if v > 9 else "#f39c12")
                      for a, v in area_hours.items()]

        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#1a1f2e")
        ax.set_facecolor("#1a1f2e")
        bars = ax.barh(area_hours.index, area_hours.values,
                       color=colors_bar, height=0.6)
        ax.tick_params(colors="#8892b0", labelsize=9)
        ax.spines[:].set_color("#2d3561")
        ax.set_xlabel("Avg Hours/Day", color="#8892b0", fontsize=10)
        ax.grid(True, alpha=0.15, axis="x", color="#2d3561")
        for bar in bars:
            ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                    f"{bar.get_width():.1f}h",
                    va="center", color="#8892b0", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── area comparison table ─────────────────────────────────
    st.markdown("<br><div class='section-header'>📋 All Areas Comparison — " +
                selected_month_name + "</div>", unsafe_allow_html=True)

    month_df   = df[df["month"] == selected_month]
    area_table = month_df.groupby("area")["outage_hours"].mean().round(1).reset_index()
    area_table.columns = ["Area", "Avg Outage Hours"]
    area_table["Risk"]     = area_table["Avg Outage Hours"].apply(lambda h: risk_label(h)[0])
    area_table["Severity"] = area_table["Area"].map(AREA_SEVERITY).str.upper()
    area_table = area_table.sort_values("Avg Outage Hours")
    area_table["Rank"] = range(1, len(area_table)+1)
    area_table = area_table[["Rank","Area","Avg Outage Hours","Risk","Severity"]]

    st.dataframe(
        area_table.style.apply(
            lambda row: ["background-color: #1e3a2f" if row["Area"]==selected_area
                         else "" for _ in row], axis=1
        ),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════

elif page == "📈 Analytics":

    st.markdown("""
    <h1 style='font-size:32px;font-weight:800;color:#ccd6f6;margin-bottom:4px;'>
        📈 Analytics
    </h1>
    <p style='color:#8892b0;font-size:15px;margin-bottom:32px;'>
        Deep dive into Karachi load shedding patterns
    </p>
    """, unsafe_allow_html=True)

    # ── heatmap ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>🗺️ Outage Heatmap — Area × Month</div>",
                unsafe_allow_html=True)

    pivot = df.pivot_table(values="outage_hours",
                           index="area", columns="month", aggfunc="mean")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#1a1f2e")
    ax.set_facecolor("#1a1f2e")
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".1f",
                linewidths=0.3, ax=ax,
                cbar_kws={"label":"Avg Hours/Day"},
                annot_kws={"size":8})
    ax.tick_params(colors="#ccd6f6", labelsize=9)
    ax.set_xlabel("Month", color="#8892b0")
    ax.set_ylabel("")
    plt.title("Load Shedding Heatmap — Karachi Areas", color="#ccd6f6",
              fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── two charts side by side ───────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>📊 Season Distribution</div>",
                    unsafe_allow_html=True)
        season_avg = df.groupby("season")["outage_hours"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#1a1f2e")
        ax.set_facecolor("#1a1f2e")
        colors_s = ["#e74c3c","#f39c12","#3498db","#27ae60"]
        bars = ax.bar(season_avg.index, season_avg.values, color=colors_s)
        ax.tick_params(colors="#8892b0")
        ax.spines[:].set_color("#2d3561")
        ax.set_ylabel("Avg Hours/Day", color="#8892b0", fontsize=10)
        ax.grid(True, alpha=0.15, axis="y")
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
                    f"{b.get_height():.1f}h", ha="center",
                    color="#ccd6f6", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("<div class='section-header'>📉 Distribution of Outage Hours</div>",
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#1a1f2e")
        ax.set_facecolor("#1a1f2e")
        ax.hist(df["outage_hours"], bins=40, color="#64ffda", alpha=0.7, edgecolor="#1a1f2e")
        ax.tick_params(colors="#8892b0")
        ax.spines[:].set_color("#2d3561")
        ax.set_xlabel("Outage Hours", color="#8892b0", fontsize=10)
        ax.set_ylabel("Frequency", color="#8892b0", fontsize=10)
        ax.grid(True, alpha=0.15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── temperature vs outage ─────────────────────────────────
    st.markdown("<div class='section-header'>🌡️ Temperature vs Outage Hours</div>",
                unsafe_allow_html=True)

    sample = df.sample(3000, random_state=42)
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="#1a1f2e")
    ax.set_facecolor("#1a1f2e")
    sc = ax.scatter(sample["temperature_c"], sample["outage_hours"],
                    c=sample["demand_index"], cmap="plasma",
                    alpha=0.4, s=10)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Demand Index", color="#8892b0")
    cbar.ax.yaxis.set_tick_params(color="#8892b0")
    ax.tick_params(colors="#8892b0")
    ax.spines[:].set_color("#2d3561")
    ax.set_xlabel("Temperature (°C)", color="#8892b0", fontsize=11)
    ax.set_ylabel("Outage Hours", color="#8892b0", fontsize=11)
    ax.grid(True, alpha=0.1)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── stats summary ─────────────────────────────────────────
    st.markdown("<br><div class='section-header'>📋 Dataset Summary</div>",
                unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Areas Covered", df["area"].nunique())
    col3.metric("Date Range", "2022–2024")
    col4.metric("Model R²", f"{model_r2:.4f}")


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — CHATBOT
# ══════════════════════════════════════════════════════════════

elif page == "💬 Chatbot":

    st.markdown("""
    <h1 style='font-size:32px;font-weight:800;color:#ccd6f6;margin-bottom:4px;'>
        💬 KarachiWatts Chatbot
    </h1>
    <p style='color:#8892b0;font-size:15px;margin-bottom:24px;'>
        Powered by Groq · llama-3.3-70b-versatile
    </p>
    """, unsafe_allow_html=True)

    # check API key
    api_key = os.getenv("GROQ_API_KEY","")
    if not api_key or api_key == "your_groq_api_key_here":
        st.error("⚠️ Groq API key not found. Add it to your `.env` file.")
        st.code("GROQ_API_KEY=gsk_xxxxxxxxxxxx")
        st.stop()

    # init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "👋 Assalam o Alaikum! I'm KarachiWatts — your AI load shedding assistant.\n\n"
                "Ask me anything:\n"
                "- *How many hours in Gulshan in July?*\n"
                "- *Bijli kab aayegi in Orangi Town?*\n"
                "- *Compare Korangi vs DHA*\n"
                "- *Tips to survive load shedding*"
            )
        })

    # system prompt
    SYSTEM = """You are KarachiWatts, an expert AI assistant for Karachi load shedding.
You have ML model predictions injected into your context when relevant.
Be helpful, friendly, concise. Mix English with occasional Urdu (bijli, kharaabi).
Always present ML predictions clearly with risk level and practical tips.
Keep responses under 150 words unless comparison or detailed tips are needed."""

    from chatbot import find_area, find_month, find_day, build_context

    # display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-user'>👤 <b>You:</b> {msg['content']}</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'>🤖 <b>KarachiWatts:</b><br>{msg['content']}</div>",
                        unsafe_allow_html=True)

    # quick question buttons
    st.markdown("<br>**Quick questions:**", unsafe_allow_html=True)
    qcols = st.columns(3)
    quick_qs = [
        "How bad is Gulshan in July?",
        "Compare Orangi vs DHA",
        "Which area has least load shedding?",
        "Tips to manage bijli",
        "Worst month for load shedding?",
        "How many hours in Korangi on Friday?",
    ]
    quick_input = None
    for i, q in enumerate(quick_qs):
        if qcols[i % 3].button(q, key=f"quick_{i}", use_container_width=True):
            quick_input = q

    # chat input
    st.markdown("<br>", unsafe_allow_html=True)
    user_input = st.chat_input("Ask about any area... (e.g. 'Bijli kab aayegi in Saddar?')")

    # handle input
    final_input = quick_input or user_input

    if final_input:
        # add user message
        st.session_state.messages.append({"role":"user","content":final_input})

        # build enriched context with ML prediction
        enriched = build_context(final_input, model)

        # call groq
        try:
            client   = Groq(api_key=api_key)
            history  = [{"role":"system","content":SYSTEM}]
            for m in st.session_state.messages[:-1]:
                history.append({"role":m["role"],"content":m["content"]})
            history.append({"role":"user","content":enriched})

            with st.spinner("⚡ KarachiWatts is thinking..."):
                resp  = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=history,
                    max_tokens=400,
                    temperature=0.7,
                )
            reply = resp.choices[0].message.content.strip()

        except Exception as e:
            reply = f"⚠️ Error: {e}"

        st.session_state.messages.append({"role":"assistant","content":reply})
        st.rerun()

    # clear button
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()