"""
main.py
KarachiWatts — AI Load Shedding Assistant
==========================================
Entry point. Runs the full pipeline:
  1. Load & explore data
  2. Preprocess & engineer features
  3. Generate visualizations
  4. Train ML models
  5. Launch Groq-powered chatbot

Usage:
  python main.py
"""

import os
import sys

DIVIDER = "=" * 60

def banner():
    print(f"""
{DIVIDER}
  ⚡  KarachiWatts — AI Load Shedding Assistant
  Lab 14 · BS Artificial Intelligence · DUET Karachi
  Powered by Groq + scikit-learn
{DIVIDER}
""")


def check_env():
    if not os.path.exists(".env"):
        print("⚠️  .env file not found!")
        print("   1. Copy .env.example to .env")
        print("   2. Add your Groq API key")
        print("   Get free key: https://console.groq.com\n")
        sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GROQ_API_KEY","")
    if not key or key == "your_groq_api_key_here":
        print("⚠️  GROQ_API_KEY is not set in your .env file!")
        print("   Open .env and paste your key.\n")
        sys.exit(1)
    print("✅ Groq API key found\n")


def check_dataset():
    path = os.path.join("data","karachi_loadshedding.csv")
    if not os.path.exists(path):
        print("📦 Dataset not found — generating now...")
        import generate_dataset  # noqa: runs on import
        print()
    else:
        print("✅ Dataset already exists\n")


def main():
    banner()
    check_env()
    check_dataset()

    # ── pipeline ──────────────────────────────────────────────
    from model import load_data, preprocess, make_plots, train

    print(f"{DIVIDER}")
    print("  STEP 1 — Loading & Exploring Data")
    print(DIVIDER)
    df = load_data()
    print(f"   Rows      : {len(df):,}")
    print(f"   Areas     : {df['area'].nunique()}")
    print(f"   Date range: {df['date'].min()} → {df['date'].max()}")

    print(f"\n{DIVIDER}")
    print("  STEP 2 — Preprocessing & Feature Engineering")
    print(DIVIDER)
    df, encoders = preprocess(df)

    print(f"\n{DIVIDER}")
    print("  STEP 3 — Generating Visualizations")
    print(DIVIDER)
    make_plots(df)

    print(f"\n{DIVIDER}")
    print("  STEP 4 — Training ML Models")
    print(DIVIDER)
    model, model_name, r2 = train(df)
    print(f"\n   Model accuracy (R²) : {r2:.4f}")
    print(f"   Mean Abs Error      : see table above")

    print(f"\n{DIVIDER}")
    print("  STEP 5 — Launching KarachiWatts Chatbot")
    print(DIVIDER)

    from chatbot import KarachiWattsBot
    bot = KarachiWattsBot(ml_model=model)
    bot.run()


if __name__ == "__main__":
    main()
