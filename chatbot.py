"""
chatbot.py
KarachiWatts AI chatbot powered by Groq (llama-3.3-70b-versatile).
The ML model predictions are injected into the system prompt so
Groq answers as a real Karachi load shedding expert.
"""

import os
from groq import Groq
from dotenv import load_dotenv
from model import (
    predict_outage, AREA_SEVERITY, SEASONS_LOOKUP,
    SEASON_MULT, TEMP_DEFAULTS,
)

load_dotenv()

AREAS = sorted(AREA_SEVERITY.keys())

SYSTEM_PROMPT = """You are KarachiWatts — an expert AI assistant for Karachi's electricity 
and load shedding situation. You have deep knowledge of:
- K-Electric and WAPDA load shedding patterns in all Karachi areas
- Area-wise severity (DHA/Clifton have least, Orangi/Landhi/Korangi/Baldia have most)
- Seasonal patterns (summer = worst, especially May-August)
- Practical tips for managing load shedding in Pakistani households
- The frustration Karachi residents feel about bijli problems

When given a prediction from the ML model, present it clearly with:
1. The predicted outage hours
2. A risk level (LOW/MODERATE/HIGH/VERY HIGH)
3. 2-3 practical tips specific to that area and season
4. A relatable, friendly tone mixing English with occasional Urdu words (bijli, kharaabi, load shedding)

Keep responses concise, helpful, and empathetic. You understand the pain of load shedding.
Never make up numbers — always use the ML prediction provided to you in the user message.

Areas you cover: Gulshan-e-Iqbal, North Nazimabad, Korangi, Landhi, Defence (DHA), Clifton,
SITE Area, Orangi Town, Malir, Saddar, Gulberg, Liaquatabad, Federal B Area, Nazimabad, Baldia Town.

If asked about something unrelated to Karachi electricity/load shedding, politely redirect.
"""


def risk_label(hours: float) -> str:
    if hours <= 3:   return "🟢 LOW"
    elif hours <= 6: return "🟡 MODERATE"
    elif hours <= 10: return "🔴 HIGH"
    else:            return "🔴 VERY HIGH"


def find_area(text: str) -> str | None:
    text_l = text.lower()
    for area in AREAS:
        if area.lower() in text_l:
            return area
    partials = {
        "gulshan":      "Gulshan-e-Iqbal",
        "nazimabad":    "North Nazimabad",
        "north naz":    "North Nazimabad",
        "dha":          "Defence (DHA)",
        "defence":      "Defence (DHA)",
        "defense":      "Defence (DHA)",
        "korangi":      "Korangi",
        "landhi":       "Landhi",
        "clifton":      "Clifton",
        "site":         "SITE Area",
        "orangi":       "Orangi Town",
        "malir":        "Malir",
        "saddar":       "Saddar",
        "gulberg":      "Gulberg",
        "liaquatabad":  "Liaquatabad",
        "federal":      "Federal B Area",
        "baldia":       "Baldia Town",
    }
    for key, val in partials.items():
        if key in text_l:
            return val
    return None


def find_month(text: str) -> int | None:
    months = {
        "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
        "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
        "january":1,"february":2,"march":3,"april":4,
        "june":6,"july":7,"august":8,"september":9,
        "october":10,"november":11,"december":12,
    }
    text_l = text.lower()
    for name, num in months.items():
        if name in text_l:
            return num
    return None


def find_day(text: str) -> str:
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    text_l = text.lower()
    for d in days:
        if d.lower() in text_l:
            return d
    return "Monday"


def build_context(user_input: str, model) -> str:
    """
    Checks if user is asking about a specific area.
    If yes, runs ML prediction and injects it into the prompt.
    """
    area  = find_area(user_input)
    month = find_month(user_input) or 7
    day   = find_day(user_input)

    if area:
        hours  = predict_outage(model, area, month, day)
        season = SEASONS_LOOKUP[month]
        risk   = risk_label(hours)
        return (
            f"[ML PREDICTION DATA — use this in your response]\n"
            f"Area: {area}\n"
            f"Month: {month} ({season})\n"
            f"Day: {day}\n"
            f"Predicted outage: {hours} hours/day\n"
            f"Risk level: {risk}\n"
            f"Severity zone: {AREA_SEVERITY.get(area,'medium').upper()}\n"
            f"---\n"
            f"User question: {user_input}"
        )

    # compare two areas
    if "compare" in user_input.lower() or " vs " in user_input.lower():
        areas_found = [a for a in AREAS if a.lower() in user_input.lower()]
        if len(areas_found) < 2:
            for key in ["dha","gulshan","korangi","orangi","clifton","malir",
                        "landhi","nazimabad","saddar","gulberg","site","baldia",
                        "liaquatabad","federal"]:
                if key in user_input.lower():
                    mapped = find_area(key)
                    if mapped and mapped not in areas_found:
                        areas_found.append(mapped)
        if len(areas_found) >= 2:
            month = find_month(user_input) or 7
            results = []
            for a in areas_found[:2]:
                h = predict_outage(model, a, month)
                results.append(f"{a}: {h}h/day ({risk_label(h)})")
            return (
                f"[ML COMPARISON DATA]\n"
                f"Month: {month}\n" +
                "\n".join(results) +
                f"\n---\nUser question: {user_input}"
            )

    # area ranking question
    if any(w in user_input.lower() for w in
           ["best area","worst area","least load","most load","ranking","which area"]):
        month = find_month(user_input) or 7
        ranking = []
        for a in AREAS:
            h = predict_outage(model, a, month)
            ranking.append((a, h))
        ranking.sort(key=lambda x: x[1])
        top3    = ranking[:3]
        bottom3 = ranking[-3:]
        data    = (
            f"[ML AREA RANKING — Month {month}]\n"
            f"LEAST load shedding: " +
            ", ".join(f"{a}({h}h)" for a,h in top3) +
            f"\nMOST load shedding: " +
            ", ".join(f"{a}({h}h)" for a,h in reversed(bottom3)) +
            f"\n---\nUser question: {user_input}"
        )
        return data

    return user_input


class KarachiWattsBot:

    def __init__(self, ml_model):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY not set.\n"
                "1. Copy .env.example → .env\n"
                "2. Paste your Groq API key\n"
                "Get free key: https://console.groq.com"
            )
        self.client    = Groq(api_key=api_key)
        self.ml_model  = ml_model
        self.history   = []
        self.model_id  = "llama-3.3-70b-versatile"

    def chat(self, user_input: str) -> str:
        # inject ML context if relevant
        enriched = build_context(user_input, self.ml_model)

        # add to history
        self.history.append({"role": "user", "content": enriched})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=600,
            temperature=0.7,
        )

        reply = response.choices[0].message.content.strip()

        # store only original user text (not enriched) for cleaner history
        self.history[-1] = {"role": "user",      "content": user_input}
        self.history.append({"role": "assistant", "content": reply})

        # keep history to last 10 turns to avoid token overflow
        if len(self.history) > 20:
            self.history = self.history[-20:]

        return reply

    def reset(self):
        self.history = []
        print("🔄 Conversation history cleared.")

    def run(self):
        print("\n" + "="*60)
        print("  ⚡  KarachiWatts — AI Load Shedding Chatbot")
        print("  Powered by Groq (llama-3.3-70b-versatile)")
        print("  Type 'exit' to quit | 'reset' to clear history")
        print("="*60)
        print("\n🤖 KarachiWatts: Assalam o Alaikum! I'm KarachiWatts,")
        print("   your AI assistant for Karachi load shedding.")
        print("   Ask me about any area — bijli schedule, hours, tips!\n")

        while True:
            try:
                user = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n🤖 Khuda Hafiz! Stay cool 🌬️")
                break

            if not user:
                continue
            if user.lower() in ["exit", "quit", "bye", "khuda hafiz"]:
                print("🤖 KarachiWatts: Khuda Hafiz! May your bijli never go! 🌬️")
                break
            if user.lower() == "reset":
                self.reset()
                continue

            print("\n🤖 KarachiWatts: ", end="", flush=True)
            try:
                reply = self.chat(user)
                print(reply)
            except Exception as e:
                print(f"[Error: {e}]")
            print()
