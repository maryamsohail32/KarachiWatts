"""
generate_dataset.py
Generates a realistic synthetic Karachi load shedding dataset.
Run once before anything else: python generate_dataset.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

AREAS = {
    "Gulshan-e-Iqbal":  {"base_hours": 6,  "severity": "medium"},
    "North Nazimabad":  {"base_hours": 8,  "severity": "medium"},
    "Korangi":          {"base_hours": 10, "severity": "high"},
    "Landhi":           {"base_hours": 12, "severity": "high"},
    "Defence (DHA)":    {"base_hours": 2,  "severity": "low"},
    "Clifton":          {"base_hours": 2,  "severity": "low"},
    "SITE Area":        {"base_hours": 8,  "severity": "medium"},
    "Orangi Town":      {"base_hours": 12, "severity": "high"},
    "Malir":            {"base_hours": 10, "severity": "high"},
    "Saddar":           {"base_hours": 6,  "severity": "medium"},
    "Gulberg":          {"base_hours": 8,  "severity": "medium"},
    "Liaquatabad":      {"base_hours": 10, "severity": "high"},
    "Federal B Area":   {"base_hours": 6,  "severity": "medium"},
    "Nazimabad":        {"base_hours": 8,  "severity": "medium"},
    "Baldia Town":      {"base_hours": 12, "severity": "high"},
}

SEASONS_MAP = {
    1:"winter", 2:"winter", 3:"spring", 4:"spring",
    5:"summer", 6:"summer", 7:"summer", 8:"summer",
    9:"autumn", 10:"autumn", 11:"winter", 12:"winter",
}

SEASON_MULT = {"summer":1.6, "spring":1.1, "autumn":0.9, "winter":0.7}
DAY_MULT    = {0:1.0, 1:1.0, 2:1.0, 3:1.0, 4:1.1, 5:0.8, 6:0.7}


def generate(start="2022-01-01", end="2024-12-31"):
    records = []
    cur = datetime.strptime(start, "%Y-%m-%d")
    fin = datetime.strptime(end,   "%Y-%m-%d")

    while cur <= fin:
        month   = cur.month
        season  = SEASONS_MAP[month]
        weekday = cur.weekday()
        s_mult  = SEASON_MULT[season]
        d_mult  = DAY_MULT[weekday]

        for area, info in AREAS.items():
            noise       = np.random.normal(0, 0.8)
            hours       = round(max(0, min(24, info["base_hours"] * s_mult * d_mult + noise)), 1)
            temp_base   = {"winter":20,"spring":28,"summer":38,"autumn":30}[season]
            temperature = round(temp_base + np.random.normal(0, 2), 1)
            incidents   = max(1, int(hours / 2.5) + random.randint(-1, 1))
            overloaded  = 1 if hours > 9 else 0
            demand      = round(min(10, (hours/12)*10*s_mult*(temperature/38)), 1)
            is_holiday  = 1 if (month==8 and cur.day==14) or (month==3 and cur.day==23) else 0

            records.append({
                "date":            cur.strftime("%Y-%m-%d"),
                "day_of_week":     cur.strftime("%A"),
                "month":           month,
                "season":          season,
                "area":            area,
                "severity":        info["severity"],
                "outage_hours":    hours,
                "incidents":       incidents,
                "temperature_c":   temperature,
                "demand_index":    demand,
                "feeder_overload": overloaded,
                "is_holiday":      is_holiday,
            })
        cur += timedelta(days=1)

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("⚡ Generating Karachi load shedding dataset...")
    df = generate()
    out = os.path.join("data", "karachi_loadshedding.csv")
    os.makedirs("data", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"✅ Saved → {out}")
    print(f"   Shape : {df.shape}  |  Areas : {df['area'].nunique()}  |  Rows : {len(df):,}")
