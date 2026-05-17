# ⚡ KarachiWatts — AI Load Shedding Assistant

> *"Bijli kab aayegi?"* — KarachiWatts answers with AI.

An intelligent load shedding prediction system for Karachi, combining Machine Learning with a Groq-powered chatbot (llama-3.3-70b) and a full Streamlit dashboard. Built for **Lab 14 · BS Artificial Intelligence · DUET**.

---

## 🗂️ Project Structure

```
KarachiWatts/
│
├── main.py               ← Run this for terminal mode (full pipeline)
├── dashboard.py          ← Run this for the web dashboard
├── generate_dataset.py   ← Generates the dataset (auto-runs if missing)
├── model.py              ← ML pipeline: load, preprocess, train, visualize
├── chatbot.py            ← Groq AI chatbot with ML prediction injection
├── build_notebook.py     ← Generates the Jupyter notebook
│
├── data/
│   └── karachi_loadshedding.csv   ← Generated dataset (16,440 rows)
│
├── plots/                ← Auto-generated visualizations (7 charts)
│
├── notebooks/
│   └── KarachiWatts_Lab14.ipynb   ← Jupyter notebook for submission
│
├── .env                  ← Your Groq API key goes here (create this)
├── .env.example          ← Template for .env
├── requirements.txt      ← Python dependencies
└── .gitignore
```

---

## 🚀 Setup (5 minutes)

### 1. Open in VS Code
Open the `KarachiWatts` folder in VS Code.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get your free Groq API key
1. Go to → https://console.groq.com
2. Sign up (free)
3. Click **API Keys** → **Create API Key**
4. Copy the key

### 4. Set up your `.env` file
```bash
cp .env.example .env
```
Open `.env` and paste your key:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

### 5a. Run the Dashboard (recommended)
```bash
streamlit run dashboard.py
```
Opens automatically in your browser at `http://localhost:8501`

### 5b. Run Terminal Mode
```bash
python main.py
```
Runs the full pipeline then launches the chatbot in terminal.

---

## 🖥️ Dashboard Pages

### 🏠 Dashboard
- 4 live metric cards (predicted hours, area average, worst month, severity)
- Smart tip based on prediction
- Monthly trend chart for selected area
- All-areas comparison bar chart for selected month
- Full sortable comparison table

### 📈 Analytics
- Outage heatmap — all areas x all months
- Season distribution chart
- Outage hours distribution histogram
- Temperature vs outage scatter plot
- Dataset summary stats

### 💬 Chatbot
- Groq-powered (llama-3.3-70b-versatile)
- ML predictions injected automatically
- Quick question buttons
- Full conversation history
- Clear chat button

---

## 💬 What You Can Ask the Chatbot

| Question | Example |
|----------|---------|
| Area prediction | "How many hours in Gulshan in July?" |
| Urdu style | "Bijli kab aayegi in Orangi Town?" |
| Compare areas | "Compare Korangi vs DHA in August" |
| Best/worst area | "Which area has least load shedding?" |
| Seasonal trends | "Worst month for load shedding?" |
| Tips | "How to survive load shedding?" |

---

## 📊 Model Performance

| Model | MAE | R2 |
|-------|-----|----|
| Linear Regression | 1.36h | 0.860 |
| Gradient Boosting | 0.43h | 0.984 |
| Random Forest | 0.26h | 0.991 |

Random Forest selected as best model — R2 = 0.991.

---

## 🌍 Areas Covered

| Area | Severity |
|------|----------|
| Defence (DHA) | Low |
| Clifton | Low |
| Gulshan-e-Iqbal | Medium |
| North Nazimabad | Medium |
| Saddar | Medium |
| Gulberg | Medium |
| Nazimabad | Medium |
| Federal B Area | Medium |
| SITE Area | Medium |
| Korangi | High |
| Landhi | High |
| Orangi Town | High |
| Malir | High |
| Liaquatabad | High |
| Baldia Town | High |

---

## 🎓 Lab 14 Requirements — All Covered

- Problem definition and dataset selection
- Data loading and exploration (Pandas)
- Data cleaning and preprocessing
- Feature engineering
- Visualizations (Matplotlib + Seaborn) — 7 charts
- ML model training and evaluation (scikit-learn)
- Model comparison and feature importance
- AI chatbot (Groq llama-3.3-70b)
- Full Streamlit web dashboard
- Conclusions and analysis

---

## 🚀 Pushing to GitHub

```bash
git init
git add .
git commit -m "Initial commit - KarachiWatts AI Load Shedding Assistant"
git remote add origin https://github.com/YOUR_USERNAME/KarachiWatts.git
git branch -M main
git push -u origin main
```

Note: Your .env file is protected by .gitignore — your API key will NOT be uploaded.

---

Built with love for Karachi. BS AI, MARYAM SOHAIL AHMED
.