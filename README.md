# ⚡ KarachiWatts — AI Load Shedding Assistant

> *"Bijli kab aayegi?"* — KarachiWatts answers with AI.

An intelligent load shedding prediction system for Karachi, combining Machine Learning with a Groq-powered chatbot (llama-3.3-70b). 

---

## 🗂️ Project Structure

```
KarachiWatts/
│
├── main.py               ← Run this to start everything
├── generate_dataset.py   ← Generates the dataset (auto-runs)
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

### 1. Clone / open in VS Code
Open this folder in VS Code.

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
# Copy the template
cp .env.example .env
```
Open `.env` and replace with your key:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

### 5. Run the project
```bash
python main.py
```

That's it. Everything runs automatically:
- Dataset is generated if missing
- Data is explored and preprocessed
- 7 visualizations are saved to `plots/`
- 3 ML models are trained and compared
- Chatbot launches automatically

---

## 💬 Chatbot — What You Can Ask

| Question | Example |
|----------|---------|
| Area prediction | *"How many hours in Gulshan in July?"* |
| Urdu style | *"Bijli kab aayegi in Orangi Town?"* |
| Compare areas | *"Compare Korangi vs DHA in August"* |
| Best/worst area | *"Which area has least load shedding?"* |
| Seasonal trends | *"Worst month for load shedding?"* |
| Tips | *"How to survive load shedding?"* |

Type `reset` to clear conversation history.  
Type `exit` or `bye` to quit.

---

## 📓 Jupyter Notebook

To generate the notebook:
```bash
python build_notebook.py
```
Then open `notebooks/KarachiWatts_Lab14.ipynb` in Jupyter or VS Code.

---

## 📊 Model Performance

| Model | MAE | R² |
|-------|-----|----|
| Linear Regression | 1.36h | 0.860 |
| Gradient Boosting | 0.43h | 0.984 |
| **Random Forest** | **0.26h** | **0.991** |

Random Forest selected as best model with **R² = 0.991**.

---

## 🌍 Areas Covered

| Area | Severity |
|------|----------|
| Defence (DHA) | 🟢 Low |
| Clifton | 🟢 Low |
| Gulshan-e-Iqbal | 🟡 Medium |
| North Nazimabad | 🟡 Medium |
| Saddar | 🟡 Medium |
| Gulberg | 🟡 Medium |
| Nazimabad | 🟡 Medium |
| Federal B Area | 🟡 Medium |
| SITE Area | 🟡 Medium |
| Korangi | 🔴 High |
| Landhi | 🔴 High |
| Orangi Town | 🔴 High |
| Malir | 🔴 High |
| Liaquatabad | 🔴 High |
| Baldia Town | 🔴 High |

---


*Built with ❤️ for Karachi · 
BY MARYAM SOHAIL AHMED*
