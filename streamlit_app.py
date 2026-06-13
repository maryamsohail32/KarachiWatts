import streamlit as st
from dashboard import load_model_and_data, show_dashboard
from chatbot import KarachiWattsBot
from build_notebook import build_notebook
from generate_dataset import generate

# Sidebar navigation
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ["Dashboard", "Analytics", "Chatbot"])

# Credits footer
st.sidebar.markdown("CREATED BY: MARYAM SOHAIL AHMED")

# Routing
if choice == "Dashboard":
    show_dashboard()

elif choice == "Analytics":
    st.write("📊 Analytics Notebook Builder")
    build_notebook()

elif choice == "Chatbot":
    st.write("⚡ KarachiWatts Chatbot")

    # Load ML model from dashboard.py
    model, df, r2 = load_model_and_data()
    bot = KarachiWattsBot(ml_model=model)

    # Chat interface
    user_input = st.text_input("Ask KarachiWatts about load shedding:")
    if user_input:
        reply = bot.chat(user_input)
        st.markdown(f"**🤖 KarachiWatts:** {reply}")
