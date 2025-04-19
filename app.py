import streamlit as st
import pandas as pd
import requests
import os
import io
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = "stock_data_fullform.csv"

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
MODEL_ID = "deepseek/deepseek-r1:free"

st.set_page_config(page_title="💼 Stock CRM Assistant", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(CSV_PATH)
        num_cols = [
            'Current Market Price (Rs.)', 'Price to Earnings Ratio',
            'Market Capitalization (Rs. Cr.)', 'Dividend Yield (%)',
            'Net Profit', 'Sales Growth (%)', 'Profit Growth (%)'
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        return None

def extract_relevant_data(query, df):
    keywords = query.lower().split()
    mask = df['Company Name'].str.lower().str.contains('|'.join(keywords), na=False)
    return df[mask] if mask.any() else df

def generate_crm_response(query, context_md):
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    messages = [
        {"role": "system",
         "content": ("You are a professional Stock CRM Analyst Assistant. "
                     "Always base your answer strictly on the given table. "
                     "Be structured, concise, and analytical.")},
        {"role": "user", "content": f"DATA SNAPSHOT:\n{context_md}\n\nUser Query: {query}"}
    ]
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1024
    }
    try:
        resp = requests.post(api_url, headers=headers, json=payload)
        resp_json = resp.json()
        return resp_json["choices"][0]["message"]["content"] \
               if "choices" in resp_json else f"Unexpected response: {resp_json}"
    except Exception as e:
        return f"API error: {e}"

# --- Load data & init state
df = load_data()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role":..., "content":...}

# --- Sidebar: clear & download
with st.sidebar:
    st.header("⚙️ Chat Options")
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Cleared!")
    if st.session_state.chat_history:
        buf = io.StringIO()
        for msg in st.session_state.chat_history:
            who = "User" if msg["role"]=="user" else "CRM Bot"
            buf.write(f"{who}: {msg['content']}\n\n")
        st.download_button("📥 Download Chat", buf.getvalue(),
                           file_name="stock_crm_chat.txt", mime="text/plain")

# --- Main chat area
st.subheader("🗨️ Conversation")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Pinned input at bottom
if df is not None:
    user_input = st.chat_input("Ask about any stock (e.g. “Analyze TCS Q4 performance”)")
    if user_input:
        # append & display user
        st.session_state.chat_history.append({"role":"user","content":user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # call API, append & display assistant
        context_md = extract_relevant_data(user_input, df).to_markdown(index=False)
        reply = generate_crm_response(user_input, context_md)

        st.session_state.chat_history.append({"role":"assistant","content":reply})
        with st.chat_message("assistant"):
            st.write(reply)

# --- Extras
with st.expander("📊 Full Stock Data Table"):
    st.dataframe(df, use_container_width=True, height=300)

with st.expander("ℹ️ Setup Guide"):
    st.markdown("""
1. Get your OpenRouter API key from [OpenRouter](https://openrouter.ai)  
2. Create a `.env` file:""")
