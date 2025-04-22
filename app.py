import streamlit as st
import pandas as pd
import requests
import io
import re
from sqlalchemy import create_engine

# Fetch API key and DB URL from st.secrets
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
DATABASE_URL = st.secrets["DATABASE_URL"]
MODEL_ID = "deepseek/deepseek-r1:free"

st.set_page_config(page_title="💼 Stock CRM Assistant", layout="wide")

# Load stock data from PostgreSQL
@st.cache_data
def load_data():
    try:
        engine = create_engine(DATABASE_URL)
        query = "SELECT * FROM stock_data"
        df = pd.read_sql(query, engine)
        num_cols = ['current_market_price', 'price_to_earnings_ratio', 'market_cap',
                    'dividend_yield', 'net_profit', 'profit_growth', 'sales', 'sales_growth']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

def classify_query(query):
    query = query.lower()
    fundamental_patterns = [
        r'pe ratio of', r'price to earnings of', r'market cap of',
        r'dividend yield of', r'net profit of', r'profit growth of',
        r'sales growth of', r'financials of', r'\beps of\b',
        r'earnings per share of', r'return on equity of', r'roce of',
        r'show me.*financial data', r'what is.*current market price of',
        r'give me.*financial metrics for'
    ]
    if any(re.search(pattern, query) for pattern in fundamental_patterns):
        return "fundamental"
    if df is not None:
        company_names = df['company_name'].str.lower().unique()
        if any(name in query for name in company_names):
            data_terms = ['ratio', 'yield', 'profit', 'sales', 'growth', 
                         'price', 'cap', 'financial', 'metric', 'data']
            if any(term in query for term in data_terms):
                return "fundamental"
    return "theoretical"

def extract_relevant_data(query, df):
    clean_query = re.sub(r'[^a-zA-Z0-9\s]', '', query.lower())
    keywords = clean_query.split()
    matched_df = df[df['company_name'].str.lower().str.contains('|'.join(keywords), na=False)]
    if not matched_df.empty:
        st.session_state.last_companies = matched_df['company_name'].str.lower().unique().tolist()
        st.session_state.last_user_query = query
        return matched_df.head(3)
    fallback_companies = []
    for word in keywords:
        fallback_df = df[df['company_name'].str.lower().str.contains(word, na=False)]
        if not fallback_df.empty:
            fallback_companies.append(word)
    if fallback_companies:
        st.session_state.last_companies = fallback_companies
    elif st.session_state.last_companies:
        matched_df = df[df['company_name'].str.lower().str.contains('|'.join(st.session_state.last_companies), na=False)]
        return matched_df.head(3)
    return df.head(3)

def generate_crm_response(query, context_md=None, query_type="theoretical"):
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    if query_type == "theoretical":
        system_prompt = (
            "You are a professional Stock Market Analyst. Your task is to answer theoretical questions about "
            "stock market concepts, investing strategies, and financial analysis techniques.\n\n"
            "STRICT RULES FOR THEORETICAL ANSWERS:\n"
            "1. NEVER mention any specific companies from the database\n"
            "2. NEVER use actual numerical data from the database\n"
            "3. Only provide general explanations using hypothetical examples\n"
            "4. All examples must use generic terms like 'Company A' or 'Industry X'\n"
            "5. Never refer to any real financial metrics from the database\n\n"
            "Provide clear, conceptual explanations without referencing any real data. "
            "If you don't know the answer, say 'I don't know' rather than guessing."
        )
    else:
        system_prompt = (
            "You are a professional Stock CRM Analyst Assistant. "
            "You help users analyze fundamental stock data from a database. "
            "When presenting data:\n"
            "1. Be precise with numbers and metrics\n"
            "2. Always cite the source as 'our database'\n"
            "3. If data isn't available, explain what similar data exists\n"
            "4. Provide context for the numbers when possible"
        )

    messages = [{"role": "system", "content": system_prompt}]
    for chat in st.session_state.chat_history[-5:]:
        messages.append({"role": chat["role"], "content": chat["content"]})
    if st.session_state.last_user_query and len(query.strip().split()) <= 5:
        query = f"{st.session_state.last_user_query} FOLLOW-UP: {query}"
    if context_md and query_type == "fundamental":
        messages.append({
            "role": "user",
            "content": f"DATA SNAPSHOT:\n{context_md}\n\nUser Query: {query}"
        })
    else:
        messages.append({"role": "user", "content": query})
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1024
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"] \
            if "choices" in response_data else f"Unexpected response: {response_data}"
    except Exception as e:
        return f"API error: {e}"

df = load_data()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.last_companies = []
    st.session_state.last_user_query = ""

with st.sidebar:
    st.header("\u2699\ufe0f Chat Options")
    if st.button("Clear Chat History \U0001F9B9"):
        st.session_state.chat_history = []
        st.session_state.last_companies = []
        st.session_state.last_user_query = ""
        st.success("Cleared!")
    if st.session_state.chat_history:
        buf = io.StringIO()
        for msg in st.session_state.chat_history:
            who = "User" if msg["role"] == "user" else "CRM Bot"
            buf.write(f"{who}: {msg['content']}\n\n")
        st.download_button("Download Chat", buf.getvalue(),
                           file_name="stock_crm_chat.txt", mime="text/plain")

st.subheader("Conversation")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if df is not None:
    user_input = st.chat_input("Ask about stocks (e.g. 'Explain PE ratio' or 'Show Coal India financials')")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        query_type = classify_query(user_input)

        if query_type == "theoretical":
            reply = generate_crm_response(user_input, query_type="theoretical")
        else:
            context_md = extract_relevant_data(user_input, df).to_markdown(index=False)
            reply = generate_crm_response(user_input, context_md, query_type="fundamental")

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

if df is not None:
    with st.expander("Full Stock Data Table"):
        st.dataframe(df, use_container_width=True, height=300)

with st.expander("Setup Guide"):
    st.markdown("""
1. On [Streamlit Cloud](https://streamlit.io/cloud), go to **Settings → Secrets**.
2. Add:
```toml
OPENROUTER_API_KEY = "your_api_key_here"
DATABASE_URL = "your_database_url_here""")
