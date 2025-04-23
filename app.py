import streamlit as st
import pandas as pd
import requests
import os
import io
import re
from dotenv import load_dotenv
from sqlalchemy import create_engine
import logging

# Set up logging for debugging and error tracking
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# API and DB keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_ID = "deepseek/deepseek-r1:free"

st.set_page_config(page_title="💼 Stock Fundamental Analyst", layout="wide")

# Load stock data from PostgreSQL
@st.cache_data
def load_data():
    try:
        engine = create_engine(DATABASE_URL)
        query = "SELECT * FROM stock_data"
        df = pd.read_sql(query, engine)
        
        # Ensure column names match the CSV structure
        column_mapping = {
            'serial_number': 'Serial Number',
            'company_name': 'Company Name',
            'current_market_price': 'Current Market Price (Rs.)',
            'price_to_earnings_ratio': 'Price to Earnings Ratio',
            'market_cap': 'Market Capitalization (Rs. Cr.)',
            'dividend_yield': 'Dividend Yield (%)',
            'net_profit': 'Net Profit This Quarter (Rs. Cr.)',
            'profit_growth': 'Quarterly Profit Variation (%)',
            'sales': 'Sales This Quarter (Rs. Cr.)',
            'sales_growth': 'Quarterly Sales Variation (%)',
            'roce': 'Return on Capital Employed (%)',
            'avg_pat': 'Average PAT in Last 10 Years (Rs. Cr.)',
            'avg_dividend': 'Average Dividend Payout in Last 3 Years (%)'
        }
        
        # Rename columns to match CSV
        df = df.rename(columns=column_mapping)
        
        # Convert numeric columns
        num_cols = [
            'Current Market Price (Rs.)', 'Price to Earnings Ratio', 
            'Market Capitalization (Rs. Cr.)', 'Dividend Yield (%)',
            'Net Profit This Quarter (Rs. Cr.)', 'Quarterly Profit Variation (%)',
            'Sales This Quarter (Rs. Cr.)', 'Quarterly Sales Variation (%)',
            'Return on Capital Employed (%)', 'Average PAT in Last 10 Years (Rs. Cr.)',
            'Average Dividend Payout in Last 3 Years (%)'
        ]
        
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        logging.info("Data loaded successfully from PostgreSQL.")
        return df
    except Exception as e:
        logging.error(f"Error loading data from the database: {e}")
        st.error(f"Database connection failed: {e}")
        return None

# Enhanced query classification
def classify_query(query):
    query = query.lower()
    
    # More specific patterns for fundamental data queries
    fundamental_patterns = [
        r'pe ratio of', r'price to earnings of', r'market cap of',
        r'dividend yield of', r'net profit of', r'profit growth of',
        r'sales growth of', r'financials of', r'\beps of\b',
        r'earnings per share of', r'return on equity of', r'roce of',
        r'show me.*financial data', r'what is.*current market price of',
        r'give me.*financial metrics for', r'compare.*financials',
        r'list.*highest.*dividend', r'list.*lowest.*pe', r'ranking.*by.*market cap'
    ]
    
    # Check for fundamental patterns
    if any(re.search(pattern, query) for pattern in fundamental_patterns):
        return "fundamental"
    
    # Check for company names (only if query is clearly asking for data)
    if df is not None:
        company_names = df['Company Name'].str.lower().unique()
        if any(name in query for name in company_names):
            # Only treat as fundamental if asking for specific data
            data_terms = ['ratio', 'yield', 'profit', 'sales', 'growth', 
                         'price', 'cap', 'financial', 'metric', 'data',
                         'compare', 'list', 'show', 'what is', 'details']
            if any(term in query for term in data_terms):
                return "fundamental"
    
    # Default to theoretical
    return "theoretical"

# Data extractor for fundamental queries
def extract_relevant_data(query, df):
    try:
        clean_query = re.sub(r'[^a-zA-Z0-9\s]', '', query.lower())
        keywords = clean_query.split()

        # Handle ranking/comparison queries
        if any(word in query.lower() for word in ['list', 'ranking', 'top', 'highest', 'lowest']):
            if 'pe' in query.lower() or 'price to earnings' in query.lower():
                return df.sort_values('Price to Earnings Ratio').head(5)
            elif 'dividend' in query.lower():
                return df.sort_values('Dividend Yield (%)', ascending=False).head(5)
            elif 'market cap' in query.lower():
                return df.sort_values('Market Capitalization (Rs. Cr.)', ascending=False).head(5)
            elif 'profit growth' in query.lower():
                return df.sort_values('Quarterly Profit Variation (%)', ascending=False).head(5)
        
        # Handle company-specific queries
        matched_df = df[df['Company Name'].str.lower().str.contains('|'.join(keywords), na=False)]

        if not matched_df.empty:
            st.session_state.last_companies = matched_df['Company Name'].str.lower().unique().tolist()
            st.session_state.last_user_query = query
            return matched_df.head(3)

        fallback_companies = []
        for word in keywords:
            fallback_df = df[df['Company Name'].str.lower().str.contains(word, na=False)]
            if not fallback_df.empty:
                fallback_companies.append(word)

        if fallback_companies:
            st.session_state.last_companies = fallback_companies
        elif st.session_state.last_companies:
            matched_df = df[df['Company Name'].str.lower().str.contains('|'.join(st.session_state.last_companies), na=False)]
            return matched_df.head(3)

        return df.head(3)
    except Exception as e:
        logging.error(f"Error extracting relevant data: {e}")
        st.error(f"Error while processing your query: {e}")
        return df.head(3)  # Return some default data in case of error

# Enhanced response generator
def generate_crm_response(query, context_md=None, query_type="theoretical"):
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    # Strict system prompts
    if query_type == "theoretical":
        system_prompt = """You are an expert on Stock Market Concepts. Your task is to answer theoretical questions about  
stock market concepts, investing strategies, financial analysis techniques, and market terminologies.

STRICT RULES FOR THEORETICAL ANSWERS:

1. Only use fundamental data (e.g., revenue, earnings, P/E ratio) from the database strictly for explanation or illustration—do not reference actual real-time or historical stock prices.
2. Provide clear, conceptual explanations without referencing real stock tickers or external sources.
3. Explain financial ratios, metrics, and technical indicators in a simplified and educational manner.
4. Use standard financial terminology and explain terms like market cap, EPS, dividend yield, beta, etc. when relevant.
5. When asked about strategies, explain general frameworks such as value investing, growth investing, momentum trading, swing trading, etc., without endorsing specific companies.
6. If asked about charts or patterns, describe technical analysis tools like candlestick patterns, moving averages, RSI, MACD, Bollinger Bands, etc., in a theoretical context only.
7. Clarify types of markets (bull/bear), types of orders (market, limit, stop-loss), and types of investors/traders (retail, institutional, intraday, long-term).
8. When discussing training or learning paths, offer guidance on topics to study (e.g., financial statements, macroeconomic indicators, behavioral finance) but do not link to or suggest specific providers.
9. Refrain from providing investment advice or stock recommendations.
10. If you don't know the answer, simply say "I don't know" instead of guessing."""
    else:  # fundamental
        system_prompt = """You are a Stock Market Fundamental Data Analyst.
Your role is to help users analyze and interpret fundamental stock data from a structured database.

GUIDELINES FOR PRESENTING AND ANALYZING FUNDAMENTAL DATA:

1. Be precise with all numerical data, including stock prices, P/E ratios, market cap, revenue, net profit, EPS, ROE, ROCE, and debt-equity ratio.
2. Analyze core fundamental parameters of companies including but not limited to:
   - Current Market Price (CMP)
   - Market Capitalization
   - Price-to-Earnings (PE) Ratio
   - Earnings Per Share (EPS)
   - Return on Equity (ROE)
   - Debt to Equity Ratio
   - Net Profit Margin, Operating Margin
   - Revenue Growth, Profit Growth, etc.
3. You must answer any query regarding Indian stocks, Indian companies, or the Indian Stock Market, including indices like NIFTY 50, Sensex, Bank NIFTY, etc.
4. Support queries about industry-specific analysis in India—e.g., IT, Pharma, Banking, FMCG, Auto, Infra.
5. Respond to ranking or comparison queries, such as:
   - "Lowest PE stocks in the Indian stock market"
   - "Highest revenue companies in India"
   - "Top Indian companies by ROE or EPS"
   - "Best dividend-paying Indian stocks"
6. When asked for screeners or filters, provide logical, data-backed outputs, such as:
   - "Stocks with market cap > ₹10,000 Cr and PE < 15"
   - "Companies with consistent profit growth > 10% YoY for 5 years"
7. Clarify financial jargon or metrics when needed to help users understand the significance of the data.
8. Avoid providing technical analysis or price predictions—focus strictly on fundamentals.
9. Do not guess or assume unavailable data—if something is not found in the database, respond with "Data not available."
10. Ensure all answers are fact-driven, clear, and easy to understand, even for beginners."""

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Add chat history context
    for chat in st.session_state.chat_history[-5:]:
        messages.append({"role": chat["role"], "content": chat["content"]})

    # Handle follow-up questions
    if st.session_state.last_user_query and len(query.strip().split()) <= 5:
        query = f"{st.session_state.last_user_query} FOLLOW-UP: {query}"

    # Add the data context if available (for fundamental queries only)
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
        response.raise_for_status()  # This will raise an exception for HTTP errors
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"] \
            if "choices" in response_data else f"Unexpected response: {response_data}"
    except requests.exceptions.RequestException as e:
        logging.error(f"Error with API request: {e}")
        return f"API request failed: {e}"

# --- Load Data & Init State ---
df = load_data()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.last_companies = []
    st.session_state.last_user_query = ""

# --- Sidebar ---
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
            who = "User" if msg["role"] == "user" else "Analyst"
            buf.write(f"{who}: {msg['content']}\n\n")
        st.download_button("Download Chat", buf.getvalue(),
                         file_name="stock_analysis_chat.txt", mime="text/plain")

# --- Chat Display ---
st.title("💼 Stock Fundamental Analysis Assistant")
st.subheader("Ask about Indian stocks or market concepts")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Chat Input + AI Processing ---
if df is not None:
    user_input = st.chat_input("Ask about stocks (e.g. 'Explain PE ratio' or 'Show Coal India financials')")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Classify the query
        query_type = classify_query(user_input)
        
        if query_type == "theoretical":
            # For theoretical questions, don't need database context
            reply = generate_crm_response(user_input, query_type="theoretical")
        else:  # fundamental
            # For fundamental questions, extract relevant data
            relevant_data = extract_relevant_data(user_input, df)
            context_md = relevant_data.to_markdown(index=False)
            reply = generate_crm_response(user_input, context_md, query_type="fundamental")

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

# --- Data Table ---
if df is not None:
    with st.expander("📊 Full Stock Data Table"):
        st.dataframe(df, use_container_width=True, height=300)

# --- Setup Guide ---
with st.expander("⚙️ Setup Guide"):
    st.markdown("""
1. Get your OpenRouter API key from [OpenRouter](https://openrouter.ai)  
2. Create a `.env` file with this content:
```env
OPENROUTER_API_KEY=your_api_key_here
DATABASE_URL=your_database_url_here
```""")
