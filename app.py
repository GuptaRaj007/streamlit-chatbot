import streamlit as st
import pandas as pd
import requests
import os
import io
import re
from dotenv import load_dotenv
from sqlalchemy import create_engine
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.docstore.document import Document  # For proper document handling

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# API and DB keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_ID = "deepseek/deepseek-r1:free"

st.set_page_config(page_title="💼 Stock Fundamental Analyst", layout="wide")

# Initialize embeddings with updated package
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Custom OpenRouter client
def openrouter_chat_completion(messages, temperature=0.3, max_tokens=1024):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",  # Add required headers
        "X-Title": "Stock Analyst"
    }
    
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30  # Add timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"OpenRouter API error: {e}")
        return f"Error generating response: {str(e)}"

# Load stock data from PostgreSQL and prepare vector store
@st.cache_resource
def load_and_prepare_data():
    try:
        engine = create_engine(DATABASE_URL)
        query = "SELECT * FROM stock_data"
        df = pd.read_sql(query, engine)
        
        # Standardize column names
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
        
        # Create proper Document objects
        documents = []
        for _, row in df.iterrows():
            content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
            documents.append(Document(
                page_content=content,
                metadata={"company": row['Company Name']}
            ))
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        
        logging.info("Data loaded and vector store created successfully.")
        return df, vectorstore
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        st.error(f"Data loading failed: {e}")
        return None, None

# Enhanced query classifier
def classify_query(query, vectorstore, threshold=0.75):
    query = query.lower()
    
    fundamental_patterns = [
        r'pe ratio of', r'price to earnings of', r'market cap of',
        r'dividend yield of', r'net profit of', r'profit growth of',
        r'sales growth of', r'financials of', r'\beps of\b',
        r'earnings per share of', r'return on equity of', r'roce of',
        r'show me.*financial data', r'what is.*current market price of',
        r'give me.*financial metrics for', r'compare.*financials',
        r'list.*highest.*dividend', r'list.*lowest.*pe', r'ranking.*by.*market cap',
        r'price of', r'value of', r'financial data', r'financial details',
        r'stock data', r'stock details'
    ]
    
    if any(re.search(pattern, query) for pattern in fundamental_patterns):
        return "fundamental"
    
    if vectorstore:
        try:
            similar_docs = vectorstore.similarity_search_with_score(query, k=1)
            if similar_docs and similar_docs[0][1] < threshold:
                return "fundamental"
        except Exception as e:
            logging.error(f"Similarity search error: {e}")
    
    theoretical_patterns = [
        r'what is', r'explain', r'how does', r'define', r'meaning of',
        r'difference between', r'compare.*and', r'benefits of', r'risks of',
        r'pros and cons', r'advantages', r'disadvantages', r'concept of',
        r'theory of', r'basics of', r'introduction to'
    ]
    
    if any(re.search(pattern, query) for pattern in theoretical_patterns):
        return "theoretical"
    
    if 'last_query_type' in st.session_state and st.session_state.last_query_type == "fundamental":
        return "fundamental"
    return "theoretical"

# Data retrieval function for RAG
def retrieve_relevant_data(query, vectorstore, df, k=3):
    try:
        company_names = df['Company Name'].str.lower().unique()
        matched_companies = [name for name in company_names if name in query.lower()]
        
        if matched_companies:
            matched_df = df[df['Company Name'].str.lower().isin(matched_companies)]
            if not matched_df.empty:
                return matched_df
        
        if vectorstore:
            docs = vectorstore.similarity_search(query, k=k)
            company_names_from_docs = [doc.metadata.get('company', '') for doc in docs]
            if company_names_from_docs:
                matched_df = df[df['Company Name'].isin(company_names_from_docs)]
                if not matched_df.empty:
                    return matched_df
        
        return df.head(k)
    except Exception as e:
        logging.error(f"Error in retrieve_relevant_data: {e}")
        return df.head(k)

# Prompt templates
def get_theoretical_prompt():
    template = """You are an expert on Stock Market Concepts. Answer the following question about stock market theory.

Question: {question}

Guidelines:
1. Provide clear, conceptual explanations without referencing specific stocks unless asked
2. Explain financial terms and concepts in simple language
3. Use examples where helpful, but keep them generic
4. If unsure, say "I don't know" rather than guessing

Answer:"""
    return ChatPromptTemplate.from_template(template)

def get_fundamental_prompt():
    template = """You are a Stock Market Data Analyst. Analyze the following data and answer the question.

Relevant Data:
{context}

Question: {question}

Guidelines:
1. Be precise with all numerical data
2. Only use the provided data - don't make up numbers
3. Explain what the metrics mean when appropriate
4. Compare companies if requested
5. If data is insufficient, say "The available data doesn't show that information"

Answer:"""
    return ChatPromptTemplate.from_template(template)

# Initialize the app
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.last_query_type = "theoretical"
    st.session_state.last_companies = []

# Load data and models
df, vectorstore = load_and_prepare_data()

# --- Sidebar ---
with st.sidebar:
    st.header("\u2699\ufe0f Chat Options")
    if st.button("Clear Chat History \U0001F9B9"):
        st.session_state.chat_history = []
        st.session_state.last_companies = []
        st.session_state.last_query_type = "theoretical"
        st.success("Chat history cleared!")

    if st.session_state.chat_history:
        buf = io.StringIO()
        for msg in st.session_state.chat_history:
            who = "User" if msg["role"] == "user" else "Analyst"
            buf.write(f"{who}: {msg['content']}\n\n")
        st.download_button(
            "Download Chat", 
            buf.getvalue(),
            file_name="stock_analysis_chat.txt", 
            mime="text/plain"
        )

# --- Chat Display ---
st.title("💼 Stock Fundamental Analysis Assistant")
st.subheader("Ask about Indian stocks or market concepts")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Chat Input + Processing ---
# --- Chat Input + Processing ---
if df is not None and vectorstore is not None:
    user_input = st.chat_input("Ask about stocks (e.g. 'Explain PE ratio' or 'Show Coal India financials')")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        query_type = classify_query(user_input, vectorstore)
        st.session_state.last_query_type = query_type
        
        # Prepare messages with conversation history
        messages = []
        
        # Add system message based on query type
        if query_type == "theoretical":
            system_message = """You are an expert on Stock Market Concepts. Answer questions about stock market theory.
            Guidelines:
            1. Provide clear, conceptual explanations
            2. Explain financial terms in simple language
            3. Use examples where helpful
            4. If unsure, say "I don't know" rather than guessing"""
        else:
            system_message = """You are a Stock Market Data Analyst. Analyze data and answer questions.
            Guidelines:
            1. Be precise with numerical data
            2. Only use provided data
            3. Explain metrics when appropriate
            4. Compare companies if requested
            5. Say "Data not available" if insufficient"""
        
        messages.append({"role": "system", "content": system_message})
        
        # Add conversation history (last 3 exchanges)
        for msg in st.session_state.chat_history[-6:]:  # Keep last 3 exchanges (6 messages)
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # For fundamental queries, add relevant data context
        if query_type == "fundamental":
            relevant_data = retrieve_relevant_data(user_input, vectorstore, df)
            context_md = relevant_data.to_markdown(index=False)
            messages[-1]["content"] = f"DATA CONTEXT:\n{context_md}\n\nQUESTION: {messages[-1]['content']}"
        
        # Get response from OpenRouter
        reply = openrouter_chat_completion(messages)
        
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
DATABASE_URL=your_database_url_here""")
