import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="Social Media Comment Analyzer",
    layout="wide"
)

st.title("📊 Social Media Comment Analyzer")
st.write("Upload social media comments and get sentiment & public opinion insights")

# ===============================
# LOAD MODELS (CACHED)
# ===============================
@st.cache_resource
def load_models():
    svm_model = joblib.load("models/svm_model.pkl")
    xgb_model = joblib.load("models/xgb_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return svm_model, xgb_model, vectorizer


svm_model, xgb_model, vectorizer = load_models()

# ===============================
# CONSTANTS
# ===============================
SARCASM_PATTERNS = [
    "yeah right", "sure buddy", "nice job", "lol", "lmao",
    "great job", "amazing 🙄", "as if", "/s"
]

BAD_WORDS = [
    "stupid", "idiot", "dumb", "hate", "worst",
    "shit", "fuck", "crap", "nonsense"
]

# ===============================
# HELPER FUNCTIONS
# ===============================
def detect_comment_column(df):
    text_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            avg_len = df[col].astype(str).str.len().mean()
            if avg_len > 15:
                text_cols.append(col)

    priority_keywords = ["comment", "text", "tweet", "review", "message"]
    for col in text_cols:
        if any(k in col.lower() for k in priority_keywords):
            return col

    return text_cols[0] if text_cols else None


def detect_sarcasm(text):
    text = text.lower()
    return any(p in text for p in SARCASM_PATTERNS)


def most_used_bad_word(texts):
    counter = {}
    for text in texts:
        for word in BAD_WORDS:
            if re.search(rf"\b{word}\b", text.lower()):
                counter[word] = counter.get(word, 0) + 1
    return max(counter, key=counter.get) if counter else "None"


def generate_ai_insights(total, pos, neg, pos_pct, neg_pct, sarcastic, bad_word):
    if pos_pct > neg_pct:
        sentiment = "mostly positive"
    elif neg_pct > pos_pct:
        sentiment = "mostly negative"
    else:
        sentiment = "mixed"

    insight = f"""
PUBLIC SENTIMENT ANALYSIS REPORT
--------------------------------
Total comments analyzed: {total}

Sentiment Breakdown:
- Positive comments: {pos} ({pos_pct:.2f}%)
- Negative comments: {neg} ({neg_pct:.2f}%)

Sarcasm Detection:
- Sarcastic comments detected: {sarcastic}

Language Usage:
- Most frequently used offensive word: {bad_word}

Overall Public Opinion:
The overall public sentiment is {sentiment}. Users have expressed strong opinions,
and the tone of discussion indicates noticeable engagement with the post or video.
"""

    return insight.strip()

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader(
    "Upload file (CSV, Excel, JSON)",
    type=["csv", "xlsx", "json"]
)

if uploaded_file:
    # -------------------------------
    # LOAD DATA
    # -------------------------------
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="latin1")
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_json(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # -------------------------------
    # DETECT COMMENT COLUMN
    # -------------------------------
    comment_col = detect_comment_column(df)
    if not comment_col:
        st.error("❌ Could not detect comment column.")
        st.stop()

    st.success(f"✅ Detected comment column: **{comment_col}**")

    comments = df[comment_col].astype(str)

    # -------------------------------
    # PREDICTIONS
    # -------------------------------
    X = vectorizer.transform(comments)
    df["prediction"] = xgb_model.predict(X)

    df["sentiment"] = df["prediction"].map({
        1: "Positive",
        0: "Negative"
    })

    # -------------------------------
    # SARCASM HANDLING
    # -------------------------------
    df["sarcastic"] = comments.apply(detect_sarcasm)
    df.loc[df["sarcastic"], "sentiment"] = "Negative"

    # -------------------------------
    # METRICS
    # -------------------------------
    total = len(df)
    positive_count = (df["sentiment"] == "Positive").sum()
    negative_count = (df["sentiment"] == "Negative").sum()
    sarcastic_count = df["sarcastic"].sum()

    positive_pct = (positive_count / total) * 100
    negative_pct = (negative_count / total) * 100

    bad_word = most_used_bad_word(comments.tolist())

    # -------------------------------
    # DISPLAY METRICS
    # -------------------------------
    st.subheader("📊 Sentiment Breakdown")

    col1, col2 = st.columns(2)
    col1.metric("Positive", f"{positive_pct:.2f}%", f"{positive_count} comments")
    col2.metric("Negative", f"{negative_pct:.2f}%", f"{negative_count} comments")

    st.write(f"**Most used bad word:** `{bad_word}`")

    # -------------------------------
    # AI INSIGHTS (NO TORCH)
    # -------------------------------
    st.subheader("🧠 Public Opinion Insight")

    insight_text = generate_ai_insights(
        total,
        positive_count,
        negative_count,
        positive_pct,
        negative_pct,
        sarcastic_count,
        bad_word
    )

    st.text_area("Generated Insight", insight_text, height=300)

    # -------------------------------
    # FILTERED VIEWS
    # -------------------------------
    st.subheader("🟢 Positive Comments")
    st.dataframe(df[df["sentiment"] == "Positive"][[comment_col]])

    st.subheader("🔴 Negative Comments")
    st.dataframe(df[df["sentiment"] == "Negative"][[comment_col]])

    # -------------------------------
    # DOWNLOADS
    # -------------------------------
    st.subheader("⬇️ Download Results")

    st.download_button(
        "Download Classified Data",
        df.to_csv(index=False),
        "classified_comments.csv"
    )

    st.download_button(
        "Download AI Insights",
        insight_text,
        "public_opinion_insights.txt"
    )
