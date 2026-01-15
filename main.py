import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from transformers import pipeline

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="Social Media Comment Analyzer",
    layout="wide"
)

st.title("📊 Social Media Comment Analyzer (ML + LLM)")
st.write("Upload social media comments and get sentiment & public opinion insights")

# ===============================
# LOAD MODELS (CACHED)
# ===============================
@st.cache_resource
def load_models():
    svm_model = joblib.load("models/svm_model.pkl")
    xgb_model = joblib.load("models/xgb_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    insight_llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )

    return svm_model, xgb_model, vectorizer, insight_llm


svm_model, xgb_model, vectorizer, insight_llm = load_models()

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


def generate_ai_insights(pos_pct, neg_pct, bad_word):
    prompt = (
        "Instruction: You are a social media sentiment analyst.\n\n"
        f"Positive sentiment percentage: {pos_pct:.2f}%\n"
        f"Negative sentiment percentage: {neg_pct:.2f}%\n"
        f"Most frequently used offensive word: {bad_word}\n\n"
        "Task:\n"
        "1. Describe the overall public sentiment.\n"
        "2. Explain what people think about the post or video.\n"
        "3. Mention whether the reaction is mostly positive or negative.\n\n"
        "Answer:"
    )

    try:
        output = insight_llm(
            prompt,
            max_length=200,
            min_length=60,
            do_sample=False,
            truncation=True
        )

        generated_text = output[0]["generated_text"].strip()

        # SAFETY FALLBACK
        if len(generated_text) < 20:
            raise ValueError("LLM returned insufficient output")

        return generated_text

    except Exception:
        # GUARANTEED FALLBACK INSIGHT
        return (
            f"The overall public sentiment is "
            f"{'positive' if pos_pct > neg_pct else 'negative'}. "
            f"Approximately {pos_pct:.2f}% of users expressed positive opinions, "
            f"while {neg_pct:.2f}% showed negative reactions. "
            f"The frequent use of the word '{bad_word}' indicates strong emotional responses. "
            f"Overall, public opinion suggests mixed but opinionated engagement with the post."
        )



# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader(
    "Upload file (CSV, Excel, JSON)",
    type=["csv", "xlsx", "json"]
)

if uploaded_file:
    # -------------------------------
    # LOAD DATA (FIXED)
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
    # SARCASM
    # -------------------------------
    df["sarcastic"] = comments.apply(detect_sarcasm)
    df.loc[df["sarcastic"], "sentiment"] = "Negative"

    # -------------------------------
    # METRICS + PERCENTAGES
    # -------------------------------
    total = len(df)
    positive_count = (df["sentiment"] == "Positive").sum()
    negative_count = (df["sentiment"] == "Negative").sum()

    positive_pct = (positive_count / total) * 100
    negative_pct = (negative_count / total) * 100

    bad_word = most_used_bad_word(comments.tolist())

    # -------------------------------
    # DISPLAY METRICS
    # -------------------------------
    st.subheader("📊 Sentiment Breakdown")

    col1, col2 = st.columns(2)
    col1.metric(
        "Positive",
        f"{positive_pct:.2f}%",
        f"{positive_count} comments"
    )
    col2.metric(
        "Negative",
        f"{negative_pct:.2f}%",
        f"{negative_count} comments"
    )

    st.write(f"**Most used bad word:** `{bad_word}`")

    # -------------------------------
    # AI INSIGHTS (FIXED)
    # -------------------------------
    st.subheader("🧠 AI Public Opinion Insight")

    with st.spinner("Generating insights..."):
        insight_text = generate_ai_insights(
            positive_pct,
            negative_pct,
            bad_word
        )

    st.success("Insight generated successfully")
    st.write(insight_text)

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
