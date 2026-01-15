# Social Media Comment Sentiment Analyzer
(Linear SVM + XGBoost + Streamlit + LLM Insights)

## 📌 Project Overview

This project is an end-to-end sentiment analysis system designed to analyze social media comments (e.g., Twitter, YouTube, Instagram).

It:

Bifurcates comments into Positive and Negative
Detects sarcastic comments (treated as negative)
Generates AI-based public opinion insights
Works with any uploaded dataset format
Uses traditional ML models (Linear SVM & XGBoost) for classification
Uses a free, local LLM (no API key) for qualitative insights
Provides downloadable results
The application is deployed using Streamlit for an interactive user interface.

## 🎯 Problem Statement

Social media platforms generate massive volumes of unstructured text data.
Manually analyzing public opinion is inefficient and error-prone.
This project solves that problem by:
Automatically detecting the comment column
Classifying sentiment at scale
Providing high-level insights into public opinion
Ensuring fast, explainable, and cost-effective analysis

## 🧠 Models Used
### 1️⃣ Linear Support Vector Machine (SVM)

Good for high-dimensional sparse text data

Fast inference

Strong baseline for sentiment classification

### 2️⃣ XGBoost (Final Model)

Handles non-linear decision boundaries

Better performance compared to linear models

Used as the final sentiment predictor

### 3️⃣ LLM for Insights (Free & Offline)

Model: google/flan-t5-small

Used only for insight generation

Runs locally using Hugging Face Transformers

### 💾 Model Saving Using .pkl Files
After training the models in Jupyter Notebook, we saved them using joblib:

import joblib

joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

### **Why .pkl Files?**

Fast loading

Preserves trained weights

Enables reuse without retraining

Ideal for deployment

## 🖥️ How the Streamlit App Works 

### 1️⃣ Upload Dataset

Supported formats:
CSV, Excel (.xlsx), JSON

### 2️⃣ Automatic Comment Column Detection

The app intelligently detects the comment/text column using:
Data type checks
Average text length
Column name keywords (comment, tweet, review, etc.)

### 3️⃣ Sentiment Classification

Text is vectorized using the saved TF-IDF vectorizer
Predictions are made using XGBoost
Results labelled as Positive or Negative

### 4️⃣ Sarcasm Detection

Rule-based detection using common sarcasm phrases
Sarcastic comments are counted and treated as negative

### 5️⃣ AI Insight Generation

The LLM generates insights such as:

Overall public sentiment
Public opinion on the post/video
Dominant emotional tone
Most frequently used offensive word

### 6️⃣ Download Results

Users can download:
Classified comments (CSV)
AI-generated public opinion insights (TXT)

## How to run the app
Install Dependencies
- pip install -r requirements.txt

Run Streamlit App
- streamlit run app.py


