import os
import pickle
import sys
from typing import Any, Dict, Tuple

import numpy as np
import streamlit as st

# Configure project root path for modular imports
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.utils import calibrate_probability, clean_text

# Directory constants for non-transformer models
MODEL_DIR: str = os.path.join(BASE_DIR, "models")

# Hugging Face Hub Model Identifiers (Solution 2)
DISTILBERT_REPO: str = "distilbert-base-uncased-finetuned-sst-2-english"
SARCASM_REPO: str = "heytitan/roberta-base-sarcasm"  # Replace with your specific HF repo if hosted elsewhere

MAX_SEQUENCE_LENGTH: int = 200
BERT_MAX_LENGTH: int = 128

# Confidence threshold required to flag sarcasm
SARCASM_THRESHOLD: float = 0.85


# ============================================================
# PICKLE MODEL LOADING
# ============================================================

@st.cache_resource
def _load_pickle(filename: str) -> Any:
    """
    Load and cache a pickle model only when it is first needed.
    """
    filepath = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Model file not found: {filepath}"
        )

    with open(filepath, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def get_tfidf():
    return _load_pickle("tfidf_vectorizer.pkl")


@st.cache_resource
def get_naive_bayes():
    return _load_pickle("naive_bayes_model.pkl")


@st.cache_resource
def get_logistic_regression():
    return _load_pickle("logistic_regression_model.pkl")


@st.cache_resource
def get_lstm_tokenizer():
    return _load_pickle("tokenizer.pkl")


# ============================================================
# LSTM LOADING
# ============================================================

@st.cache_resource
def get_lstm_model():
    """
    Import TensorFlow only when the LSTM model is used.
    This prevents TensorFlow from loading during normal app startup.
    """
    from tensorflow.keras.models import load_model

    model_path = os.path.join(MODEL_DIR, "lstm_model.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"LSTM model not found: {model_path}"
        )

    return load_model(model_path)


# ============================================================
# DISTILBERT LOADING (Hugging Face Hub)
# ============================================================

@st.cache_resource
def get_distilbert():
    """
    Load DistilBERT directly from Hugging Face Hub.
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_REPO)

    model.eval()

    return tokenizer, model


# ============================================================
# SARCASM MODEL LOADING (Hugging Face Hub)
# ============================================================

@st.cache_resource
def get_sarcasm_model():
    """
    Load RoBERTa sarcasm model directly from Hugging Face Hub.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(SARCASM_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(SARCASM_REPO)

    model.eval()

    return tokenizer, model


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _extract_positive_index(model_classes: np.ndarray) -> int:
    """
    Find the probability index corresponding to the positive class.
    """
    classes_list = list(model_classes)

    for index, label in enumerate(classes_list):
        if str(label).lower() == "positive":
            return index

    return 1


def _extract_bert_positive_index(bert_model) -> int:
    """
    Find the positive class index from the DistilBERT model.
    """
    if hasattr(bert_model.config, "id2label"):
        for index, label in bert_model.config.id2label.items():
            if "positive" in str(label).lower():
                return int(index)

    return 1


def _extract_sarcasm_index(roberta_model) -> int:
    """
    Find the sarcasm class index from the RoBERTa model.
    """
    if hasattr(roberta_model.config, "id2label"):
        for index, label in roberta_model.config.id2label.items():
            label_text = str(label).lower()

            if any(
                key in label_text
                for key in ["sarcastic", "sarcasm", "pos", "1"]
            ):
                return int(index)

    return 1


# ============================================================
# SARCASM PREDICTION
# ============================================================

def predict_sarcasm(review: str) -> Tuple[bool, float]:
    """
    Evaluate input text for sarcastic tone.
    """

    import torch

    roberta_tokenizer, roberta_model = get_sarcasm_model()

    inputs = roberta_tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=BERT_MAX_LENGTH,
    )

    with torch.no_grad():
        outputs = roberta_model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    sarcasm_idx = _extract_sarcasm_index(roberta_model)

    sarcasm_prob = float(
        probabilities[sarcasm_idx].item()
    )

    is_sarcastic = sarcasm_prob >= SARCASM_THRESHOLD

    return is_sarcastic, round(sarcasm_prob, 4)
