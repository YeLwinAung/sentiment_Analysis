import os
import pickle
import sys
from typing import Any, Dict, Tuple

numpy import np
import streamlit as st

# Configure base directory path relative to current file location
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import string preprocessing and calibration utilities
try:
    from app.utils import calibrate_probability, clean_text
except ImportError:

    def clean_text(text: str) -> str:
        return text.strip().lower()

    def calibrate_probability(text: str, prob: float) -> float:
        return prob


# Directory path configurations
MODEL_DIR: str = os.path.join(BASE_DIR, "models")
DISTILBERT_DIR: str = os.path.join(MODEL_DIR, "distilbert_model")
SARCASM_DIR: str = os.path.join(MODEL_DIR, "roberta_sarcasm")

# Model hyperparameter thresholds
MAX_SEQUENCE_LENGTH: int = 200
BERT_MAX_LENGTH: int = 128
SARCASM_THRESHOLD: float = 0.85


# Model loader functions
@st.cache_resource
def _load_pickle(filename: str) -> Any:
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        filepath = os.path.join(BASE_DIR, "app", "models", filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pickle file missing: {filepath}")

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


@st.cache_resource
def get_lstm_model():
    from tensorflow.keras.models import load_model

    model_path = os.path.join(MODEL_DIR, "lstm_model.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "app", "models", "lstm_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM model file missing: {model_path}")

    return load_model(model_path)


@st.cache_resource
def get_distilbert():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    target_dir = DISTILBERT_DIR
    if not os.path.exists(target_dir):
        target_dir = os.path.join(BASE_DIR, "app", "models", "distilbert_model")
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"DistilBERT directory missing: {target_dir}")

    tokenizer = AutoTokenizer.from_pretrained(target_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        target_dir, local_files_only=True
    )
    model.eval()

    return tokenizer, model


@st.cache_resource
def get_sarcasm_model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    target_dir = SARCASM_DIR
    if not os.path.exists(target_dir):
        target_dir = os.path.join(BASE_DIR, "app", "models", "roberta_sarcasm")
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Sarcasm directory missing: {target_dir}")

    tokenizer = AutoTokenizer.from_pretrained(target_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        target_dir, local_files_only=True
    )
    model.eval()

    return tokenizer, model


# Label indexing helper functions
def _extract_positive_index(model_classes: np.ndarray) -> int:
    classes_list = list(model_classes)
    for index, label in enumerate(classes_list):
        if str(label).lower() in ["positive", "pos", "1"]:
            return index
    return 1


def _extract_bert_positive_index(bert_model) -> int:
    if hasattr(bert_model.config, "id2label"):
        for index, label in bert_model.config.id2label.items():
            if "positive" in str(label).lower():
                return int(index)
    return 1


def _extract_sarcasm_index(roberta_model) -> int:
    if hasattr(roberta_model.config, "id2label"):
        for index, label in roberta_model.config.id2label.items():
            label_text = str(label).lower()
            if any(
                key in label_text for key in ["sarcastic", "sarcasm", "pos", "1"]
            ):
                return int(index)
    return 1


# Prediction and adjustment pipelines
def predict_sarcasm(review: str) -> Tuple[bool, float]:
    try:
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

        probabilities = torch.softmax(outputs.logits, dim=1)[0]
        sarcasm_idx = _extract_sarcasm_index(roberta_model)
        sarcasm_prob = float(probabilities[sarcasm_idx].item())
        is_sarcastic = sarcasm_prob >= SARCASM_THRESHOLD

        return is_sarcastic, round(sarcasm_prob, 4)
    except Exception:
        return False, 0.0


def _apply_sarcasm_adjustment(
    raw_prob: float, is_sarcastic: bool, sarcasm_prob: float
) -> float:
    if is_sarcastic:
        weight = (sarcasm_prob - SARCASM_THRESHOLD) / (1.0 - SARCASM_THRESHOLD)
        weight = max(0.0, min(1.0, weight))
        inverted = 1.0 - raw_prob
        return raw_prob * (1 - weight) + inverted * weight
    return raw_prob


def derive_7class_sentiment(prob: float) -> str:
    if prob >= 0.90:
        return "Overwhelmingly Positive"
    elif prob >= 0.75:
        return "Very Positive"
    elif prob >= 0.55:
        return "Positive"
    elif prob >= 0.45:
        return "Mixed"
    elif prob >= 0.25:
        return "Negative"
    elif prob >= 0.10:
        return "Very Negative"
    else:
        return "Overwhelmingly Negative"


def _build_result(review: str, raw_prob: float) -> Dict[str, Any]:
    raw_prob = max(0.0, min(1.0, raw_prob))
    is_sarcastic, sarcasm_prob = predict_sarcasm(review)
    adjusted_prob = _apply_sarcasm_adjustment(
        raw_prob, is_sarcastic, sarcasm_prob
    )
    calibrated_prob = calibrate_probability(review, adjusted_prob)
    calibrated_prob = max(0.0, min(1.0, float(calibrated_prob)))

    return {
        "sentiment": derive_7class_sentiment(calibrated_prob),
        "positive_prob": round(calibrated_prob, 4),
        "is_sarcastic": is_sarcastic,
        "sarcasm_prob": sarcasm_prob,
    }


# Public API model inference methods
def predict_naive_bayes(review: str) -> Dict[str, Any]:
    try:
        tfidf = get_tfidf()
        naive_bayes = get_naive_bayes()
        text = clean_text(review) or review
        vector = tfidf.transform([text])
        probabilities = naive_bayes.predict_proba(vector)[0]
        pos_idx = _extract_positive_index(naive_bayes.classes_)
        return _build_result(review, float(probabilities[pos_idx]))
    except Exception:
        return _build_result(review, 0.50)


def predict_logistic_regression(review: str) -> Dict[str, Any]:
    try:
        tfidf = get_tfidf()
        logistic_model = get_logistic_regression()
        text = clean_text(review) or review
        vector = tfidf.transform([text])
        probabilities = logistic_model.predict_proba(vector)[0]
        pos_idx = _extract_positive_index(logistic_model.classes_)
        return _build_result(review, float(probabilities[pos_idx]))
    except Exception:
        return _build_result(review, 0.50)


def predict_lstm(review: str) -> Dict[str, Any]:
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        lstm_model = get_lstm_model()
        lstm_tokenizer = get_lstm_tokenizer()
        text = clean_text(review) or review
        sequence = lstm_tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(
            sequence,
            maxlen=MAX_SEQUENCE_LENGTH,
            padding="pre",
            truncating="pre",
        )
        raw_prob = float(lstm_model.predict(padded_sequence, verbose=0)[0][0])
        return _build_result(review, raw_prob)
    except Exception:
        return _build_result(review, 0.50)


def predict_distilbert(review: str) -> Dict[str, Any]:
    try:
        import torch

        bert_tokenizer, bert_model = get_distilbert()
        inputs = bert_tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=BERT_MAX_LENGTH,
        )

        with torch.no_grad():
            outputs = bert_model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)[0]
        pos_idx = _extract_bert_positive_index(bert_model)
        raw_prob = float(probabilities[pos_idx].item())
        return _build_result(review, raw_prob)
    except Exception:
        return _build_result(review, 0.50)
