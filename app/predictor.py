import os
import pickle
import re
import sys
from typing import Any, Callable, Dict

# Set environment variables for TensorFlow logging and optimization
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Add application and source directories to python path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")

for path in [BASE_DIR, SRC_DIR, APP_DIR]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import tensorflow as tf
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Optional spaCy import handling
try:
    import spacy
except ImportError:
    spacy = None

nlp = None

if spacy is not None:
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None

# Import project utilities
from utils import ID2LABEL, process_sentiment_output

# Directory configuration for saved models and global cache initialization
MODELS_DIR = os.path.join(BASE_DIR, "models")
_model_cache: Dict[str, Any] = {}


def normalize_typos_and_elongations(text: str) -> str:
    """Normalize repeated characters and common informal expressions."""
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    slang_map = {
        "luvd": "loved",
        "luvs": "loves",
        "awsum": "awesome",
        "freaking": "really",
    }

    tokens = text.split()

    return " ".join(slang_map.get(token.lower(), token) for token in tokens)


def enhanced_clean_raw(text: str) -> str:
    """Clean text while preserving original capitalization."""
    return normalize_typos_and_elongations(text.strip())


def enhanced_clean_text(text: str) -> str:
    """Clean and lowercase text for traditional ML models."""
    return normalize_typos_and_elongations(text.lower().strip())


def detect_idiomatic_adverbs(text: str) -> bool:
    """Detect expressions like 'like it bad', 'love it crazy', 'enjoy it hard'."""
    if nlp is None:
        return False

    doc = nlp(text.lower())

    positive_verbs = {
        "like",
        "love",
        "enjoy",
        "want",
        "miss",
        "dig",
    }

    intensifier_modifiers = {
        "bad",
        "hard",
        "crazy",
    }

    for token in doc:
        if token.text in intensifier_modifiers and token.dep_ == "advmod":
            if (
                token.head.lemma_ in positive_verbs
                and token.head.pos_ in {"VERB", "HEAD"}
            ):
                return True

    return False


def adjust_for_structural_patterns(text: str, score: float) -> float:
    """Adjust sentiment score using linguistic patterns."""
    lower_text = text.lower()

    # Idiomatic positive expressions check
    if detect_idiomatic_adverbs(text):
        score = max(score, 0.90)

    # Double negation patterns matching
    double_neg_pattern = (
        r"\b(can'?t|cannot)\s+say\s+(?:that\s+)?i\s+didn'?t\b|\bdidn'?t\s+dislike\b"
    )

    if re.search(double_neg_pattern, lower_text):
        if score < 0.50:
            score = 0.65 + (0.50 - score) * 0.40

    # Contrastive language pattern adjustments
    contrast_words = [
        " but ",
        " however ",
        " although ",
        " yet ",
    ]

    if any(word in lower_text for word in contrast_words):
        score = 0.50 + (score - 0.50) * 0.35

    return float(np.clip(score, 0.0, 1.0))


def get_sarcasm():
    """Load and cache the sarcasm detection model."""
    if "sarcasm" not in _model_cache:
        path = os.path.join(MODELS_DIR, "roberta_sarcasm")

        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True
        ).eval()

        _model_cache["sarcasm"] = (tokenizer, model)

    return _model_cache["sarcasm"]


def get_distilbert():
    """Load and cache the DistilBERT sentiment model."""
    if "distilbert" not in _model_cache:
        path = os.path.join(MODELS_DIR, "distilbert_model")

        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True
        ).eval()

        _model_cache["distilbert"] = (tokenizer, model)

    return _model_cache["distilbert"]


def get_naive_bayes():
    """Load and cache Naive Bayes and TF-IDF models."""
    if "naive_bayes" not in _model_cache:
        nb_path = os.path.join(MODELS_DIR, "naive_bayes_model.pkl")

        tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

        with open(nb_path, "rb") as file:
            nb_model = pickle.load(file)

        with open(tfidf_path, "rb") as file:
            tfidf = pickle.load(file)

        _model_cache["naive_bayes"] = (nb_model, tfidf)

    return _model_cache["naive_bayes"]


def get_logistic_regression():
    """Load and cache Logistic Regression model."""
    if "logistic" not in _model_cache:
        lr_path = os.path.join(MODELS_DIR, "logistic_regression_model.pkl")

        with open(lr_path, "rb") as file:
            lr_model = pickle.load(file)

        _, tfidf = get_naive_bayes()

        _model_cache["logistic"] = (lr_model, tfidf)

    return _model_cache["logistic"]


def get_bilstm():
    """Load and cache BiLSTM model and tokenizer."""
    if "bilstm" not in _model_cache:
        lstm_path = os.path.join(MODELS_DIR, "lstm_model.keras")

        tokenizer_path = os.path.join(MODELS_DIR, "tokenizer.pkl")

        lstm_model = tf.keras.models.load_model(lstm_path)

        with open(tokenizer_path, "rb") as file:
            tokenizer = pickle.load(file)

        _model_cache["bilstm"] = (lstm_model, tokenizer)

    return _model_cache["bilstm"]


def _run_model_pipeline(
    model_name: str,
    text: str,
    prob_extractor: Callable[[str], np.ndarray],
) -> Dict[str, Any]:
    """Run probability extraction and sentiment calibration."""
    try:
        probs = prob_extractor(text)

        score_weights = np.array([0.0, 0.25, 0.50, 0.75, 1.0])

        raw_score = float(np.sum(probs * score_weights))

        adjusted_score = adjust_for_structural_patterns(text, raw_score)

        processed = process_sentiment_output(text, adjusted_score)

        return {
            "model": model_name,
            "class_id": processed["predicted_id"],
            "sentiment": processed["predicted_label"],
            "confidence": round(float(np.max(probs)), 4),
            "positive_prob": processed["calibrated_score"],
            "probabilities": [round(float(prob), 4) for prob in probs],
        }

    except Exception:
        return {
            "model": model_name,
            "class_id": 2,
            "sentiment": "Neutral",
            "confidence": 0.20,
            "positive_prob": 0.50,
            "probabilities": [0.20] * 5,
        }


def predict_sarcasm(text: str) -> Dict[str, Any]:
    """Predict whether the review is sarcastic."""
    try:
        tokenizer, model = get_sarcasm()

        raw_text = enhanced_clean_raw(text)

        inputs = tokenizer(
            raw_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )

        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().numpy()

        sarcasm_prob = (
            float(probs[1]) if len(probs) > 1 else float(probs[0])
        )

        # Suppress false positives for very short reviews
        if len(raw_text.strip().split()) <= 4:
            sarcasm_prob = min(sarcasm_prob, 0.20)

        return {
            "is_sarcastic": sarcasm_prob >= 0.85,
            "sarcasm_prob": round(sarcasm_prob, 4),
        }

    except Exception:
        return {
            "is_sarcastic": False,
            "sarcasm_prob": 0.0,
        }


def predict_distilbert(text: str) -> Dict[str, Any]:
    """Predict sentiment using DistilBERT."""
    try:
        tokenizer, model = get_distilbert()

        raw_text = enhanced_clean_raw(text)

        inputs = tokenizer(
            raw_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )

        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().numpy()

        pred_class_id = int(np.argmax(probs))

        score_weights = np.array([0.0, 0.25, 0.50, 0.75, 1.0])

        raw_score = float(np.sum(probs * score_weights))

        # Sarcasm probability check and score adjustment
        sarcasm_result = predict_sarcasm(text)

        if pred_class_id in [3, 4] and sarcasm_result["is_sarcastic"]:
            raw_score *= 1.0 - sarcasm_result["sarcasm_prob"]

        adjusted_score = adjust_for_structural_patterns(text, raw_score)

        processed = process_sentiment_output(text, adjusted_score)

        return {
            "model": "DistilBERT",
            "class_id": processed["predicted_id"],
            "sentiment": processed["predicted_label"],
            "confidence": round(float(np.max(probs)), 4),
            "positive_prob": processed["calibrated_score"],
            "probabilities": [round(float(prob), 4) for prob in probs],
            "is_sarcastic": sarcasm_result["is_sarcastic"],
            "sarcasm_prob": sarcasm_result["sarcasm_prob"],
        }

    except Exception:
        return {
            "model": "DistilBERT",
            "class_id": 2,
            "sentiment": "Neutral",
            "confidence": 0.20,
            "positive_prob": 0.50,
            "probabilities": [0.20] * 5,
        }


def predict_naive_bayes(text: str) -> Dict[str, Any]:
    """Predict sentiment using Naive Bayes."""

    def extract(review: str) -> np.ndarray:
        model, tfidf = get_naive_bayes()

        transformed = tfidf.transform([enhanced_clean_text(review)])

        return model.predict_proba(transformed)[0]

    return _run_model_pipeline("Naive Bayes", text, extract)


def predict_logistic_regression(text: str) -> Dict[str, Any]:
    """Predict sentiment using Logistic Regression."""

    def extract(review: str) -> np.ndarray:
        model, tfidf = get_logistic_regression()

        transformed = tfidf.transform([enhanced_clean_text(review)])

        return model.predict_proba(transformed)[0]

    return _run_model_pipeline("Logistic Regression", text, extract)


def predict_bilstm(text: str) -> Dict[str, Any]:
    """Predict sentiment using BiLSTM."""

    def extract(review: str) -> np.ndarray:
        model, tokenizer = get_bilstm()

        sequences = tokenizer.texts_to_sequences([enhanced_clean_text(review)])

        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequences,
            maxlen=100,
            padding="post",
            truncating="post",
        )

        return model.predict(padded, verbose=0)[0]

    return _run_model_pipeline("BiLSTM", text, extract)


# Backwards compatibility reference
predict_lstm = predict_bilstm


def predict_all_models(text: str) -> Dict[str, Dict[str, Any]]:
    """Run prediction across all sentiment models."""
    return {
        "distilbert": predict_distilbert(text),
        "logistic_regression": predict_logistic_regression(text),
        "naive_bayes": predict_naive_bayes(text),
        "bilstm": predict_bilstm(text),
    }
