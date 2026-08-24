import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

MODEL_DIR = os.path.join(PARENT_DIR, "models")
SARCASM_DIR = os.path.join(MODEL_DIR, "sarcasm")

# Safe Transformer / Torch imports
_HAS_TRANSFORMERS = False
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _HAS_TRANSFORMERS = True
except ImportError:
    pass


def predict_sarcasm(text: str) -> tuple[bool, float]:
    """Predicts sarcasm using RoBERTa if available, falling back to rule-based heuristics."""
    if not _HAS_TRANSFORMERS or not os.path.exists(SARCASM_DIR):
        # Fallback heuristic if local model/tokenizer fails to load
        sarcastic_words = {"yeah right", "sure", "totally", "oh great", "whatever", "marvelous"}
        is_sarcastic = any(w in text.lower() for w in sarcastic_words)
        return is_sarcastic, 0.8 if is_sarcastic else 0.1

    try:
        # Pass use_fast=False to prevent tokenizer json deserialization crashes
        tokenizer = AutoTokenizer.from_pretrained(SARCASM_DIR, local_files_only=True, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(SARCASM_DIR, local_files_only=True)
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
            
        prob_sarcastic = float(probs[1]) if isinstance(probs, list) and len(probs) > 1 else 0.5
        return (prob_sarcastic > 0.5), prob_sarcastic

    except Exception as e:
        print(f"[BACKEND WARNING] Sarcasm model failed to load ({e}). Using heuristic fallback.")
        sarcastic_words = {"yeah right", "sure", "totally", "oh great", "whatever"}
        is_sarcastic = any(w in text.lower() for w in sarcastic_words)
        return is_sarcastic, 0.8 if is_sarcastic else 0.1


def predict_distilbert(text: str) -> dict:
    """Predicts sentiment using DistilBERT model if available, otherwise returning a safe baseline."""
    distilbert_dir = os.path.join(MODEL_DIR, "distilbert")
    
    if _HAS_TRANSFORMERS and os.path.exists(distilbert_dir):
        try:
            tokenizer = AutoTokenizer.from_pretrained(distilbert_dir, local_files_only=True, use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained(distilbert_dir, local_files_only=True)
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
            
            pos_prob = float(probs[1]) if isinstance(probs, list) and len(probs) > 1 else 0.5
            return {"positive_prob": pos_prob, "label": "POSITIVE" if pos_prob >= 0.5 else "NEGATIVE"}
        except Exception as e:
            print(f"[BACKEND WARNING] DistilBERT loading failed ({e}). Returning baseline.")

    # Safe fallback output
    return {"positive_prob": 0.5, "label": "NEUTRAL"}


def predict_logistic_regression(text: str) -> dict:
    """Logistic Regression model placeholder / loader."""
    # Placeholder returning standard dict structure
    return {"positive_prob": 0.5, "label": "NEUTRAL"}


def predict_naive_bayes(text: str) -> dict:
    """Naive Bayes model placeholder / loader."""
    # Placeholder returning standard dict structure
    return {"positive_prob": 0.5, "label": "NEUTRAL"}


def predict_lstm(text: str) -> dict:
    """LSTM model placeholder / loader."""
    # Placeholder returning standard dict structure
    return {"positive_prob": 0.5, "label": "NEUTRAL"}
