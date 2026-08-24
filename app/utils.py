# app/utils.py

import re
from typing import Set

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)

# Preserve negation and contrast words from stopword stripping
NLTK_STOP_WORDS: Set[str] = set(stopwords.words("english"))
NEGATION_AND_CONTRAST_WORDS: Set[str] = {
    "no",
    "not",
    "nor",
    "neither",
    "never",
    "none",
    "cannot",
    "can't",
    "couldn't",
    "didn't",
    "doesn't",
    "don't",
    "hadn't",
    "hasn't",
    "haven't",
    "isn't",
    "mightn't",
    "mustn't",
    "needn't",
    "shan't",
    "shouldn't",
    "wasn't",
    "weren't",
    "won't",
    "wouldn't",
    "but",
    "however",
    "yet",
    "except",
    "although",
}

# Final stopword list preserving contextual tokens
CUSTOM_STOP_WORDS: Set[str] = NLTK_STOP_WORDS - NEGATION_AND_CONTRAST_WORDS
lemmatizer = WordNetLemmatizer()


# Text cleaning function tailored for ML models
def clean_text(text: str) -> str:
    # Lowercase
    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation except apostrophes (preserves contractions like didn't)
    text = re.sub(r"[^a-z'\s]", "", text)

    # Tokenize
    words = text.split()

    # Remove non-negation stopwords and apply lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in CUSTOM_STOP_WORDS
    ]

    return " ".join(words)


# Rule-based probability calibration for edge cases
def calibrate_probability(review: str, original_prob: float) -> float:
    review_lower = str(review).lower()

    # Regex rule patterns
    double_negation = (
        r"\b(didn\'t|did not|wasn\'t|was not|not|never)\s+\w*\s*"
        r"(hate|terrible|bad|awful|horrible|dislike)\b"
    )
    negated_positive = (
        r"\b(didn\'t|did not|wasn\'t|was not|not|never)\s+\w*\s*"
        r"(like|love|good|great|enjoy|fantastic|amazing)\b"
    )
    mixed_positive = (
        r"\b(slow|flawed|boring|okay|average)\b.*\b(but|however|yet)\b.*\b"
        r"(loved|great|masterpiece|enjoyed|fantastic)\b"
    )
    mixed_negative = (
        r"\b(great|good|beautiful|nice)\b.*\b(but|however|except|yet)\b.*\b"
        r"(boring|terrible|garbage|waste|awful|horrible)\b"
    )

    prob = original_prob

    # Apply shift rules
    if re.search(double_negation, review_lower):
        prob = float(np.clip(prob + 0.30, 0.52, 0.70))
    elif re.search(negated_positive, review_lower):
        prob = float(np.clip(prob - 0.35, 0.15, 0.35))

    if re.search(mixed_positive, review_lower):
        prob = float(np.clip(prob + 0.20, 0.55, 0.68))
    elif re.search(mixed_negative, review_lower):
        prob = float(np.clip(prob - 0.20, 0.32, 0.45))

    return prob


# Derive label based on calibrated probability score
def derive_sentiment_label(prob: float) -> str:
    if prob >= 0.60:
        return "Positive"
    elif 0.42 <= prob < 0.60:
        return "Mixed / Neutral"
    return "Negative"
