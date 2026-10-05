"""
SPEC prep.py — SELF-CONTAINED
"""

import re
import string
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TEXT_COL = "clean_text"
LABEL_COL = "category"

_LEMMATIZER = WordNetLemmatizer()
_NEGATION = {"not", "no", "nor", "never", "without"}
_STOPWORDS = set(stopwords.words("english")) - _NEGATION

_CONTRACTIONS = {
    r"can't": "can not", r"won't": "will not", r"n't": " not ",
    r"'re": " are", r"'s": " is", r"'d": " would",
    r"'ll": " will", r"'ve": " have", r"'m": " am",
}


def _phase_universal(text: str) -> str:
    # Lower case formatting
    text = text.lower()
    
    # Remove white space
    text = re.sub(r"\s+", " ", text).strip()
    
    # Expand contracton
    for pat, repl in _CONTRACTIONS.items():
        text = re.sub(pat, repl, text)
    return text


def _phase_domain(text: str) -> str:
    text = re.sub(r"\ {4}\s?\d{4}", " cardnumber", text)
    text = re.sub(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", " cardnumber", text)
    text = re.sub(r"\b\d{8,17}\b", " accountnumber", text)
    text = re.sub(r"[$£€¥]\s?\d[\d,]*\.?\d*", " moneyamount", text)
    text = re.sub(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b", " date", text)
    return text


def _phase_noise(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"\b\d+\b", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _phase_nlp(text: str) -> str:
    return " ".join(
        _LEMMATIZER.lemmatize(t)
        for t in text.split()
        if t not in _STOPWORDS and len(t) > 1
    )


def clean_text(text) -> str:
    if not text or not isinstance(text, str):
        return ""
    return _phase_nlp(_phase_noise(_phase_domain(_phase_universal(text))))


# ── data prep ───────────────────────────────────────────────────
def load_raw(split: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / f"{split}.csv")


def build_processed(force: bool = False) -> None:
    for split in ("train", "test"):
        out = PROCESSED_DIR / f"{split}_clean.csv"
        if out.exists() and not force:
            continue
        df = load_raw(split)
        df[TEXT_COL] = df["text"].apply(clean_text)
        df[["text", TEXT_COL, LABEL_COL]].to_csv(out, index=False)


def prep_data():
    build_processed()
    train = pd.read_csv(PROCESSED_DIR / "train_clean.csv")
    test = pd.read_csv(PROCESSED_DIR / "test_clean.csv")
    return (
        train[TEXT_COL], train[LABEL_COL],
        test[TEXT_COL],  test[LABEL_COL],
    )


if __name__ == "__main__":
    # PARITY CHECK vs cell verification notebook
    samples = {
        "Please explain the exchange rates.": "please explain exchange rate",
        "I don't recognize a charge on my statement.": "not recognize charge statement",
        "I wasn't able to do a transfer to an account": "not able transfer account",
    }
    for raw, expected in samples.items():
        got = clean_text(raw)
        print(f"[{'OK ' if got == expected else 'DIFF'}] {raw!r} -> {got!r}")

    X_tr, y_tr, X_te, y_te = prep_data()
    print(f"train: {X_tr.shape} | intents: {y_tr.nunique()} | test: {X_te.shape}")