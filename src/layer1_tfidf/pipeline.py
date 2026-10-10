"""
SPEC pipeline.py — FACTORY ONLY
Build an unfitted sklearn Pipeline: TfidfVectorizer + classifier.
Training lives in eval.py (per-fold CV, anti-leakage).
"""
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import numpy as np

# Add stopwords filter to preserve important stopwords for bigram implementation 
base_stopwords = set(ENGLISH_STOP_WORDS)
NEGATION_KEEPERS = {"not", "no", "without", "never", "none", "nor"}
PHRASAL_KEEPERS = {"up", "down", "in", "out", "on", "off"} 

custom_stopwords = list(base_stopwords - NEGATION_KEEPERS - PHRASAL_KEEPERS)

# Defaults grounded in EDA decisions
TFIDF_PARAMS = {
    "ngram_range": (1, 2),   # bigrams break Tier-1 leak ties
    "max_df": 0.95,          # drop genre words (card, pending, transfer)
    "min_df": 2,             # drop hapax noise
    "stop_words" : custom_stopwords,
    "sublinear_tf": True,    # log-scale TF dominance
    "strip_accents": "unicode",
    "dtype": np.float32,
}

LOGREG_PARAMS = {
    "max_iter": 1000,
    "class_weight": "balanced",  # protect minority intents (75 vs 227)
    "C": 1.0
}


def _build_classifier(name: str):
    """Pick a classifier by name. Fail loud on unknown names."""
    name = name.lower()
    if name == "logreg":
        return LogisticRegression(**LOGREG_PARAMS)
    if name == "svc":
        return LinearSVC(max_iter=5000, class_weight="balanced", C=1.0)
    if name == "nb":
        return MultinomialNB(alpha=1.0)
    raise ValueError(f"Unknown classifier: {name!r}. Pick: logreg | svc | nb")


def build_pipeline(clf: str = "logreg") -> Pipeline:
    """Assemble vectorizer + classifier. NOT fitted."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf",   _build_classifier(clf)),
    ]) 
    
    
    
if __name__ == "__main__":
    print("=== PIPELINE FACTORY SMOKE TEST ===")
    
    # 1. Instantiate all 3 variants
    for name in ("logreg", "svc", "nb"):
        pipe = build_pipeline(clf=name)
        # Print pipeline structure: "tfidf:TfidfVectorizer → clf:LogisticRegression"
        steps = " → ".join(f"{n}:{type(t).__name__}" for n, t in pipe.steps)
        print(f"[{name:>6}] {steps}")

    # 2. Verify hyperparameters (sanity check)
    pipe_lr = build_pipeline("logreg")
    tfidf = pipe_lr.named_steps["tfidf"]
    clf = pipe_lr.named_steps["clf"]
    
    print(f"\n[CHECK] TF-IDF ngram_range : {tfidf.ngram_range}")
    print(f"[CHECK] TF-IDF max_df      : {tfidf.max_df}")
    print(f"[CHECK] TF-IDF sublinear_tf: {tfidf.sublinear_tf}")
    print(f"[CHECK] TF-IDF stopwords: {tfidf.stop_words}")
    print(f"[CHECK] LogReg C           : {clf.C}")
    print(f"[CHECK] LogReg class_weight: {clf.class_weight}")

    print("\n✅ Factory ready. Training lives in eval.py.")