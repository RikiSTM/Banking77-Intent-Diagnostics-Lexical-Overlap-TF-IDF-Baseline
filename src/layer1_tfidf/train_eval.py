"""
SPEC train_eval.py — STEP 1: Setup & Load Data
"""
# 1. Fix sklearn paths
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd

# 2. Fix local module paths (absolute from project root)
from src.layer1_tfidf.pipeline import build_pipeline
from src.layer1_tfidf.prep import prep_data


"""
SPEC train_eval.py — STEP 2: Cross-Validation Loop
"""

def evaluate_cv(X, y, n_splits: int = 5, clf: str = "logreg") -> list:
    """Run Stratified K-Fold CV and return list of F1-macro per fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        # 1. Potong data buat fold ini
        X_tr_fold, X_val_fold = X[train_idx], X[val_idx]
        y_tr_fold, y_val_fold = y[train_idx], y[val_idx]

        # 2. Rakit mesin BARU (fresh instance, anti-bocor antar fold)
        pipe = build_pipeline(clf=clf)

        # 3. Latih mesinnya pakai data fold ini
        pipe.fit(X_tr_fold, y_tr_fold)

        # 4. Tes tebakan pakai data validation fold ini
        y_pred = pipe.predict(X_val_fold)
        
        # 5. Hitung skor (Macro F1 biar adil ke kelas minoritas)
        score = f1_score(y_val_fold, y_pred, average="macro")
        fold_scores.append(score)
        print(f"  [Fold {fold}/{n_splits}] Macro-F1: {score:.4f}")

    return fold_scores

if __name__ == "__main__":
    print("=== STEP 1: LOADING DATA ===")
    X_tr, y_tr, X_te, y_te = prep_data()
    
    X_tr_s, y_tr_s = pd.Series(X_tr), pd.Series(y_tr)
    mask_tr = X_tr_s.notna()
    X_tr, y_tr = X_tr_s[mask_tr].values, y_tr_s[mask_tr].values
    
    X_te_s, y_te_s = pd.Series(X_te), pd.Series(y_te)
    mask_te = X_te_s.notna()
    X_te, y_te = X_te_s[mask_te].values, y_te_s[mask_te].values
    
    print("\n=== STEP 2: 5-FOLD CROSS-VALIDATION ===")
    scores = evaluate_cv(X_tr, y_tr)
   
    mean_f1 = np.mean(scores)
    std_f1 = np.std(scores)
    print(f"\n🎯 CV Macro-F1 Mean: {mean_f1:.4f} (± {std_f1:.4f})")
    
    print("\n=== STEP 3: FINAL TRAIN & TEST EVALUATION ===")
    
    # 1. Final Fit: Learn from 100% of training data
    final_pipe = build_pipeline(clf="logreg")
    final_pipe.fit(X_tr, y_tr)

    # 2. Predict on the sealed Test Set (Ujian Asli)
    y_pred_test = final_pipe.predict(X_te)

    # 3. Calculate final metrics (need accuracy_score imported at the top!)
    
    test_f1 = f1_score(y_te, y_pred_test, average="macro")
    test_acc = accuracy_score(y_te, y_pred_test)

    print(f"🏆 Final Test Macro-F1 : {test_f1:.4f}")
    print(f"🏆 Final Test Accuracy: {test_acc:.4f}")