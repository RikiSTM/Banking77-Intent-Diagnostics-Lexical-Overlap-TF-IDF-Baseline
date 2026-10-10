"""
SPEC diagnostics.py — THE INTERROGATION ROOM
Analyze WHERE and WHY the model fails using classification report and confusion matrix.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.layer1_tfidf.pipeline import build_pipeline
from src.layer1_tfidf.prep import prep_data


def run_diagnostics():
    print("=== LOADING DATA & TRAINING FINAL MODEL ===")
    X_tr, y_tr, X_te, y_te = prep_data()
    
    # Sanitize NaN (just in case some texts vanished during cleaning)
    df_tr = pd.DataFrame({'text': X_tr, 'label': y_tr}).dropna(subset=['text'])
    X_tr = df_tr['text'].values
    y_tr = df_tr['label'].values
    
    df_te = pd.DataFrame({'text': X_te, 'label': y_te}).dropna(subset=['text'])
    X_te = df_te['text'].values
    y_te = df_te['label'].values

    pipe = build_pipeline(clf="logreg")
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    
    classes = np.unique(np.concatenate([y_tr, y_te]))

    print("\n=== 1. PRECISION / RECALL PER CLASS (The 'Who is failing?' Report) ===")
    # Generate report as dict to filter only problematic classes (F1 < 0.70)
    report_dict = classification_report(
        y_te, y_pred, target_names=classes, output_dict=True, zero_division=0
    )
    
    failing_classes = []
    for intent, scores in report_dict.items():
        if isinstance(scores, dict) and scores['f1-score'] < 0.70 and intent not in ['accuracy', 'macro avg', 'weighted avg']:
            failing_classes.append((intent, scores['precision'], scores['recall'], scores['f1-score']))
            
    failing_classes.sort(key=lambda x: x[3]) # Sort by lowest F1-score
    
    if not failing_classes:
        print("🎉 Wow! All classes have an F1-score above 0.70. Model is highly robust!")
    else:
        print(f"⚠️ Found {len(failing_classes)} intents with F1-score < 0.70:")
        for intent, p, r, f1 in failing_classes[:10]: # Print top 10 worst
            print(f"  - {intent:<35} | Prec: {p:.2f} | Rec: {r:.2f} | F1: {f1:.2f}")

    print("\n=== 2. CONFUSION MATRIX (The 'Who are they confused with?' Report) ===")
    cm = confusion_matrix(y_te, y_pred, labels=classes)
    np.fill_diagonal(cm, 0) # Zero out the diagonal (correct predictions)
    
    # Extract top 10 highest error cells
    flat_indices = np.argsort(cm, axis=None)[::-1][:10]
    row_indices, col_indices = np.unravel_index(flat_indices, cm.shape)
    
    print("🔍 TOP 10 MISCLASSIFICATIONS (True Leaks):")
    for r, c in zip(row_indices, col_indices):
        actual_intent = classes[r]
        predicted_intent = classes[c]
        count = cm[r, c]
        if count > 0:
            print(f"  - Actual Intent: {actual_intent:<30} | Predicted As: {predicted_intent:<30} | ({count}x errors)")


if __name__ == "__main__":
    run_diagnostics()