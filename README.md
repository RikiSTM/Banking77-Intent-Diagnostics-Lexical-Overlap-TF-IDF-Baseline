# Mini Project NLP #1: Banking-77 TF-IDF Baseline & Representation Audit

> *Dataset provided by PolyAI (Banking77) under CC BY 4.0 License.*

## 1. The Core Problem (Hook)
With 77 distinct banking intents, many user queries are nearly identical in vocabulary but entirely different in meaning (e.g., "card_arrival" vs "order_physical_card"). Where exactly does a purely lexical model go blind? This project establishes a strict TF-IDF + Linear baseline to document the precise failures of sparse representations before transitioning to deep learning.

## 2. Methodology
- **Data:** Banking77 (10,003 train / 3,080 test splits).
- **Phase 1: Lexical Leak Detection (EDA):** Pairwise Jaccard similarity across intent vocabularies with a **low-DF (high-IDF) spotlight**. This separates true discriminator leaks (shared rare words) from mere genre overlap (shared common words). 
  - *Actionable Output:* A tiered watchlist (`jaccard_queue_full.json`) identifying high-risk confusion pairs *before* modeling.
- **Phase 2: Pipeline:** `TfidfVectorizer(ngram_range=(1,2))` -> Linear Classifier. (Bigrams are specifically deployed to break ties in the high-risk pairs identified in Phase 1).
- **Models Evaluated:** `LogisticRegression` (balanced), `LinearSVC`, `MultinomialNB`.
- **Validation Discipline:** 5-Fold Stratified CV on the training set for tuning. The test set is evaluated **strictly once** at the very end to prevent data leakage.

## 3. Results (Baseline Metrics)
*Note: Evaluated strictly once on the test set after CV tuning.*

| Model | CV Macro-F1 (Mean ± Std) | Test Macro-F1 |
| :--- | :--- | :--- |
| Logistic Regression | `[TO BE MEASURED]` | `[TO BE MEASURED]` |
| Linear SVC | `[TO BE MEASURED]` | `[TO BE MEASURED]` |
| Multinomial NB | `[TO BE MEASURED]` | `[TO BE MEASURED]` |

## 4. Hypothesis vs. Reality
Before modeling, 5 "confused pairs" were hypothesized based on prefix clustering and lexical overlap. 
* `[HYPOTHESIS 1]` -> Result: `[MEASUREMENT]`
* `[HYPOTHESIS 2]` -> Result: `[MEASUREMENT]`
* `[HYPOTHESIS 3]` -> Result: `[MEASUREMENT]`
* `[HYPOTHESIS 4]` -> Result: `[MEASUREMENT]`
* `[HYPOTHESIS 5]` -> Result: `[MEASUREMENT]`

## 5. Failure Gallery & Error Taxonomy
Manual audit of 20+ misclassifications revealed 4 distinct representation failures in the TF-IDF approach:
1. **Synonym Blindness:** `[Example to be added from errors.csv]`
2. **Word Order Ignorance:** `[Example to be added from errors.csv]`
3. **Zero Semantic Overlap:** `[Example to be added from errors.csv]`
4. **Ambiguity / Potential Label Noise:** `[Example to be added from errors.csv]`

## 6. Representation Blindness: OOV & Lexical Retrieval
**OOV (Out-Of-Vocabulary) Audit:**
`[X]%` of tokens in the test set were completely ignored because they did not exist in the training vocabulary. *Insight: Unseen words are dropped silently—the vectorizer doesn't error out, it just goes blind.*

**Lexical Retrieval Demo (Proto-RAG):**
*Finding similar words ≠ finding similar meaning.*

| Query | Nearest Neighbor (Cosine Sim) | Status |
| :--- | :--- | :--- |
| `[Query 1]` | `[Neighbor 1]` | ✅ Good lexical match |
| `[Query 2]` | `[Neighbor 2]` | ❌ Zero semantic overlap |

## 7. To Be Measured (The Bridge to W4)
This project serves as a foundational test bed for contextual architectures.
- **TF-IDF = sparse lexical embedding v0.** (Week 4 will introduce dense & contextual embeddings).
- **Classifier head here = linear + softmax = BERT head architecture.**
- **Multinomial softmax in LR = softmax in LLM output.** The math is identical; the difference is purely a distribution over classes vs. a distribution over tokens.

## 8. Limitations
This is a mini-project built for baseline evaluation. No deep hyperparameter grid searches were conducted, and metrics apply strictly to this linear Bag-of-Words setup.
