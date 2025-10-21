# Cyberbullying Detection

Detecting cyberbullying from short social text using classic ML and simple neural models, with robust preprocessing and adversarial stress-tests.

## 🔍 Overview

* **Goal:** Binary classify messages as **Aggressive** vs **Non-Aggressive**.
* **Use cases:** Content moderation, campus/community monitoring, early warning.
* **What’s special:** Multiple text representations (TF-IDF, Word2Vec, log-count weighting), side-by-side model baselines, and **adversarial checks** on sensitive keywords.

## 📦 Dataset

* **Source:** Two CSV files: `Aggressive_All.csv` and `Non_Aggressive_All.csv`.
* **Size:** ≈118k rows each.
* **Columns:** `No.`, `Message`.
* **Notes:** The “Aggressive” file mixes explicit insults, biased/hostile opinions, and noise; the “Non-Aggressive” file contains normal statements but also many ambiguous tokens.

> Please add the actual download link and licensing terms here if you publish the data.

## 🧹 Preprocessing & Features

* Cleaning: lowercasing, punctuation & digit removal, stop-word filtering, stemming/tokenization.
* Representations:

  1. **TF-IDF** (optionally **PCA** to reduce dimension) → scaling.
  2. **Word2Vec** average word vectors → scaling.
  3. **Naïve Bayes log-count weighting** (log-count ratio per class).
* Split: **80/20** train/test with cross-validation; compare with/without standardization and information gain.

## 🤖 Models

* **Classical:** SVM, Random Forest, Logistic Regression (mainly on TF-IDF).
* **Neural (simple):** MLP for vector features (e.g., 128-Dropout-128-Dropout-Softmax).
* **Experimental:** Deep feedforward (128→64→1, ReLU/Sigmoid).
* **Error handling:** Adversarial tests with sensitive words (e.g., “fuck / shit / sucks”) embedded in neutral/positive context to reduce keyword-triggered false positives.

> If you later use BERT/Transformers or other embeddings, document it here.

## 📈 Evaluation

* Metrics: Accuracy, Precision, Recall, F1; Confusion Matrix.
* **Subset testing** on sensitive-keyword samples to check over-triggering.
* (Optional) AUC and PR curves.

## 🧪 Adversarial Checks (Sensitive Words)

* Build neutral/positive sentences that contain sensitive tokens and verify the model doesn’t blindly flag them.
* If it still over-fires:

  * Add such samples to training (data balancing/weighting).
  * Switch to context-aware encoders (e.g., BERT) or use class-/sample-weighted loss.

## 🔭 Future Work

* Transformer encoders with attention.
* Hierarchical/typed labels for bullying categories.
* Hard-example mining and data augmentation (EDA, back-translation).

If you share your actual script names/args and metric tables, I’ll tailor this to your repo exactly.
