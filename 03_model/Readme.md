# Toxic Comment Classification using TF-IDF and Logistic Regression

This project presents a **baseline machine learning model** for the **toxic-comment-classification**. It uses **TF-IDF vectorization** combined with **Logistic Regression** to detect different types of toxic comments in text.

---

## Problem Statement

The dataset includes user-generated comments, and the goal is to classify them into one or more of the following **6 toxic categories**:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

This is a **multi-label classification problem**, as a single comment may belong to multiple categories.

---

## Preprocessing Steps

- Convert text to lowercase.
- Remove punctuation, special characters, and numbers.
- Remove English stopwords.
- Apply basic lemmatization using `nltk`.

---

## Feature Extraction

- Uses `TfidfVectorizer` from `sklearn` to convert preprocessed text into numerical vectors.
- Limits features to a fixed number (e.g., `max_features=10000`) for efficiency and regularization.

---

## Model Building

- Trains one **Logistic Regression model per label** using a One-vs-Rest strategy (`sklearn`'s `OneVsRestClassifier`).
- The dataset is split into training and test sets (typically 80/20 split).

---

## Evaluation

- Classification performance is measured using:
  - `Precision`
  - `Recall`
  - `F1-Score`
  - `Accuracy`
- Evaluation is done per label using `classification_report` from `sklearn.metrics`.
- A **bar chart** is generated to visually compare F1-scores across the six categories.

---

## Results

- The model performs **well on common labels** like `toxic`, `obscene`, and `insult`.
- It performs **poorly on rare classes** like `threat` and `identity_hate`, revealing class imbalance.
- Bar plots highlight discrepancies in performance between frequent and rare classes.

---

## Final Remarks

- This notebook serves as a **solid baseline** before applying more advanced models like **BERT** or **DistilBERT**.
- It is **fast, interpretable, and easy to deploy**.
- Potential improvements:
  - Apply **class weighting** to handle imbalance.
  - Add **custom preprocessing** and **feature engineering**.
  - Perform **hyperparameter tuning**.

---

## Files

- `TF-IDF_logistic_regression_model.ipynb` – Main notebook for training and evaluating the model.
- `comments.csv` – (Assumed) source dataset for toxic comment classification.

---

## Dependencies

- Python 3.7+
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `nltk`

---

## 🔗 Acknowledgment

Based on the dataset from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge).
