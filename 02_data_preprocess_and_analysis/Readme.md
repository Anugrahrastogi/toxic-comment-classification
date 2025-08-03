# Data Preprocessing and Exploratory Analysis for Toxic Comment Classification

This notebook performs data cleaning and exploratory data analysis (EDA) for the **Toxic Comment Classification**. It prepares the dataset for downstream machine learning and deep learning models.

---

## Objective

To analyze, clean, and understand the structure and imbalance of toxic comment labels to build a solid foundation for modeling.

---

## Dataset Overview

Each comment in the dataset is labeled with one or more of the following categories:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

---

## Key Components

### 1. Data Loading
- Loads `comments.csv`, which includes `comment_text` and label columns.

### 2. Labeling Clean Comments
- Adds a `clean` column for comments with no toxic tags.
- Computes:
  - Total number of comments.
  - Number of clean comments.
  - Total number of toxic tags.

### 3. Class Imbalance Analysis
- Shows that ~90% of comments are **non-toxic**.
- Highlights the challenge of **class imbalance** in multi-label classification.

### 4. Co-occurrence of Toxic Tags
- Correlation heatmaps (Pearson) are used to explore how toxic labels co-occur.
- Discusses limitations of Pearson correlation with binary data.
- Suggests using **Cramér’s V** or **confusion matrices** for deeper insights.

### 5. Visualizations
- **Bar plots** showing the frequency of each toxic label.
- **Word Clouds**:
  - One for **clean comments**.
  - One for **toxic comments**.

---

## Libraries Used

- `pandas`, `numpy` — Data manipulation
- `matplotlib`, `seaborn` — Plotting and visualizations
- `nltk` — Tokenization, stopword removal, lemmatization
- `wordcloud` — Word cloud generation

---

## Outcomes

- Provides clear visual insights into the **label distribution**, **data imbalance**, and **textual patterns**.
- Confirms that classes like `threat` and `identity_hate` are rare, posing modeling challenges.
- Lays a solid foundation for:
  - Feature extraction using TF-IDF
  - Modeling using Logistic Regression or Transformers (e.g., BERT)

---

## Final Remarks

- This notebook is a **critical first step** in the toxic comment classification pipeline.
- It helps identify preprocessing needs and modeling strategies.
- Recommended improvements:
  - Address class imbalance (e.g., SMOTE, class weights).
  - Incorporate advanced preprocessing techniques.
