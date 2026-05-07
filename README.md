# 🎬 Enhancing Movie Recommendation Systems Using NLP: Integrating Sentiment Analysis for Improved Predictions

> **Master's Thesis · IU International University of Applied Sciences · Data Science · 2025**
> **Hybrid NLP + sentiment-aware movie recommender that outperforms all traditional baselines on the IMDb dataset.**

---

## Overview

This research builds and compares a full suite of movie recommendation approaches, from classic demographic and collaborative filtering to BERT-based hybrid models which shows that injecting **sentiment analysis from user reviews** meaningfully improves recommendation quality.

The system tackles two real-world problems:
- **Recommendation quality** — hybrid model (content + collaborative + sentiment) beats individual methods on RMSE & MAE
- **Cold start problem** — DistilBERT multi-label genre classifier predicts genres from descriptions alone (macro F1: 0.55)

---

## Repository Structure

| File | Description |
|------|-------------|
| `demographic filtering recomm...` | Popularity/rating-weighted demographic baseline |
| `collaborative filtering recomm...` | User-based & item-based CF with cosine/Pearson similarity + SVD |
| `content based recommendation ...` | TF-IDF and BERT embedding-based content filtering |
| `cold start prob predicting genre.i...` | DistilBERT multi-label genre classifier for cold-start handling |
| `compare recommendation with a...` | Side-by-side comparison of all methods |

---

## Methods Implemented

### Recommendation Systems
| Approach | Technique |
|----------|-----------|
| Demographic Filtering | Weighted rating formula (IMDB-style) |
| Collaborative Filtering | User-based & Item-based (cosine, Pearson), SVD |
| Content-Based Filtering | TF-IDF vectors, BERT contextual embeddings |
| **Hybrid Model** | CF + CBF signals + sentiment score injection |

### Sentiment Analysis (for review-enhanced recommendations)
| Model | Accuracy | Notes |
|-------|----------|-------|
| VADER | 59% | Lexicon-based; good on positive reviews (F1=0.71) |
| TF-IDF + SMOTE | 69% | Best traditional; balanced classes |
| SVM | 69% | Best classical ML model |
| **DistilBERT (oversampling)** | **76%** | Best overall; F1=0.77 (neg), 0.75 (pos) |

### Cold Start — Genre Prediction
| Model | Macro F1 | Micro F1 |
|-------|----------|----------|
| DistilBERT multi-label classifier | 0.55 | 0.54 |
| Best genres: Horror (0.81), Drama (0.78), Adventure (0.75) | | |

### Hybrid Model (Final System)
- **Lowest RMSE: 0.84, MAE: 0.40** (model-based CF)
- Sentiment integration further reduced RMSE to 0.99, MAE to 0.69 vs. mean-centred baseline
- Precision: 0.60 | Recall: 1.00 | F1: 0.75 on recommendation set

---

## 📊 Dataset

**IMDb Dataset** — movie metadata + user-generated reviews
- Preprocessing: text cleaning, tokenisation, SMOTE for class balancing
- Evaluation metrics: Accuracy, F1-score (macro/micro/weighted), RMSE, MAE

---

## 🔑 Key Findings

1. **Hybrid > traditional**: The hybrid recommendation system outperforms all individual approaches across RMSE and MAE.
2. **Sentiment matters**: Adding DistilBERT-based sentiment scores to the hybrid model meaningfully shifts recommendations vs. the sentiment-agnostic baseline.
3. **Cold start solvable**: Genre prediction from descriptions alone achieves fair F1 (0.55 macro), with strong performance on major genres.
4. **DistilBERT best for sentiment**: 76% accuracy with balanced class F1s, compared to 69% for the best classical model (SVM).

---

## 🛠️ Tech Stack

`Python` · `scikit-learn` · `HuggingFace Transformers` · `DistilBERT` · `BERT` · `TF-IDF` · `VADER` · `SVD` · `SMOTE` · `pandas` · `matplotlib`

---

## 👩‍💻 Author

**Hadiya Hasan**  
M.Sc. Data Science — IU International University of Applied Sciences  
Supervisor: Prof. Dr. Thorsten Fröhlich  
Submission: June 2025

---

## 📄 Citation

```
Hasan, H. (2025). Enhancing Movie Recommendation Systems Using NLP: 
Integrating Sentiment Analysis for Improved Predictions. 
Master's Thesis, IU International University of Applied Sciences.
```
