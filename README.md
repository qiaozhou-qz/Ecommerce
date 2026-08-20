# Women's Clothing E-Commerce Reviews — Rating Prediction & Sentiment Analysis

A team project analyzing customer reviews of women's clothing to understand what drives ratings and to build predictive models for rating and return likelihood, using exploratory analysis, sentiment analysis, predictive modeling, and clustering in R.

## Business Question

Can customer review text and metadata be used to understand what drives product ratings, and can that text be used to predict ratings — with the goal of improving e-commerce marketing and sales strategy?

## Data

- **Source:** [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews) (Kaggle)
- **Size:** 19,662 cleaned reviews, 11 columns
- **Key fields:** Clothing ID, Age, Title, Review Text, Rating, Recommended IND, Positive Feedback Count, Division Name, Department Name, Class Name

## Methodology

Built in R using `tidytext`, `tm`, `caret`, `rpart`, and `cluster`. The analysis follows four stages:

**1. Exploratory Analysis**
- Reviewer demographics (age groups) and product/department popularity
- Correlation between review text characteristics (length, uppercase ratio, exclamation marks) and rating

**2. Sentiment Analysis**
- Bing lexicon (binary positive/negative classification)
- NRC sentiment polarity and emotion lexicon (8 emotion categories)
- AFINN lexicon (sentiment scoring, -5 to +5)
- Word clouds for overall and positive-vs-negative term frequency

**3. Predictive Modeling**
- Text features prepared via tokenization, stopword removal, stemming, and both TF and TF-IDF document-term matrices, for both `Review.Text` and `Title`
- CART (decision tree) and linear regression models trained to predict `Rating` from text features

**4. Clustering**
- Hierarchical and k-means clustering on non-text (numeric) features to segment reviews
- Cluster-then-predict approach: separate regression/tree models trained per cluster, then combined, compared against a single model on the full dataset

## Key Results

**Exploratory findings:**
- Reviewers aged 21–40 write the most reviews; "Dresses," "Knits," and "Blouses" are the most reviewed product classes; "Tops" is the most reviewed department
- Average rating: 4.18 (median: 5) — reviews skew strongly positive
- Review length, uppercase usage, and exclamation marks all show low correlation with rating; exclamation marks have a slightly higher relationship (r ≈ 0.18) than the others

**Sentiment findings:**
- ~80% of words used across all reviews are positive (Bing lexicon)
- "Positive" and "trust" are the most common emotion categories (NRC)
- As ratings increase, positive/joy word counts rise and negative/disgust word counts fall
- Mean AFINN sentiment score: 1.71 (median: 1.85), reviews skew positive overall

**Predictive modeling results (RMSE):**

| Feature | Method | RMSE |
|---|---|---|
| Review.Text | Regression | **0.901** (best) |
| Review.Text | CART | 1.010 |
| Title | Regression | 1.067 |
| Title | CART | 1.076 |

Review.Text consistently outperforms Title as a predictor, and TF and TF-IDF weighting produced identical RMSE in this analysis. A regression tree trained on return likelihood found that reviews containing the word "disappoint" had a 92% associated likelihood of the item being returned.

**Clustering + prediction results (SSE):**

| Approach | SSE |
|---|---|
| Single model, full dataset | 2,262.32 |
| Cluster-then-predict (regression) | 1,643.99 |
| Cluster-then-predict (tree) | 1,643.26 (best) |

Clustering before prediction meaningfully reduced error versus a single model on the full dataset, with the cluster + tree approach performing best overall.

## Limitations & Future Work

- Clustering was applied only to numeric fields; a full text-based clustering/dendrogram analysis (attempted for `Title`) was computationally limited and not completed for `Review.Text`
- Only CART and linear regression were tested; other models (logistic regression, random forest) were identified as a next step but not implemented
- Further exploratory work on review sentiment by product category was scoped as a future extension

## Tools

R, tidytext, tm, ggplot2, caret, rpart, cluster, wordcloud, qdap

## Team

Finance Group 3 — Columbia University: Harsh Dhanuka, Arik Shinkarevsky, Emmanuel Gyeng, Rafael Kazandjis, Umay Ayyub, Qiao Zhou, Harshitha Tummalapalli
