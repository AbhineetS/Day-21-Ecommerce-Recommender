# 🛒 Day 21 — E-commerce Product Recommendation System

This project builds a **practical recommendation engine** for e-commerce platforms using  
**Item-Based Collaborative Filtering**, **Content-Based Filtering (TF-IDF)**, and a **Hybrid Model**.

It mimics real-world systems used by Amazon, Flipkart, and Shopify stores to boost conversions and personalize product discovery.

---

## 🚀 Overview

- Generates or loads **product + user interaction datasets**
- Builds a **User–Item rating matrix**
- Computes **item-to-item similarity** with cosine distance
- Generates **content embeddings** using TF-IDF on product descriptions
- Produces **3 recommendation types**:
  - 📌 *Item-based CF*
  - 📌 *Content-based (description similarity)*
  - 📌 *Hybrid score (CF + Content)*
- Evaluates with **Leave-One-Out Precision@10**
- Saves results in:  
  - `cf_recommendations.txt`  
  - `content_recommendations.txt`  
  - `hybrid_recommendations.txt`  

---

## 🧠 Workflow

1. **Data Loading / Generation**  
   Creates synthetic product catalog + user ratings if no dataset exists.

2. **Item-Based Collaborative Filtering**  
   Computes cosine similarity between product vectors to find similar items.

3. **Content-Based Filtering**  
   TF-IDF on product descriptions → cosine similarity → nearest neighbors.

4. **Hybrid Recommender**  
   Combines CF + Content with adjustable weight `alpha`.

5. **Evaluation (Precision@K)**  
   Quick leave-one-out evaluation to measure recommendation quality.

---

## 📊 Example Output (Demo)

### ⭐ Item-CF Recommendations (User 1)
- Electronics Product 23  
- Home Product 7  
- Sports Product 52  
*(10 items total)*  

### ⭐ Content-Based Similar Items (Seed Product)
- Books Product 44  
- Books Product 81  
- Books Product 102  

### ⭐ Hybrid Recommendations
- Electronics Product 19  
- Sports Product 59  
- Beauty Product 3  

---

## 🧩 Tech Stack

Python | Pandas | NumPy | Scikit-learn | TF-IDF Vectorizer | Cosine Similarity

---

## ▶️ Running the Project

```bash
source ../Day-01-Titanic/venv/bin/activate
pip install -r requirements.txt
python3 run_recommender.py
```

---

## 🔗 Connect

LinkedIn: https://www.linkedin.com/in/abhineet-s  
GitHub: https://github.com/AbhineetS/Day-21-Ecommerce-Recommender