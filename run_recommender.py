#!/usr/bin/env python3
"""
run_recommender.py
Day 21 — E-commerce Product Recommendation (Item-CF + Content-based + Hybrid)
Self-contained demo: generates synthetic data if none present.
Outputs:
 - prints recommendations for a sample user
 - saves cf_recommendations.txt, content_recommendations.txt, hybrid_recommendations.txt
 - saves simple evaluation: precision@K (leave-one-out)
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def generate_demo_data(path_products="products_demo.csv", path_interactions="interactions_demo.csv"):
    """Create a small demo products + interactions dataset and save to disk."""
    print("📦 No dataset found — generating demo product & interaction CSVs...")
    n_products = 120
    n_users = 150
    categories = [
        "Electronics", "Home", "Sports", "Books", "Beauty", "Toys", "Clothing", "Grocery"
    ]

    # Products: id, title, category, description
    products = []
    for i in range(1, n_products + 1):
        cat = random.choice(categories)
        title = f"{cat} Product {i}"
        # simple description (repeat category-related tokens)
        desc = (
            f"{cat} premium product. "
            f"Best for {cat.lower()} lovers. "
            f"Quality {cat.lower()}, durable and value."
        )
        products.append((i, title, cat, desc))
    df_products = pd.DataFrame(products, columns=["productId", "title", "category", "description"])
    df_products.to_csv(path_products, index=False)

    # Interactions: userId, productId, rating (1-5) - generate implicit-kind behaviour with some noise
    interactions = []
    for user in range(1, n_users + 1):
        n_actions = random.randint(6, 25)
        liked_categories = random.sample(categories, k=random.randint(1, 3))
        candidate_products = df_products[df_products["category"].isin(liked_categories)]
        chosen = candidate_products.sample(n=min(n_actions, candidate_products.shape[0]), replace=False)
        for _, row in chosen.iterrows():
            # rating biased for liked category
            rating = min(5, max(1, int(np.round(np.random.normal(loc=4.0, scale=0.9)))))
            interactions.append((user, int(row.productId), rating))
    df_inter = pd.DataFrame(interactions, columns=["userId", "productId", "rating"])
    df_inter.to_csv(path_interactions, index=False)

    print(f"✅ Demo products saved to {path_products}")
    print(f"✅ Demo interactions saved to {path_interactions}")
    return df_products, df_inter


def load_or_generate(products_fp="products_demo.csv", interactions_fp="interactions_demo.csv"):
    p_prod = Path(products_fp)
    p_inter = Path(interactions_fp)
    if p_prod.exists() and p_inter.exists():
        print("📥 Loading existing datasets...")
        df_products = pd.read_csv(p_prod)
        df_inter = pd.read_csv(p_inter)
    else:
        df_products, df_inter = generate_demo_data(products_fp, interactions_fp)
    return df_products, df_inter


def build_user_item_matrix(df_inter):
    """Pivot interactions into user-item matrix (users x items). Missing -> 0"""
    pivot = df_inter.pivot_table(index="userId", columns="productId", values="rating", fill_value=0)
    return pivot


def item_based_cf(pivot, top_k=10):
    """Compute item-item cosine similarity and return similarity matrix (productId order = pivot.columns)."""
    print("\n🔧 Building item-item collaborative filter (cosine)...")
    # items as columns; we compute similarity between column vectors
    items = pivot.columns
    item_vectors = pivot.T.values  # shape: (n_items, n_users)
    sim = cosine_similarity(item_vectors)  # (n_items, n_items)
    sim_df = pd.DataFrame(sim, index=items, columns=items)
    return sim_df


def content_based_embeddings(df_products):
    """TF-IDF on product descriptions -> content similarity (productId order)."""
    print("\n🧩 Building content-based TF-IDF (descriptions)...")
    tfidf = TfidfVectorizer(max_features=1000, stop_words="english")
    descs = df_products["description"].fillna("").astype(str).values
    X = tfidf.fit_transform(descs)  # shape: (n_products, n_features)
    content_sim = cosine_similarity(X)  # (n_products, n_products)
    prod_ids = df_products["productId"].astype(int).values
    sim_df = pd.DataFrame(content_sim, index=prod_ids, columns=prod_ids)
    return sim_df, tfidf


def recommend_for_user_itemcf(user_id, pivot, item_sim_df, top_k=10):
    """Generate item-CF recommendations for a user using weighted sum of item similarities."""
    if user_id not in pivot.index:
        print(f"⚠️ user {user_id} not found in pivot — returning empty list.")
        return []

    user_ratings = pivot.loc[user_id]
    rated_mask = user_ratings > 0
    unrated_items = user_ratings[~rated_mask].index.tolist()
    scores = {}
    for item in unrated_items:
        # compute weighted sum of similarities with items the user rated
        similar_items = item_sim_df[item]
        # only consider items the user rated
        weighted = (similar_items[rated_mask.index] * user_ratings).sum()
        norm = similar_items[rated_mask.index].sum() + 1e-9
        scores[item] = weighted / norm

    # top-K by score
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [pid for pid, _ in top]


def recommend_for_itemseed_content(seed_product_id, content_sim_df, df_products, top_k=10):
    """Recommend top similar products by content for a seed product id."""
    if seed_product_id not in content_sim_df.index:
        return []
    sims = content_sim_df[seed_product_id].drop(index=seed_product_id).sort_values(ascending=False)
    chosen = sims.head(top_k).index.tolist()
    return df_products.set_index("productId").loc[chosen]["title"].tolist(), chosen


def hybrid_recommend(user_id, pivot, item_sim_df, content_sim_df, alpha=0.6, top_k=10):
    """
    Hybrid: combines normalized item-CF score and average content similarity score.
    alpha weights item-CF (0..1).
    """
    if user_id not in pivot.index:
        return []

    user_ratings = pivot.loc[user_id]
    rated_mask = user_ratings > 0
    unrated_items = user_ratings[~rated_mask].index.tolist()

    # Precompute content-based "popularity-like" scores: mean content similarity to user's rated items
    content_scores = {}
    for item in unrated_items:
        # average content similarity between item and each item user rated
        sims = content_sim_df.loc[item, rated_mask.index]
        # multiply by user rating to bias towards highly rated seeds
        weighted = (sims * user_ratings).sum()
        norm = sims.sum() + 1e-9
        content_scores[item] = weighted / norm

    # item-cf scores
    cf_scores = {}
    for item in unrated_items:
        sims = item_sim_df[item]
        weighted = (sims * user_ratings).sum()
        norm = sims.sum() + 1e-9
        cf_scores[item] = weighted / norm

    # normalize both score types to [0,1]
    cf_vals = np.array(list(cf_scores.values()))
    cont_vals = np.array(list(content_scores.values()))
    if cf_vals.max() - cf_vals.min() > 1e-9:
        cf_norm = (cf_vals - cf_vals.min()) / (cf_vals.max() - cf_vals.min())
    else:
        cf_norm = cf_vals * 0.0
    if cont_vals.max() - cont_vals.min() > 1e-9:
        cont_norm = (cont_vals - cont_vals.min()) / (cont_vals.max() - cont_vals.min())
    else:
        cont_norm = cont_vals * 0.0

    items = list(cf_scores.keys())
    final_scores = {}
    for idx, item in enumerate(items):
        final_scores[item] = alpha * cf_norm[idx] + (1 - alpha) * cont_norm[idx]

    top = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [pid for pid, _ in top]


def leave_one_out_precision_at_k(pivot, item_sim_df, k=10, n_users=100):
    """Simple leave-one-out evaluation: for each sampled user, hold out one positive item and test if recommender recovers it."""
    users = pivot.index.tolist()
    sampled_users = random.sample(users, min(n_users, len(users)))
    hits = 0
    total = 0
    for u in sampled_users:
        user_ratings = pivot.loc[u]
        pos_items = user_ratings[user_ratings > 0].index.tolist()
        if len(pos_items) < 2:
            continue
        hold = random.choice(pos_items)
        # create a training user vector with holdout zeroed
        train_vec = user_ratings.copy()
        train_vec[hold] = 0
        # build simple score for unrated items using item similarity (recompute on-the-fly using provided sim)
        rated_mask = train_vec > 0
        unrated = train_vec[train_vec == 0].index.tolist()
        scores = {}
        for item in unrated:
            sims = item_sim_df[item]
            weighted = (sims[rated_mask.index] * train_vec).sum()
            norm = sims[rated_mask.index].sum() + 1e-9
            scores[item] = weighted / norm
        topk = [pid for pid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]
        if hold in topk:
            hits += 1
        total += 1
    precision_at_k = hits / total if total > 0 else 0.0
    return precision_at_k, total


def main():
    df_products, df_inter = load_or_generate("products_demo.csv", "interactions_demo.csv")

    # basic stats
    print(f"\n📊 Products: {df_products.shape} | Interactions: {df_inter.shape}")

    # pivot
    pivot = build_user_item_matrix(df_inter)
    print(f"User-item matrix: {pivot.shape}")

    # item-based CF
    item_sim_df = item_based_cf(pivot)

    # content-based
    content_sim_df, tfidf = content_based_embeddings(df_products)

    # pick a sample user to show recommendations
    sample_user = pivot.index[0]  # first user
    print(f"\nSample user id: {sample_user}")

    cf_recs = recommend_for_user_itemcf(sample_user, pivot, item_sim_df, top_k=10)
    cf_titles = df_products.set_index("productId").loc[cf_recs]["title"].tolist() if cf_recs else []
    print("\nItem-based CF recommendations (titles):")
    for t in cf_titles:
        print(" -", t)

    # content-based: pick a seed product (random)
    seed_pid = int(df_products["productId"].sample(1).iloc[0])
    print(f"\nSeed product: {seed_pid} — {df_products.set_index('productId').loc[seed_pid, 'title']}")
    content_titles, content_ids = recommend_for_itemseed_content(seed_pid, content_sim_df, df_products, top_k=10)
    print("Content-based (by description/genre) recs (titles):")
    for t in content_titles:
        print(" -", t)

    # hybrid
    hybrid_recs = hybrid_recommend(sample_user, pivot, item_sim_df, content_sim_df, alpha=0.65, top_k=10)
    hybrid_titles = df_products.set_index("productId").loc[hybrid_recs]["title"].tolist() if hybrid_recs else []
    print("\nHybrid recommendations (titles):")
    for t in hybrid_titles:
        print(" -", t)

    # save outputs
    Path("cf_recommendations.txt").write_text("\n".join(cf_titles))
    Path("content_recommendations.txt").write_text("\n".join(content_titles))
    Path("hybrid_recommendations.txt").write_text("\n".join(hybrid_titles))
    print("\n✅ Saved recommendation text files: cf_recommendations.txt, content_recommendations.txt, hybrid_recommendations.txt")

    # quick evaluation (leave-one-out precision@K)
    print("\n🔎 Running quick leave-one-out precision@K evaluation on item-CF (demo)...")
    prec, total = leave_one_out_precision_at_k(pivot, item_sim_df, k=10, n_users=100)
    print(f"Precision@10 (sampled users): {prec:.4f} (evaluated on {total} users)")

    print("\n🎯 Done. Tweak alpha, TF-IDF params, or use real datasets for better results.")


if __name__ == "__main__":
    main()