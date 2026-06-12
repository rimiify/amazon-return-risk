import pandas as pd

def load_electronics_reviews():
    df = pd.read_csv("data/1429_1.csv", low_memory=False)
    
    keep_cols = [
        'name',
        'brand',
        'categories',
        'reviews.rating',
        'reviews.text',
        'reviews.title',
        'reviews.numHelpful',
        'reviews.doRecommend',
    ]
    
    df = df[keep_cols].copy()
    df.columns = [
        'product_name',
        'brand',
        'category',
        'rating',
        'review_text',
        'review_title',
        'helpful_votes',
        'do_recommend',
    ]
    
    df = df.dropna(subset=['review_text', 'rating'])
    df['rating'] = df['rating'].astype(float)
    df['helpful_votes'] = df['helpful_votes'].fillna(0).astype(int)
    df['do_recommend'] = df['do_recommend'].fillna(True)
    
    print(f"Clean shape: {df.shape}")
    print(f"\nRating distribution:\n{df['rating'].value_counts().sort_index()}")
    print(f"\nSample row:\n{df.iloc[0]}")
    
    df.to_csv("data/electronics_clean.csv", index=False)
    print("\nSaved to data/electronics_clean.csv")
    
    return df

if __name__ == "__main__":
    df = load_electronics_reviews()