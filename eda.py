import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

os.makedirs("notebooks/charts", exist_ok=True)

df = pd.read_csv("data/electronics_features.csv")

# Chart 1: Mismatch score distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['mismatch_score'], bins=50, color='#534AB7', alpha=0.8, edgecolor='white')
ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero mismatch')
ax.set_title('Rating vs Sentiment Mismatch Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Mismatch Score (positive = overrated)', fontsize=12)
ax.set_ylabel('Number of Reviews', fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig('notebooks/charts/mismatch_distribution.png', dpi=150)
plt.close()
print("Saved chart 1: mismatch_distribution.png")

# Chart 2: Average mismatch by rating
fig, ax = plt.subplots(figsize=(8, 5))
avg_mismatch = df.groupby('rating')['mismatch_score'].mean()
bars = ax.bar(avg_mismatch.index, avg_mismatch.values, color='#534AB7', alpha=0.8, edgecolor='white')
ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax.set_title('Average Mismatch Score by Star Rating', fontsize=14, fontweight='bold')
ax.set_xlabel('Star Rating', fontsize=12)
ax.set_ylabel('Avg Mismatch Score', fontsize=12)
for bar, val in zip(bars, avg_mismatch.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('notebooks/charts/mismatch_by_rating.png', dpi=150)
plt.close()
print("Saved chart 2: mismatch_by_rating.png")

# Chart 3: The headline finding
# Average sentiment score vs average rating — normalized
fig, ax = plt.subplots(figsize=(10, 5))
grouped = df.groupby('rating').agg(
    avg_sentiment=('sentiment_score', 'mean'),
    avg_normalized_rating=('rating_normalized', 'mean')
).reset_index()
ax.plot(grouped['rating'], grouped['avg_sentiment'], 
        marker='o', color='#534AB7', linewidth=2.5, label='Actual sentiment (VADER)', markersize=8)
ax.plot(grouped['rating'], grouped['avg_normalized_rating'], 
        marker='s', color='#D85A30', linewidth=2.5, linestyle='--', label='Normalized star rating', markersize=8)
ax.set_title('What Customers Write vs What They Rate', fontsize=14, fontweight='bold')
ax.set_xlabel('Star Rating', fontsize=12)
ax.set_ylabel('Score (-1 to +1)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/charts/sentiment_vs_rating.png', dpi=150)
plt.close()
print("Saved chart 3: sentiment_vs_rating.png")

print("\nAll charts saved to notebooks/charts/")
print("\nKey finding:")
print(df.groupby('rating')['mismatch_score'].mean().round(3))