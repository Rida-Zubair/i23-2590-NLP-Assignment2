import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Configure matplotlib for better Urdu text support
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)

# Sample Urdu words and their TF-IDF scores
words = ["کے", "میں", "ہے", "اور", "سے", "کہ"]
scores = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55]

# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(words, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])

# Customize the plot
plt.title("Top TF-IDF Words (Urdu)", fontsize=16, fontweight='bold')
plt.xlabel("Words", fontsize=14)
plt.ylabel("TF-IDF Score", fontsize=14)

# Add value labels on top of bars
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.2f}', ha='center', va='bottom', fontweight='bold')

# Improve layout
plt.tight_layout()
plt.grid(axis='y', alpha=0.3)

# Save the plot
output_path = Path('tfidf_urdu_words.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Chart saved as: {output_path.absolute()}")

# Display the plot
plt.show()