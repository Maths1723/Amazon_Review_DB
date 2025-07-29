import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from collections import Counter
import re
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import sys
from scipy.stats import spearmanr
import os # Import the os module for path manipulation

# --- Configuration ---

FILE_PATH = 'data/amazon_product_reviews.csv'
OUTPUT_DIR = 'outputs/' # Define the output directory

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

# --- 1. Load Data ---
print("1. Loading Data...")
df = None
try:
    df = pd.read_csv(FILE_PATH)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print("First 5 rows:")
    print(df.head())
    print("Actual columns in DataFrame upon loading:")
    print(df.columns)
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}. Please check the path or upload the file.")
    print("Exiting the program as the dataset is crucial.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during file loading: {e}")
    sys.exit(1)

# --- 2. Data Cleaning and Preprocessing ---
print("\n2. Data Cleaning and Preprocessing...")

column_mapping = {
    'Text': 'review_text',
    'Score': 'rating',
    'Summary': 'review_title',
    'Time': 'review_date'
}

existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
df = df.rename(columns=existing_columns)
print(f"Columns after renaming: {df.columns.tolist()}")

required_columns = ['review_text', 'rating']
for col in required_columns:
    if col not in df.columns:
        print(f"Error: Essential column '{col}' not found in the dataset AFTER renaming. "
              "Please verify your `column_mapping` and original CSV headers.")
        sys.exit(1)

initial_rows = df.shape[0]
df.dropna(subset=['review_text', 'rating'], inplace=True)
print(f"Dropped {initial_rows - df.shape[0]} rows with missing 'review_text' or 'rating'. New shape: {df.shape}")

df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df.dropna(subset=['rating'], inplace=True)
df['rating'] = df['rating'].astype(int)
print("Converted 'rating' column to integer.")

if 'review_date' in df.columns:
    df['review_date'] = pd.to_datetime(df['review_date'], unit='s', errors='coerce')
    df.dropna(subset=['review_date'], inplace=True)
    print(f"Cleaned 'review_date' column. New shape: {df.shape}")
else:
    print("Column 'review_date' not found. Skipping date-based analysis.")

df['sentiment_category'] = df['rating'].apply(
    lambda x: 'Positive' if x >= 4 else ('Negative' if x <= 2 else 'Neutral')
)
print("Created 'sentiment_category' based on rating.")
print(df['sentiment_category'].value_counts())


# --- 3. Exploratory Data Analysis (EDA) ---
print("\n3. Exploratory Data Analysis (EDA)...")

# --- 3.1 Distribution of Ratings (Interactive Bar Chart) ---
print("\n3.1 Distribution of Ratings (Interactive Bar Chart)")
fig_rating = px.histogram(df, x='rating', nbins=5, title='Distribution of Product Ratings',
                          labels={'rating': 'Rating (1-5 Stars)'},
                          category_orders={"rating": [1, 2, 3, 4, 5]},
                          color='rating',
                          color_discrete_sequence=px.colors.qualitative.Plotly
                         )
fig_rating.write_html(os.path.join(OUTPUT_DIR, "overall_rating_distribution.html"))
print(f"Plot saved to {os.path.join(OUTPUT_DIR, 'overall_rating_distribution.html')}")

# --- 3.2 Distribution of Sentiment Categories (Interactive Pie Chart) ---
print("\n3.2 Distribution of Sentiment Categories (Interactive Pie Chart)")
sentiment_counts = df['sentiment_category'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']
fig_sentiment_pie = px.pie(sentiment_counts, values='Count', names='Sentiment',
                           title='Overall Sentiment Distribution',
                           hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
fig_sentiment_pie.write_html(os.path.join(OUTPUT_DIR, "overall_sentiment_distribution.html"))
print(f"Plot saved to {os.path.join(OUTPUT_DIR, 'overall_sentiment_distribution.html')}")

# --- 3.3 Top Products by Number of Reviews (to identify the most reviewed product) ---
most_reviewed_product_id = None
if 'ProductId' in df.columns:
    print("\n3.3 Identifying the Most Reviewed Product...")
    top_products_series = df['ProductId'].value_counts()
    if not top_products_series.empty:
        most_reviewed_product_id = top_products_series.index[0]
        num_reviews_top_product = top_products_series.iloc[0]
        print(f"Most reviewed Product ID: '{most_reviewed_product_id}' with {num_reviews_top_product} reviews.")

        # Optional: Display the top 10 products by review count visually
        print("\nTop 10 Products by Number of Reviews (Interactive Bar Chart - by ProductId)")
        top_products_df = top_products_series.nlargest(10).reset_index()
        top_products_df.columns = ['Product ID', 'Number of Reviews']
        fig_top_products = px.bar(top_products_df, x='Product ID', y='Number of Reviews',
                                  title='Top 10 Products by Number of Reviews',
                                  color='Number of Reviews', color_continuous_scale=px.colors.sequential.Viridis)
        fig_top_products.write_html(os.path.join(OUTPUT_DIR, "top_10_products.html"))
        print(f"Plot saved to {os.path.join(OUTPUT_DIR, 'top_10_products.html')}")
    else:
        print("No products found to determine the most reviewed one.")
else:
    print("\nSkipping 'Most Reviewed Product' analysis: 'ProductId' column not found.")

# --- Filter for the Most Reviewed Product and perform focused EDA ---
df_most_reviewed = None
if most_reviewed_product_id:
    print(f"\n4. Focusing EDA on Product ID: '{most_reviewed_product_id}'")
    df_most_reviewed = df[df['ProductId'] == most_reviewed_product_id].copy()
    print(f"Filtered dataset shape for most reviewed product: {df_most_reviewed.shape}")

    # Calculate review length (characters) and word count for the focused product
    df_most_reviewed['review_length'] = df_most_reviewed['review_text'].apply(len)
    df_most_reviewed['word_count'] = df_most_reviewed['review_text'].apply(lambda x: len(str(x).split()))


    # --- 4.1 Distribution of Ratings for Most Reviewed Product ---
    print(f"\n4.1 Distribution of Ratings for Product '{most_reviewed_product_id}' (Interactive Bar Chart)")
    fig_rating_prod = px.histogram(df_most_reviewed, x='rating', nbins=5,
                                   title=f'Distribution of Ratings for Product {most_reviewed_product_id}',
                                   labels={'rating': 'Rating (1-5 Stars)'},
                                   category_orders={"rating": [1, 2, 3, 4, 5]},
                                   color='rating',
                                   color_discrete_sequence=px.colors.qualitative.Plotly
                                  )
    fig_rating_prod.write_html(os.path.join(OUTPUT_DIR, f"product_{most_reviewed_product_id}_rating_distribution.html"))
    print(f"Plot saved to {os.path.join(OUTPUT_DIR, f'product_{most_reviewed_product_id}_rating_distribution.html')}")

    # --- 4.2 Sentiment Distribution for Most Reviewed Product ---
    print(f"\n4.2 Sentiment Distribution for Product '{most_reviewed_product_id}' (Interactive Pie Chart)")
    sentiment_counts_prod = df_most_reviewed['sentiment_category'].value_counts().reset_index()
    sentiment_counts_prod.columns = ['Sentiment', 'Count']
    fig_sentiment_pie_prod = px.pie(sentiment_counts_prod, values='Count', names='Sentiment',
                                    title=f'Sentiment Distribution for Product {most_reviewed_product_id}',
                                    hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_sentiment_pie_prod.write_html(os.path.join(OUTPUT_DIR, f"product_{most_reviewed_product_id}_sentiment_distribution.html"))
    print(f"Plot saved to {os.path.join(OUTPUT_DIR, f'product_{most_reviewed_product_id}_sentiment_distribution.html')}")


    # --- 4.3 Correlation between Rating and Review Length (Characters) for Most Reviewed Product ---
    print(f"\n4.3 Correlation Analysis: Rating vs. Review Length (Characters) for Product '{most_reviewed_product_id}'")
    if len(df_most_reviewed) > 1 and 'review_length' in df_most_reviewed.columns:
        correlation, p_value = spearmanr(df_most_reviewed['rating'], df_most_reviewed['review_length'])
        print(f"Spearman Correlation (Rating vs. Review Length): {correlation:.3f}")
        print(f"P-value: {p_value:.3f}")

        alpha = 0.05
        if p_value < alpha:
            print(f"Result: The correlation is statistically significant (p < {alpha}).")
            if correlation > 0:
                print("Interpretation: There is a positive correlation - longer reviews tend to have higher ratings.")
            elif correlation < 0:
                print("Interpretation: There is a negative correlation - longer reviews tend to have lower ratings.")
            else:
                print("Interpretation: The correlation is negligible.")
        else:
            print(f"Result: The correlation is NOT statistically significant (p >= {alpha}).")
            print("Interpretation: There is no strong linear relationship between review length (characters) and rating for this product.")
    else:
        print("Not enough data or 'review_length' column missing to perform correlation analysis.")

    # --- 4.4 Box Plot of Review Lengths (Characters) by Rating for Most Reviewed Product ---
    print(f"\n4.4 Box Plot: Review Length (Characters) by Rating for Product '{most_reviewed_product_id}'")
    if 'review_length' in df_most_reviewed.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='rating', y='review_length', data=df_most_reviewed, palette='viridis')
        plt.title(f'Review Length (Characters) Distribution by Rating for Product {most_reviewed_product_id}')
        plt.xlabel('Rating (Stars)')
        plt.ylabel('Review Length (Characters)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(OUTPUT_DIR, f"product_{most_reviewed_product_id}_review_length_boxplot.png")) # Save as PNG
        print(f"Plot saved to {os.path.join(OUTPUT_DIR, f'product_{most_reviewed_product_id}_review_length_boxplot.png')}")
        plt.show() # Still show it if running interactively
    else:
        print("Skipping Box Plot: 'review_length' column not found.")

    # --- 4.5 Scatter Plot: Number of Words vs. Rating for Most Reviewed Product ---
    print(f"\n4.5 Scatter Plot: Number of Words vs. Rating for Product '{most_reviewed_product_id}'")
    if 'word_count' in df_most_reviewed.columns and not df_most_reviewed.empty:
        fig_word_rating = px.scatter(df_most_reviewed, x='word_count', y='rating',
                                     title=f'Number of Words in Review vs. Rating for Product {most_reviewed_product_id}',
                                     labels={'word_count': 'Number of Words in Review', 'rating': 'Rating (1-5 Stars)'},
                                     hover_data=['review_title', 'review_text'],
                                     color='rating',
                                     color_continuous_scale=px.colors.sequential.Plasma,
                                     opacity=0.5
                                    )
        fig_word_rating.update_layout(xaxis_range=[0, df_most_reviewed['word_count'].quantile(0.99) * 1.1])
        fig_word_rating.write_html(os.path.join(OUTPUT_DIR, f"product_{most_reviewed_product_id}_word_count_vs_rating.html"))
        print(f"Plot saved to {os.path.join(OUTPUT_DIR, f'product_{most_reviewed_product_id}_word_count_vs_rating.html')}")
    else:
        print("Skipping Scatter Plot: 'word_count' column not found or DataFrame is empty.")


    # --- 4.6 Average Rating Over Time for Most Reviewed Product (if 'review_date' exists) ---
    if 'review_date' in df_most_reviewed.columns:
        print(f"\n4.6 Average Rating Over Time for Product '{most_reviewed_product_id}' (Interactive Line Chart)")
        df_most_reviewed['review_month'] = df_most_reviewed['review_date'].dt.to_period('M').dt.to_timestamp()
        monthly_avg_rating_prod = df_most_reviewed.groupby('review_month')['rating'].mean().reset_index()
        fig_time_series_prod = px.line(monthly_avg_rating_prod, x='review_month', y='rating',
                                       title=f'Average Rating Over Time for Product {most_reviewed_product_id}',
                                       labels={'review_month': 'Date', 'rating': 'Average Rating'})
        fig_time_series_prod.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        fig_time_series_prod.write_html(os.path.join(OUTPUT_DIR, f"product_{most_reviewed_product_id}_avg_rating_over_time.html"))
        print(f"Plot saved to {os.path.join(OUTPUT_DIR, f'product_{most_reviewed_product_id}_avg_rating_over_time.html')}")
    else:
        print(f"\nSkipping 'Average Rating Over Time' analysis for Product '{most_reviewed_product_id}': 'review_date' column not found.")

    # --- 4.7 Single Word Cloud for ALL Reviews of Most Reviewed Product ---
    print(f"\n4.7 Single Word Cloud for ALL Reviews of Product '{most_reviewed_product_id}' (Static Plot)")

    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update(['product', 'good', 'great', 'taste', 'love', 'flavor', 'food', 'like', 'just', 'get', 'dont', 'tea', 'one', 'coffee', 'amazon', 'would', 'really', 'much', 'can', 'also', 'it\'s', 'buy', 'bought', 'use', 'used', 'tried', 'even', 'little', 'bit', 'many', 'very', 'see', 'make', 'made', 'first', 'time', 'well', 'etc'])

    if isinstance(most_reviewed_product_id, str):
        product_id_words = re.findall(r'\b\w+\b', most_reviewed_product_id.lower())
        custom_stopwords.update(product_id_words)

    def generate_wordcloud(text, title, stopwords_set=None):
        if text:
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                  collocations=False, stopwords=stopwords_set,
                                  max_words=100).generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(title)
            plt.savefig(os.path.join(OUTPUT_DIR, f"product_{most_reviewed_product_id}_wordcloud.png")) # Save as PNG
            print(f"Plot saved to {os.path.join(OUTPUT_DIR, f'product_{most_reviewed_product_id}_wordcloud.png')}")
            plt.show() # Still show it if running interactively
        else:
            print(f"No text available for {title} word cloud.")

    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    if 'review_text' in df_most_reviewed.columns:
        all_reviews_text_prod = ' '.join(df_most_reviewed['review_text'].apply(preprocess_text).dropna())
        generate_wordcloud(all_reviews_text_prod, f'Word Cloud for All Reviews - Product {most_reviewed_product_id}', custom_stopwords)
    else:
        print(f"\nSkipping Word Cloud analysis for Product '{most_reviewed_product_id}': 'review_text' column not found.")

else:
    print("\nCannot perform focused EDA: Most reviewed product could not be identified.")


print("\nEDA Complete! These insights will help in designing your AI Agent's prompts.")
print("\n--- Key Takeaways for AI Agent Development ---")
print("1. Rating Distribution: Helps understand the baseline sentiment in the dataset.")
print("2. Sentiment Categories: Provides a high-level view of positive, neutral, and negative reviews.")
print("3. Review Lengths & Correlation: Shows if review length is related to rating, which can inform token limits or highlight detailed reviews (often longer and negative/positive).")
print("4. Word Count vs. Rating: Visualizes the relationship between review verbosity and assigned rating, often showing interesting clusters.")
print("5. Single Word Cloud: Offers a quick visual summary of the most frequent terms across all reviews for the most popular product, helping to identify overall themes.")
print("6. Product ID: Useful for segmenting analysis by product, especially when specific product names aren't available.")
print("7. Time Trends (if dates available): Helps to see if sentiment changes over time, indicating product updates or market shifts.")
print("\nNext steps would be to use these insights to craft effective prompts for your LLM-based agent!")