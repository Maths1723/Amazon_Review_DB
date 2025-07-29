# 🤖 Agent AI for Product Review Analysis

---

This project demonstrates the development of an **AI Agent** designed to analyze product reviews, extract key insights, and generate actionable summaries. It showcases practical experience with **Large Language Models (LLMs)**, **workflow automation**, and **data analysis with AI techniques**, directly addressing the requirements for the Junior AI Specialist role.

## ✨ Project Overview

In today's competitive market, understanding customer feedback is crucial for strategic decision-making. This project presents an automated solution to process vast amounts of product review data, turning raw text into clear, concise, and actionable intelligence.

The core components include:

* **Data Ingestion & Preprocessing:** Handling raw review data.
* **AI Agent Core:** Utilizing LLMs for sentiment analysis, topic extraction, and insightful summary generation.
* **Automated Workflow:** Orchestrating the process for continuous monitoring and reporting.

## 🚀 Key Features

* **Automated Sentiment Analysis:** Accurately classifies review sentiment (Positive, Negative, Neutral) using LLM capabilities.
* **Dynamic Topic Extraction:** Identifies recurring themes and keywords mentioned in reviews, providing granular insights beyond simple sentiment.
* **Actionable Summary Generation:** Synthesizes complex review data into concise, business-oriented recommendations for product improvement or marketing strategies.
* **Scalable Workflow:** Designed with automation in mind, allowing for integration into larger business processes.
* **Interactive Data Exploration:** Includes tools for initial data understanding and visualization of results.

## 🛠️ Technologies Used

* **Python:** Core programming language for data processing and AI agent logic.
* **Large Language Models (LLMs):** Leveraged for sophisticated text understanding and generation (e.g., using Hugging Face models for local execution or OpenAI API).
* **LangChain (Optional but recommended for the agent part):** Framework for building complex LLM applications and agentic workflows.
* **Pandas:** For efficient data manipulation and analysis.
* **Plotly Express / Matplotlib / Seaborn:** For interactive and static data visualization during EDA.
* **WordCloud:** For generating visual representations of dominant words.

## 💡 How It Works

The project operates in several stages:

1.  **Data Ingestion:** Reads product review data (e.g., from an Amazon Reviews CSV).
2.  **Exploratory Data Analysis (EDA):** An initial phase (documented in `eda_script.py` or a Jupyter notebook) to understand the dataset's characteristics. This includes:
    * **Overall Rating & Sentiment Distribution:** Visualizing the breakdown of star ratings and sentiment categories across the entire dataset.
    * **Most Reviewed Product Deep Dive:** Focusing analysis on the product with the highest number of reviews to get targeted insights.
        * Its specific rating and sentiment distributions.
        * Correlation between review length (characters) and rating.
        * Relationship between the number of words in a review and its rating (scatter plot).
        * Average rating trends over time.
        * A **single word cloud** representing the most frequent words across **all** reviews for this specific product.
    This EDA informs the design of the AI agent's prompts.
3.  **AI Agent Processing:** Each review is passed to the LLM-powered agent. The agent is prompted to:
    * Determine the **sentiment**.
    * **Extract key topics** or features discussed.
    * Generate a **concise, actionable summary** or recommendation.
4.  **Workflow Automation (n8n):** An n8n workflow orchestrates the execution, potentially triggering the analysis on new data, processing it, and outputting the insights to a structured format (e.g., a report file, a dashboard).

## 📊 Project Structure

.
├── data/
│   └── amazon_product_reviews.csv  # Your dataset (or a sample)
├── eda_script.py               # Python script for EDA (as provided previously)
|─ ai_agent.py                 # Python script for the LLM-powered agent logic
├── outputs/
│   └── analyzed_reviews.csv        # Example output of processed reviews
│   └── overall_rating_distribution.html # Example HTML output for overall rating plot
│   └── overall_sentiment_distribution.html # Example HTML output for overall sentiment plot
│   └── top_10_products.html        # Example HTML output for top products plot
│   └── product_[ID]_rating_distribution.html # Example HTML output for specific product rating plot
│   └── product_[ID]_sentiment_distribution.html # Example HTML output for specific product sentiment plot
│   └── product_[ID]_word_count_vs_rating.html # Example HTML output for word count vs rating scatter plot
│   └── product_[ID]_avg_rating_over_time.html # Example HTML output for average rating over time plot
│   └── product_[ID]_review_length_boxplot.png # Example PNG output for review length boxplot
│   └── product_[ID]_wordcloud.png # Example PNG output for the word cloud
├── requirements.txt                # Python dependencies
└── README.md                       # This file


## ⚙️ Setup and Run

To set up and run this project locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/product-review-ai-agent.git](https://github.com/yourusername/product-review-ai-agent.git)
    cd product-review-ai-agent
    ```
2.  **Prepare Data:**
    * **Download Dataset:** The project utilizes a dataset of Amazon product reviews.
        * Go to the Kaggle dataset page: [https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
        * Download the `Reviews.csv` file (or the full dataset archive and extract `Reviews.csv`).
        * Rename the downloaded file to `amazon_product_reviews.csv`.
        * Place this `amazon_product_reviews.csv` file into the `data/` directory of this project.
    * **_Important_**: Verify and adjust the column names in `eda_script.py` and `ai_agent.py` to match your dataset's headers. The current scripts expect `Text`, `Score`, `Summary`, and `Time`.
3.  **Python Environment Setup:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run EDA:**
    * Execute the EDA script:
        ```bash
        python scripts/eda_script.py
        ```
    * The script will generate several `.html` files (for interactive Plotly charts) and `.png` files (for static Matplotlib plots like box plots and word clouds) within the `outputs/` directory.
    * **To view the interactive plots**, open the generated `.html` files in your web browser (e.g., `outputs/overall_rating_distribution.html`).
5.  **Configure LLM:**
    * If using local LLMs, ensure you have the models downloaded and configured as per the `ai_agent.py` script's instructions.
    * If using OpenAI API, set your `OPENAI_API_KEY` environment variable.
6.  **Run AI Agent (Standalone for testing):**
    ```bash
    python scripts/ai_agent.py
    ```
    *(The `ai_agent.py` script will be a simplified version for demonstration, showing how it processes a few sample reviews).*
7.  **Setup n8n Workflow:**
    * Ensure you have n8n running (local or cloud instance).
    * Import the `n8n_workflow/product_review_agent_workflow.json` file into your n8n instance.
    * Configure the "Execute Command" or "HTTP Request" node in n8n to correctly call your `ai_agent.py` script.
    * Activate the workflow to see the automation in action.

## 📈 Results & Insights

* **Example Actionable Insights Generated:**
    * *Original Review:* "The phone's camera is amazing, but the battery dies too fast after the last update."
    * *AI Agent Insight:* **"Negative sentiment due to battery drain post-update. Action: Investigate recent software updates for battery optimization issues."**
    * *Original Review:* "Easy to set up and great sound quality, but customer support took ages to respond to my query."
    * *AI Agent Insight:* **"Positive product experience (setup, sound) but negative customer service experience. Action: Improve customer support response times or enhance self-help resources."**

## 💡 Future Enhancements

* Integration with live data streams (e.g., direct API connection to e-commerce platforms).
* Fine-tuning of LLMs for domain-specific language in product reviews.
* Development of a comprehensive dashboard for real-time monitoring of insights.
* More advanced topic modeling (e.g., LDA, BERTopic) for deeper thematic analysis.
