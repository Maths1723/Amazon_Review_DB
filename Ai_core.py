import pandas as pd
import json
from tqdm import tqdm
import os
import re
from collections import Counter
import numpy as np

# --- Fase 1: Preparazione e Configurazione Iniziale ---

# 1.2. Definizione delle Costanti e Percorsi
INPUT_CSV_PATH = 'data/amazon_product_reviews.csv'
# Definiamo la directory di output. Il percorso completo del file CSV sarà generato dinamicamente.
OUTPUT_DIR = 'outputs'
LLM_MODEL_NAME = 'microsoft/phi-2'
OPENAI_API_KEY_ENV_VAR = 'OPENAI_API_KEY'

# Percorso dove salveremo/caricheremo il modello locale
LOCAL_MODEL_PATH = f'models/{LLM_MODEL_NAME.replace("/", "_")}'

# Variabile per scegliere tra LLM locale e OpenAI API
USE_LOCAL_LLM = True

# Assicurati che le directory necessitano esistano
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(INPUT_CSV_PATH), exist_ok=True)
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

# --- Fase 2: Configurazione e Interazione con l'LLM ---

if USE_LOCAL_LLM:
    try:
        import torch
        from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, BitsAndBytesConfig
        print(f"Attempting to load local LLM pipeline for {LLM_MODEL_NAME}...")
        if torch.cuda.is_available():
            print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
            # DEVICE is still used to decide if BNB_CONFIG is applied, but not passed directly to pipeline if BNB is active.
            DEVICE = 0
            BNB_CONFIG = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            )
        else:
            print("CUDA is not available. Using CPU. Quantization will not be applied.")
            DEVICE = -1
            BNB_CONFIG = None
    except ImportError as e:
        print(f"Error: Required libraries not found. Please install them: pip install transformers torch bitsandbytes accelerate")
        print(f"Error details: {e}")
        USE_LOCAL_LLM = False
        DEVICE = -1
        BNB_CONFIG = None
    except Exception as e:
        print(f"Error during initial setup: {e}")
        USE_LOCAL_LLM = False
        DEVICE = -1
        BNB_CONFIG = None
else:
    DEVICE = -1
    BNB_CONFIG = None

def load_local_llm(model_name=LLM_MODEL_NAME, local_path=LOCAL_MODEL_PATH, device=DEVICE, bnb_config=BNB_CONFIG):
    print(f"Loading local LLM: {model_name}...")
    try:
        if "t5" in model_name.lower() or "bart" in model_name.lower():
            AutoModelClass = AutoModelForSeq2SeqLM
            pipeline_task = "text2text-generation"
        else:
            AutoModelClass = AutoModelForCausalLM
            pipeline_task = "text-generation"

        if os.path.exists(local_path) and os.path.isdir(local_path) and len(os.listdir(local_path)) > 0:
            print(f"Loading model from local path: {local_path}")
            tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
            model = AutoModelClass.from_pretrained(local_path, trust_remote_code=True,
                                                   quantization_config=bnb_config if device != -1 else None,
                                                   torch_dtype=torch.bfloat16 if device != -1 else None
                                                  )
        else:
            print(f"Model not found locally. Downloading and saving {model_name} to {local_path}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelClass.from_pretrained(model_name, trust_remote_code=True,
                                                   quantization_config=bnb_config if device != -1 else None,
                                                   torch_dtype=torch.bfloat16 if device != -1 else None
                                                  )
            model.save_pretrained(local_path)
            tokenizer.save_pretrained(local_path)
            print(f"Model saved to {local_path}")

        # MODIFICATION: If bnb_config is used, accelerate handles device placement, so do not pass device to pipeline.
        if bnb_config and device != -1: # bnb_config is not None and we are on GPU
            generator = pipeline(
                pipeline_task,
                model=model,
                tokenizer=tokenizer,
                # device=device # REMOVED: Accelerate handles this when quantization is used
            )
        else: # CPU or no quantization, pass device
            generator = pipeline(
                pipeline_task,
                model=model,
                tokenizer=tokenizer,
                device=device
            )
        print("Local LLM loaded successfully.")
        return generator
    except Exception as e:
        print(f"Error loading local LLM {model_name}: {e}")
        print("Falling back to OpenAI API if USE_LOCAL_LLM is set to False.")
        return None

def generate_response_local(generator, prompt, max_new_tokens=256):
    if generator is None:
        return "{'sentiment': 'Error', 'topics': [], 'actionable_insight': 'Local LLM not loaded.'}"
    try:
        if generator.tokenizer.pad_token_id is None:
            if hasattr(generator.tokenizer, 'eos_token_id') and generator.tokenizer.eos_token_id is not None:
                generator.tokenizer.pad_token_id = generator.tokenizer.eos_token_id
            else:
                generator.tokenizer.pad_token_id = 0

        if generator.task == "text2text-generation":
            result = generator(
                prompt,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1
            )
            return result[0]['generated_text'].strip()
        else:
            result = generator(
                prompt,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )
            generated_text = result[0]['generated_text']
            # Phi-2 often repeats the prompt, so we remove it
            if generated_text.startswith(prompt):
                return generated_text[len(prompt):].strip()
            return generated_text.strip()
    except Exception as e:
        print(f"Error during local LLM inference: {e}")
        return "{'sentiment': 'Error', 'topics': [], 'actionable_insight': 'LLM inference failed.'}"

# 2.2. Configurazione e Interazione con l'LLM (Opzione B: OpenAI API - Fallback/Alternativa)
if not USE_LOCAL_LLM:
    try:
        from openai import OpenAI
        print("Attempting to use OpenAI API...")
    except ImportError:
        print("Error: 'openai' library not found. Please install it: pip install openai")
        print("Cannot proceed without a valid LLM setup (local or OpenAI). Exiting.")
        exit()

def generate_response_openai(prompt, model="gpt-3.5-turbo"):
    api_key = os.getenv(OPENAI_API_KEY_ENV_VAR)
    if not api_key:
        raise ValueError(f"OpenAI API key not found in environment variable: {OPENAI_API_KEY_ENV_VAR}")

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in product review analysis. Provide output only in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return "{'sentiment': 'Error', 'topics': [], 'actionable_insight': 'OpenAI API call failed.'}"

# --- Fase 3: Caricamento e Preparazione dei Dati ---

# Crea un file CSV dummy per test SOLO se non esiste
print(f"Checking for dummy CSV at {INPUT_CSV_PATH}.")
if not os.path.exists(INPUT_CSV_PATH):
    print(f"Dummy CSV not found. Creating a dummy file at {INPUT_CSV_PATH} for testing purposes.")

    prod_A_reviews = [
        # 1-star reviews (more than 10)
        "Absolutely terrible. It broke after a week and customer service was unhelpful. A waste of money.", #1
        "Worst purchase ever. Freezes constantly and the software is full of bugs.", #1
        "Customer service was abysmal. They didn't help me at all with my issue. Never buying again.", #1
        "The software updates are terrible. They introduced more bugs than fixes. Frustrating experience.", #1
        "Constant disconnections with Wi-Fi. Unreliable for daily use. Regret this purchase.", #1
        "Very fragile, dropped it once and it shattered. Not durable at all.", #1
        "It overheats dramatically even with light use. Uncomfortable to hold after a while.", #1
        "Not compatible with my other devices. Integration is a nightmare.", #1
        "Advertised features are missing or don't work as expected. Misleading.", #1
        "Customer support was rude and unhelpful. Seriously considering a return.", #1
        "The charging port is loose after only a month. Bad build quality.", #1
        "This product is a total scam. It never worked from day one.", #1
        "Complete trash. Don't waste your money on this.", #1

        # 2-star reviews (more than 10)
        "The delivery was fast, but the item arrived damaged. Very disappointed with the packaging.", #2
        "Mediocre. The features are limited and the user interface is clunky. Not worth the upgrade.", #2
        "Screen quality is poor, colors are washed out and brightness is insufficient. Major disappointment.", #2
        "The price is way too high for what you get. Feels cheap and performs poorly.", #2
        "The speaker sound is tinny and low. Can barely hear anything without headphones.", #2
        "The user interface is not intuitive at all. Very confusing to navigate and learn.", #2
        "Poor camera performance in low light. Pictures are grainy and blurry. Expected more.", #2
        "It's loud when operating. Distracting during quiet moments.", #2
        "The construction feels flimsy, not solid at all. Worried it won't last.", #2
        "The size is too bulky for a portable device. Hard to carry around.", #2
        "Product arrived with scratches. Clearly not new. Disappointed.", #2
        "Laggy performance even with basic apps. Not worth the money.", #2
        "Barely functions, constantly crashes.", #2

        # 3-star reviews (more than 10)
        "It's okay, nothing special. The screen is decent but the sound quality is very poor.", #3
        "Good product for the price. The setup was a bit complicated, but once it's working, it's fine.", #3
        "Battery life is acceptable, but charging takes forever. Annoying.", #3
        "It performs adequately, but there are better options for the price.", #3
        "Neutral feelings. Some good features, some bad. Not blown away.", #3
        "Decent, but the build quality is questionable.", #3
        "Reliable so far, but I had higher expectations for certain features.", #3
        "It does the job, no more, no less.", #3
        "Average performance. Nothing stands out as great or terrible.", #3
        "Setup was tricky, but manageable eventually.", #3
        "Gets warm with heavy use, but not to the point of overheating.", #3
        "Sound is acceptable for basic use, but not for audiophiles.", #3
        "User interface is functional, just not polished.", #3

        # 4 & 5-star reviews (to ensure prod_A is most reviewed)
        "This product is amazing! The battery life is incredible and the camera takes stunning photos.", #5
        "I love this new gadget! So easy to use and very intuitive. Highly recommend for beginners.", #5
        "Fantastic product, exceeded my expectations. The design is sleek and it performs flawlessly.", #5
        "Good performance for the cost, surprisingly capable.", #4
        "Very sleek design, looks premium.", #4
        "Excellent battery life, lasts all day.", #5
        "Easy to set up and start using immediately.", #5
        "Camera quality is impressive, especially in good lighting.", #4
        "The screen is vibrant and sharp, great for media.", #5
        "Reliable connection, never drops out.", #4
        "Solid build quality, feels durable.", #4
        "Software is smooth and bug-free.", #5
        "Great value, highly recommend.", #5
    ]
    prod_A_ratings = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # 1-star (13)
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, # 2-star (13)
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, # 3-star (13)
        5, 5, 5, 4, 4, 5, 5, 4, 5, 4, 4, 5, 5  # 4 & 5-star (13)
    ]
    prod_A_ids = ['prod_A'] * len(prod_A_reviews)

    # Other products (to ensure prod_A is still the most reviewed and to have some other products)
    prod_B_reviews = ["ProdB is good.", "ProdB is bad."]
    prod_B_ratings = [4, 1]
    prod_B_ids = ['prod_B'] * len(prod_B_reviews)

    prod_C_reviews = ["ProdC is fine."]
    prod_C_ratings = [3]
    prod_C_ids = ['prod_C'] * len(prod_C_reviews)

    # Added a prod_0 to ensure it doesn't accidentally become max if user had one with few reviews
    prod_0_reviews = ["This is a test review for prod_0.", "Another one for prod_0."]
    prod_0_ratings = [4, 2]
    prod_0_ids = ['prod_0'] * len(prod_0_reviews)


    dummy_data = {
        'review_text': prod_A_reviews + prod_B_reviews + prod_C_reviews + prod_0_reviews,
        'rating': prod_A_ratings + prod_B_ratings + prod_C_ratings + prod_0_ratings,
        'product_id': prod_A_ids + prod_B_ids + prod_C_ids + prod_0_ids
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv(INPUT_CSV_PATH, index=False)
    print("Dummy CSV created. If you have your own data, replace this file with it.")
else:
    print(f"Using existing CSV file at {INPUT_CSV_PATH}.")


try:
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"Loaded {len(df)} reviews from {INPUT_CSV_PATH}")
except FileNotFoundError:
    print(f"Error: Input CSV file not found at {INPUT_CSV_PATH}. Even after dummy creation attempt, it's missing. Exiting.")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- Column Renaming and Validation ---
# Define common possible column names for review text, rating, and product ID
REVIEW_TEXT_COLS = ['review_text', 'Text', 'reviewText', 'Reviews', 'Body']
RATING_COLS = ['rating', 'Score', 'overall', 'Rating']
PRODUCT_ID_COLS = ['product_id', 'ProductId', 'id', 'asin', 'productID']

# Function to find and rename a column
def find_and_rename_column(df, possible_names, target_name):
    for col_name in possible_names:
        if col_name in df.columns:
            if col_name != target_name:
                df.rename(columns={col_name: target_name}, inplace=True)
                print(f"Renamed column '{col_name}' to '{target_name}'.")
            return True
    return False

# Attempt to find and rename required columns
if not find_and_rename_column(df, REVIEW_TEXT_COLS, 'review_text'):
    raise ValueError(f"Missing required column: 'review_text'. Please ensure your CSV has a review text column named one of: {REVIEW_TEXT_COLS}")
if not find_and_rename_column(df, RATING_COLS, 'rating'):
    raise ValueError(f"Missing required column: 'rating'. Please ensure your CSV has a rating column named one of: {RATING_COLS}")
if not find_and_rename_column(df, PRODUCT_ID_COLS, 'product_id'):
    raise ValueError(f"Missing required column: 'product_id'. Please ensure your CSV has a product ID column named one of: {PRODUCT_ID_COLS}")

# --- Filtering for valid reviews ---
initial_rows = len(df)
# Convert review_text to string type to handle mixed types gracefully
df['review_text'] = df['review_text'].astype(str)
# Remove reviews where 'review_text' is empty or contains only whitespace
df = df[df['review_text'].str.strip() != ''].copy()
# Remove rows where rating or product_id are NaN after renames
df.dropna(subset=['review_text', 'rating', 'product_id'], inplace=True)

rows_after_cleaning = len(df)
print(f"Removed {initial_rows - rows_after_cleaning} rows with missing or empty review text, rating, or product ID.")
print(f"Processing {rows_after_cleaning} reviews after cleaning.")


if df.empty:
    print("No reviews to process after cleaning. Exiting.")
    exit()

# Ensure rating is numeric
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df.dropna(subset=['rating'], inplace=True)
print(f"Processing {len(df)} reviews after ensuring rating is numeric.")


# --- Selezione del Prodotto più Recensito e Filtraggio Specifico ---
print("\n--- Product Selection and Specific Review Filtering ---")
# Conta le recensioni per ogni prodotto
product_review_counts = df['product_id'].value_counts()
print("Review counts per product:")
print(product_review_counts.head(5)) # Print top 5 to avoid flooding console for large datasets

if product_review_counts.empty:
    print("No products found in the dataset after cleaning. Exiting.")
    exit()

# Trova l'ID del prodotto con più recensioni
most_reviewed_product_id = product_review_counts.idxmax()
num_reviews_for_product = product_review_counts.max()

print(f"\nSelected product for analysis: '{most_reviewed_product_id}' with {num_reviews_for_product} total reviews.")

# Filter the DataFrame for the most reviewed product
df_product_reviews = df[df['product_id'] == most_reviewed_product_id].copy()

# MODIFICATION: Changed review selection logic to prioritize negative reviews up to MAX_REVIEWS_TO_PROCESS
MAX_REVIEWS_TO_PROCESS = 10 # Aim for a total of 10 reviews

# Create a pool of negative reviews (ratings 1, 2, 3) for the selected product
# Sort by rating to prioritize 1-star, then 2-star, then 3-star
negative_reviews_pool = df_product_reviews[df_product_reviews['rating'].isin([1, 2, 3])].copy()
negative_reviews_pool = negative_reviews_pool.sort_values(by='rating', ascending=True).reset_index(drop=True)

print(f"Available negative reviews for '{most_reviewed_product_id}': {len(negative_reviews_pool)}")

# Aggiorna il percorso di output per riflettere il filtro
clean_product_id = re.sub(r'[\\/:*?"<>|]', '_', str(most_reviewed_product_id))
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, f'analyzed_selected_negative_reviews_for_product_{clean_product_id}.csv')


# --- Fase 4: Progettazione del Prompt LLM (Il Cervello dell'Agente) ---

def create_prompt(review_text):
    # Esempio 1: Recensione Negativa
    example_review_1 = "This phone overheats constantly and the battery dies in an hour. Customer support was useless."
    example_output_1 = {
        "sentiment": "Negative",
        "topics": ["overheating", "battery life", "customer support"],
        "actionable_insight": "Address overheating issues and improve battery efficiency; enhance customer service training."
    }

    # Esempio 2: Recensione Positiva (per bilanciare e mostrare varietà)
    example_review_2 = "The camera is amazing and the design is super sleek. Love the performance too!"
    example_output_2 = {
        "sentiment": "Positive",
        "topics": ["camera quality", "design", "performance"],
        "actionable_insight": "Highlight camera and design in marketing; ensure consistent performance updates."
    }

    # Esempio 3: Recensione Neutra
    example_review_3 = "It's okay, nothing special. The screen is decent but the sound quality is very poor."
    example_output_3 = {
        "sentiment": "Neutral",
        "topics": ["screen quality", "sound quality"],
        "actionable_insight": "Investigate improving sound quality for future models."
    }


    prompt = f"""
    Analyze the following product review and extract the sentiment, key topics, and one actionable insight.
    The output should be a JSON object with the following keys:
    - "sentiment": "Positive", "Negative", or "Neutral"
    - "topics": A list of 3-5 main topics or features mentioned in the review (e.g., "battery life", "camera quality", "customer service", "ease of use", "design", "performance", "software", "price").
    - "actionable_insight": A concise sentence (max 20 words) suggesting a specific action for product improvement or marketing, based on the review.

    Example Review: "{example_review_1}"
    Example JSON Output: {json.dumps(example_output_1)}

    Example Review: "{example_review_2}"
    Example JSON Output: {json.dumps(example_output_2)}

    Example Review: "{example_review_3}"
    Example JSON Output: {json.dumps(example_output_3)}

    Review: "{review_text}"

    JSON Output:
    """
    return prompt

# --- Fase 5: Implementazione della Logica dell'Agente AI ---

llm_generator = None
if USE_LOCAL_LLM:
    llm_generator = load_local_llm(LLM_MODEL_NAME, local_path=LOCAL_MODEL_PATH, device=DEVICE, bnb_config=BNB_CONFIG)
    if llm_generator is None:
        print("Local LLM failed to load. Switching to OpenAI API.")
        USE_LOCAL_LLM = False

if not USE_LOCAL_LLM:
    generate_func = generate_response_openai
    print("Using OpenAI API for LLM inference.")
else:
    generate_func = generate_response_local
    print("Using local LLM for inference.")

processed_reviews_data = [] # To store successfully processed reviews
review_count = 0
attempts_made = 0
max_attempts_per_review_parse = 3 # Max retries for parsing a single review's LLM output
max_overall_review_attempts = len(negative_reviews_pool) * max_attempts_per_review_parse # Safety break

print("\nStarting AI Agent processing...")
# MODIFICATION: Using a while loop to ensure target reviews are processed, re-trying or skipping malformed outputs
pbar = tqdm(total=MAX_REVIEWS_TO_PROCESS, desc=f"Processing reviews for {most_reviewed_product_id}")

while review_count < MAX_REVIEWS_TO_PROCESS and attempts_made < max_overall_review_attempts:
    if negative_reviews_pool.empty:
        print("\nWarning: Ran out of negative reviews in the pool before reaching target count.")
        break

    # Get the next review from the pool (and remove it to avoid reprocessing)
    current_review_row = negative_reviews_pool.iloc[0]
    negative_reviews_pool = negative_reviews_pool.iloc[1:].reset_index(drop=True)
    
    review_text = current_review_row['review_text']
    prompt = create_prompt(review_text)

    llm_output_str = ""
    parsed_successfully = False
    
    for try_count in range(max_attempts_per_review_parse):
        attempts_made += 1
        try:
            if USE_LOCAL_LLM:
                llm_output_str = generate_func(llm_generator, prompt)
            else:
                llm_output_str = generate_func(prompt)

            cleaned_output = llm_output_str.strip()
            json_match = re.search(r"\{.*\}", cleaned_output, re.DOTALL)
            parsed_output = {}

            if json_match:
                try:
                    parsed_output = json.loads(json_match.group(0))
                    # Basic validation: check for expected keys
                    if all(k in parsed_output for k in ['sentiment', 'topics', 'actionable_insight']):
                        parsed_successfully = True
                        break # Successfully parsed, exit retry loop
                    else:
                        print(f"JSON parsed but missing expected keys for review: '{review_text[:50]}...' (Attempt {try_count+1}/{max_attempts_per_review_parse})")
                except json.JSONDecodeError as e:
                    print(f"Malformed JSON for review: '{review_text[:50]}...' Error: {e} (Attempt {try_count+1}/{max_attempts_per_review_parse})")
            else:
                # Fallback for non-JSON or simple sentiment detection
                if "positive" in cleaned_output.lower():
                    parsed_output = {"sentiment": "Positive", "topics": [], "actionable_insight": "No specific insight, review was positive."}
                    parsed_successfully = True
                    break
                elif "negative" in cleaned_output.lower():
                    parsed_output = {"sentiment": "Negative", "topics": [], "actionable_insight": "No specific insight, review was negative."}
                    parsed_successfully = True
                    break
                elif "neutral" in cleaned_output.lower():
                    parsed_output = {"sentiment": "Neutral", "topics": [], "actionable_insight": "No specific insight, review was neutral."}
                    parsed_successfully = True
                    break
                else:
                    print(f"No JSON object or recognizable sentiment keyword found for review: '{review_text[:50]}...' (Attempt {try_count+1}/{max_attempts_per_review_parse})")

        except Exception as e:
            print(f"An unexpected error occurred during LLM inference or parsing for review: '{review_text[:50]}...' Error: {e} (Attempt {try_count+1}/{max_attempts_per_review_parse})")
            llm_output_str = "" # Clear output if an exception occurred

    if parsed_successfully:
        # Create a new dictionary to store the row data + LLM results
        processed_row = current_review_row.to_dict()
        processed_row['llm_sentiment'] = parsed_output.get('sentiment', 'Unknown')
        processed_row['llm_topics'] = json.dumps(parsed_output.get('topics', []))
        processed_row['llm_actionable_insight'] = parsed_output.get('actionable_insight', 'No insight generated.')
        processed_reviews_data.append(processed_row)
        review_count += 1
        pbar.update(1)
    else:
        # If parsing failed after all attempts, this review is effectively discarded, and the loop continues
        print(f"Failed to parse LLM output for review (ID: {current_review_row.get('Id', 'N/A')}, Rating: {current_review_row['rating']}): '{review_text[:100]}...' after {max_attempts_per_review_parse} attempts. Skipping this review and trying another.")

pbar.close()
df_to_process = pd.DataFrame(processed_reviews_data)
print(f"AI Agent processing complete. Successfully processed {len(df_to_process)} reviews.")

# --- Fase 6: Output e Reporting ---

# 6.1. Salvataggio del Dataset Arricchito
try:
    df_to_process.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nAnalyzed reviews saved to {OUTPUT_CSV_PATH}")
except Exception as e:
    print(f"Error saving output CSV: {e}")

# 6.2. Generate single TXT file with specific format
TXT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'all_parsed_reviews_summary.txt')

print(f"\nGenerating summary text file at {TXT_OUTPUT_PATH}...")
try:
    with open(TXT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        # Using a simple CSV-like format for the TXT file
        f.write("Rating,Review Text,AI Actionable Summary\n") # Header

        for index, row in df_to_process.iterrows():
            rating = row['rating']
            # Clean review text and summary for CSV format: remove newlines, escape quotes
            review = str(row['review_text']).replace('"', '""') # Escape double quotes
            review = review.replace('\n', ' ').replace('\r', '') # Replace newlines with spaces

            ai_summary = str(row['llm_actionable_insight']).replace('"', '""') # Escape double quotes
            ai_summary = ai_summary.replace('\n', ' ').replace('\r', '') # Replace newlines with spaces

            # Quote fields if they contain commas or quotes to ensure proper CSV structure within the TXT file
            # This makes it readable as a CSV if opened in spreadsheet software
            def quote_if_needed(field_str):
                if ',' in field_str or '"' in field_str:
                    return f'"{field_str}"'
                return field_str
            
            # Ensure no extra spaces between fields and correct comma separation
            f.write(f"{quote_if_needed(str(rating))},{quote_if_needed(review)},{quote_if_needed(ai_summary)}\n")
    print("Summary text file generated successfully.")
except Exception as e:
    print(f"Error generating summary text file: {e}")


# 6.3. Riepilogo Semplice dei Risultati dell'Agente
print("\n--- AI Agent Analysis Summary ---")
if not df_to_process.empty:
    print("Sentiment Distribution (LLM-classified):")
    print(df_to_process['llm_sentiment'].value_counts())

    # Raccogli tutti i topics e contali
    all_topics = []
    for topics_str in df_to_process['llm_topics'].dropna():
        try:
            topics_list = json.loads(topics_str)
            all_topics.extend([str(topic).strip() for topic in topics_list if isinstance(topic, str)])
        except json.JSONDecodeError:
            pass

    if all_topics:
        top_topics = Counter(all_topics).most_common(10)
        print("\nTop 10 Extracted Topics:")
        for topic, count in top_topics:
            print(f"- {topic}: {count}")
    else:
        print("\nNo topics extracted or parsing failed for all reviews.")

    print("\nSample Actionable Insights:")
    num_samples = min(5, len(df_to_process))
    if num_samples > 0:
        print(df_to_process['llm_actionable_insight'].sample(num_samples, random_state=42).tolist())
    else:
        print("No actionable insights to display.")
else:
    print("No reviews were successfully processed to generate a summary.")

print("\nScript execution finished.")