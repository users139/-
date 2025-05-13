import json
import time
import os
import base64
import httpx
from PIL import Image
import io
from openai import OpenAI
# import httpx # Uncomment if custom SSL handling is needed

# --- CONFIGURATION ---
# WARNING: Hardcoding API keys is insecure for production. Use environment variables.
OPENROUTER_API_KEY = "sk-9cf64bbdee534b9982b67d82e4974a3d"  # <-- REPLACE WITH YOUR ACTUAL KEY
OPENROUTER_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# IMPORTANT: Choose a multi-modal model available on OpenRouter
# Examples: "openai/gpt-4o-mini", "anthropic/claude-3-haiku", "openai/gpt-4-vision-preview"
OPTIMIZATION_LLM_MODEL_NAME = "qwen2.5-vl-32b-instruct" # Or your preferred multi-modal model

INPUT_ASSIGNED_QUERIES_FILE = 'output_assigned_queries_new_image_path.jsonl'
INPUT_MODEL_INFO_FILE = 'model_information_1.jsonl'
OUTPUT_OPTIMIZED_QUERIES_FILE = 'optimized_queries.jsonl'

IMAGES_FOLDER = r'D:\workplace\多模数据集\MLLM-Tool\images' # Folder containing the images

TARGET_IMAGE_SIZE_MB = 1 # Target max size in MB
TARGET_IMAGE_SIZE_BYTES = TARGET_IMAGE_SIZE_MB * 1024 * 1024 # Convert MB to bytes

API_CALL_INTERVAL = 0.5 # Seconds delay between API calls
MAX_RETRIES_LLM_CALL = 3 # Max retries for a single LLM call
RETRY_DELAY = 1

# --- Proxy Settings ---
# Attempt to set proxy environment variables. This might work with libraries like requests/httpx.
PROXY_URL = "http://:@hkgpqwg00206.huawei.com:8080"
os.environ["http_proxy"] = PROXY_URL
os.environ["https_proxy"] = PROXY_URL
print(f"Attempting to set proxy to: {PROXY_URL}")


# --- Initialize OpenAI Client for OpenRouter ---
# Standard client (should respect proxy env vars if using requests/httpx)


# If you specifically need to configure proxy/SSL in httpx:

custom_http_client = httpx.Client(
        verify=False, # Set to True if you have proper certs
    )
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_API_BASE,
    http_client=custom_http_client
)





def get_image_size(image_bytes):
    """Returns the size of image bytes in bytes."""
    return len(image_bytes)

def compress_image_to_target_size(image_path, target_size_bytes, initial_quality=90, quality_step=5, max_attempts=20):
    """
    Compresses an image iteratively to be below a target size.
    Returns base64 encoded string of the compressed image or None if failed.
    """
    full_image_path = os.path.join(IMAGES_FOLDER, image_path)
    # print(f"  Processing image: {full_image_path}") # Reduced verbosity here

    try:
        img = Image.open(full_image_path).convert('RGB') # Ensure RGB for consistent saving

        # First, check if initial size is already okay (saving as JPEG with default quality)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        initial_bytes = img_byte_arr.getvalue()

        if get_image_size(initial_bytes) <= target_size_bytes:
            # print(f"  Initial JPEG size ({get_image_size(initial_bytes)} bytes) is already within target.")
            return base64.b64encode(initial_bytes).decode('utf-8')

        # print(f"  Initial JPEG size: {get_image_size(initial_bytes)} bytes. Target: < {target_size_bytes} bytes.")

        # Iterative compression using JPEG quality
        current_quality = initial_quality
        for attempt in range(max_attempts):
            if current_quality <= 5: # Don't go too low on quality
                 current_quality = 5

            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=current_quality, optimize=True)
            compressed_bytes = img_byte_arr.getvalue()
            current_size = get_image_size(compressed_bytes)

            # print(f"  Attempt {attempt+1}: Quality {current_quality}, Size {current_size} bytes.") # Reduced verbosity

            if current_size <= target_size_bytes:
                # print(f"  Successfully compressed to {current_size} bytes.")
                return base64.b64encode(compressed_bytes).decode('utf-8')

            current_quality -= quality_step
            if current_quality < 0:
                current_quality = 0 # Ensure quality is not negative

        print(f"  Error: Could not compress image '{image_path}' to target size < {target_size_bytes} bytes after {max_attempts} attempts.")
        # As a last resort, try saving with minimum quality if max_attempts reached but still too big
        if get_image_size(compressed_bytes) > target_size_bytes:
             img_byte_arr = io.BytesIO()
             img.save(img_byte_arr, format='JPEG', quality=0, optimize=True)
             compressed_bytes_min_q = img_byte_arr.getvalue()
             if get_image_size(compressed_bytes_min_q) <= target_size_bytes:
                 print(f"  Compressed with minimum quality (0) to {get_image_size(compressed_bytes_min_q)} bytes.")
                 return base64.b64encode(compressed_bytes_min_q).decode('utf-8')
             else:
                 print(f"  Even with minimum quality (0), size is {get_image_size(compressed_bytes_min_q)} bytes, still too big.")


        return None # Indicate failure to compress

    except FileNotFoundError:
        print(f"  Error: Image file not found at '{full_image_path}'.")
        return None
    except Exception as e:
        print(f"  Error processing image '{image_path}': {e}")
        return None

def load_model_information(file_path):
    """Loads model information from the JSONL file into a dictionary."""
    model_info_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    model_name = data.get("model_name")
                    if model_name:
                        model_info_dict[model_name] = data.get("model_information", {})
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line in model info: {line.strip()} - Error: {e}")
        print(f"Loaded information for {len(model_info_dict)} models.")
        return model_info_dict
    except FileNotFoundError:
        print(f"Error: Model information file '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading model information: {e}")
        return None


def generate_optimization_prompt_v2(model_name, model_info, original_query):
    """
    Generates the prompt for the multi-modal LLM to optimize the query (V2 with stricter output).
    """
    fine_functionality = model_info.get("Fine_functionality", "Not specified")
    description = model_info.get("description", "No description available.")
    domain = model_info.get("Domain", "N/A")
    model_input = model_info.get("input", "N/A")
    model_output = model_info.get("output", "N/A")


    prompt = f"""You are an expert AI assistant specializing in understanding images and user intent for specific AI models.
You are given an image, a description of a target AI model, and an original user query.
Your task is to generate a **new, optimized user query in Chinese** that is:
1.  In **Chinese**.
2.  Highly relevant to the **specific functionality** of the target model ({fine_functionality}).
3.  **Consistent with the content of the provided image.** The query should be a natural question a user would ask *after seeing this image* if they intended to use the target model on it.
4.  **Implies information present in the image** rather than explicitly stating it. For example, if the image is black and white and the model colorizes, ask "能给这张照片加点颜色吗？" instead of "帮我把这张黑白照片上色。"
5.  Your output must contain **ONLY** the single optimized query string. Do not include any other text, punctuation, or formatting (like quotes, bullet points, or introductory phrases). The output should be just the raw query text.

Here is the information about the target model:
Model Name: {model_name}
Functionality: {fine_functionality}
Description: {description}
Domain: {domain}
Input type: {model_input}
Output type: {model_output}

The original user query was: "{original_query}"

Now, look at the provided image. Based on the image content and the target model's function, generate the most appropriate, optimized Chinese user query.
Output the query BELOW this line:
---
"""
    # The image data will be added to the messages list separately
    return prompt

def optimize_query_with_llm_v2(model_name, model_info, original_query, base64_image):
    """
    Calls the multi-modal LLM to optimize the query based on the image (V2 with stricter parsing).
    Returns the optimized query string or None if failed.
    """
    prompt_text = generate_optimization_prompt_v2(model_name, model_info, original_query)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} # Assuming JPEG format after compression
            ],
        }
    ]

    # print(f"  Calling LLM for query optimization for {model_name} with query '{original_query}'...") # Reduced verbosity

    for attempt in range(MAX_RETRIES_LLM_CALL):
        try:
            completion = client.chat.completions.create(
                model=OPTIMIZATION_LLM_MODEL_NAME,
                messages=messages,
                temperature=0.5, # Balance between relevance and slight variation
                max_tokens=100, # Query should be relatively short
                # response_format={"type": "text"} # Explicitly request text format if model supports
            )
            response_content = completion.choices[0].message.content.strip()

            # --- Robust Parsing based on V2 Prompt ---
            # Find the content after the "---" line
            separator = "\n---"
            separator_index = response_content.find(separator)

            if separator_index != -1:
                # Extract content after the separator
                optimized_query = response_content[separator_index + len(separator):].strip()

                # Simple check if the extracted content looks like a query
                if 5 <= len(optimized_query) <= 100: # Basic length check, changed > 5 to >= 5
                     print(f"  Optimized query received (Attempt {attempt+1}): '{optimized_query}'")
                     return optimized_query
                else:
                     print(f"  Warning: Extracted content after separator does not look like a valid query (Attempt {attempt+1}). Content: '{optimized_query}'")

            else:
                # If separator not found, maybe the LLM ignored the format.
                # As a fallback, just take the whole response if it's short and in Chinese
                if 5 <= len(response_content) <= 100: # Basic length check on full response, changed > 5 to >= 5
                    print(f"  Warning: Separator '---' not found in LLM response (Attempt {attempt+1}). Using full response as query: '{response_content}'")
                    return response_content
                else:
                    print(f"  Warning: LLM response format unexpected and separator not found (Attempt {attempt+1}). Raw: '{response_content}'")


        except Exception as e:
            print(f"  Error calling OpenRouter API for query optimization (Attempt {attempt+1}): {e}")

        print(f"  Attempt {attempt+1}/{MAX_RETRIES_LLM_CALL} failed. Retrying in {RETRY_DELAY} seconds...")
        time.sleep(RETRY_DELAY) # Wait before retrying

    print(f"  Failed to get a valid optimized query after {MAX_RETRIES_LLM_CALL} retries for '{original_query}'.")
    return None # Indicate failure

def get_processed_count(file_path):
    """Counts the number of lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0 # File doesn't exist yet, so 0 processed
    except Exception as e:
        print(f"Error counting lines in {file_path}: {e}")
        return 0 # Treat as 0 processed if error occurs


def main():
    """Main function to process queries, optimize with LLM, and save results."""
    if OPENROUTER_API_KEY == "sk-or-v1-YOUR_OPENROUTER_API_KEY_HERE":
        print("ERROR: Please replace 'sk-or-v1-YOUR_OPENROUTER_API_KEY_HERE' with your actual OpenRouter API key in the script.")
        return

    model_information_dict = load_model_information(INPUT_MODEL_INFO_FILE)
    if not model_information_dict:
        print("Could not load model information. Exiting.")
        return

    assigned_queries = []
    try:
        with open(INPUT_ASSIGNED_QUERIES_FILE, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                if not line: continue
                try:
                    assigned_queries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line in assigned queries: {line.strip()} - Error: {e}")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_ASSIGNED_QUERIES_FILE}' not found.")
        return

    if not assigned_queries:
        print("No assigned queries found in input file. Exiting.")
        return

    print(f"Loaded {len(assigned_queries)} assigned queries from '{INPUT_ASSIGNED_QUERIES_FILE}'.")

    # Ensure images folder exists
    if not os.path.exists(IMAGES_FOLDER):
        print(f"Error: Images folder '{IMAGES_FOLDER}' not found. Please create it and place images inside.")
        return

    # --- Determine start position ---
    already_processed_count = get_processed_count(OUTPUT_OPTIMIZED_QUERIES_FILE)
    print(f"Found {already_processed_count} already processed entries in '{OUTPUT_OPTIMIZED_QUERIES_FILE}'.")

    start_index = already_processed_count
    total_entries = len(assigned_queries)

    if start_index >= total_entries:
        print("All entries seem to be already processed. Exiting.")
        return

    print(f"Starting processing from entry index {start_index} out of {total_entries}.")

    processed_count_this_run = 0
    skipped_count_this_run = 0
    total_skipped_including_previous = 0 # Keep track for accurate final count

    # --- Open output file in append mode ---
    with open(OUTPUT_OPTIMIZED_QUERIES_FILE, 'a', encoding='utf-8') as outfile:
        for index, entry in enumerate(assigned_queries):
            # --- Skip already processed entries ---
            if index < start_index:
                # print(f"Skipping entry index {index} (already processed).") # Too verbose
                continue

            # --- Actual processing starts here ---
            model_name = entry.get("model_name")
            original_query = entry.get("query")
            image_path_relative = entry.get("image_path") # This is just the filename, relative to IMAGES_FOLDER

            # Keep track of total skipped for accurate final count
            if not model_name or not original_query or not image_path_relative:
                print(f"\nSkipping incomplete entry: {entry}")
                total_skipped_including_previous += 1
                skipped_count_this_run += 1
                continue

            model_info = model_information_dict.get(model_name)
            # Check for missing info or explicit error marker from previous step
            if not model_info or isinstance(model_info, str) or model_info.get("error"):
                 print(f"\nSkipping entry for model '{model_name}' due to missing or error in model information.")
                 total_skipped_including_previous += 1
                 skipped_count_this_run += 1
                 continue

            print(f"\nProcessing entry {index + 1}/{total_entries}: Model: {model_name}, Original Query: '{original_query}', Image: {image_path_relative}")

            # --- Image Compression and Encoding ---
            # Pass the relative path to the compress function
            base64_image = compress_image_to_target_size(image_path_relative, TARGET_IMAGE_SIZE_BYTES)

            if not base64_image:
                print(f"  Skipping entry due to image processing failure for: {image_path_relative}")
                total_skipped_including_previous += 1
                skipped_count_this_run += 1
                continue

            # --- LLM Call for Query Optimization ---
            optimized_query = optimize_query_with_llm_v2(model_name, model_info, original_query, base64_image)

            if optimized_query:
                output_element = {
                    "model_name": model_name,
                    "query": optimized_query,
                    "image_name": os.path.basename(image_path_relative) # Just the filename
                }
                json_line = json.dumps(output_element, ensure_ascii=False)
                outfile.write(json_line + '\n')
                outfile.flush() # Ensure data is written to file immediately
                processed_count_this_run += 1
                print(f"  Successfully processed and saved optimized query for {model_name} / {image_path_relative}.")
            else:
                print(f"  Skipping entry due to failure in LLM query optimization for {model_name} / {image_path_relative}.")
                total_skipped_including_previous += 1
                skipped_count_this_run += 1

            # --- API Call Delay ---
            # Only add delay if we successfully processed an entry or failed after retries for one entry
            time.sleep(API_CALL_INTERVAL)

    print(f"\nProcessing complete.")
    print(f"Entries processed in this run: {processed_count_this_run}")
    print(f"Entries skipped in this run: {skipped_count_this_run}")
    print(f"Total entries in input file: {total_entries}")
    print(f"Total entries successfully processed (including previous runs): {already_processed_count + processed_count_this_run}")
    # Note: Total skipped might be slightly off if previous runs had incomplete entries counted differently
    print(f"Output appended to '{OUTPUT_OPTIMIZED_QUERIES_FILE}'.")

if __name__ == "__main__":
    main()