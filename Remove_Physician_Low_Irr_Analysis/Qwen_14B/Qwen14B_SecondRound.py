import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_id = "Qwen/Qwen-14B"
print(f"Loading model {model_id}...")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print("Model loaded successfully")

# Load data
df = pd.read_csv("/data/healthy-ml/scratch/yuexing/NeuRIPS25/After_Physician_Removal/Qwen_14B/High_Low_Irr_Data_Source_.csv")
print("Columns in dataset:")
print(df.columns.tolist())

# Function to extract answer letter using multiple patterns
def extract_answer_letter(text):
    if pd.isna(text) or not text:
        return None
    
    # Try different patterns to extract the answer letter
    patterns = [
        r"Answer:\s*([A-J])",             # "Answer: A"
        r"Answer is\s*([A-J])",           # "Answer is A"
        r"answer is\s*([A-J])",           # "answer is A"
        r"The answer is\s*([A-J])",       # "The answer is A"
        r"the answer is\s*([A-J])",       # "the answer is A"
        r"Option\s*([A-J])",              # "Option A"
        r"option\s*([A-J])",              # "option A"
        r"My answer is\s*([A-J])",        # "My answer is A"
        r"(\n|^)([A-J])\.?\s*$",          # "A." or just "A" at end or newline
        r"select option\s*([A-J])",       # "select option A"
        r"I select\s*([A-J])",            # "I select A"
        r"I choose\s*([A-J])",            # "I choose A"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            # Some patterns have the letter in group 1, others in group 2
            return match.group(1) if len(match.groups()) == 1 else match.group(2)
    
    # If no match found, check if there's a single letter at the end
    words = text.strip().split()
    if words and len(words[-1]) == 1 and words[-1].isalpha() and words[-1].upper() in "ABCDEFGHIJ":
        return words[-1].upper()
    
    return None

# Create results dataframe
results = []

# Loop through the dataset
total_rows = min(4054, len(df))
print(f"Processing {total_rows} rows...")

for idx, row in df.head(total_rows).iterrows():
    print(f"Processing row {idx+1}/{total_rows}...")
    try:
    context_text = row["High"]
        question = row["question_options"]
        
        # Improved prompt with clearer instructions
        query_full = (
            f"{context_text}\n\n"
            f"{question}\n\n"
            "Based on the information provided, select the correct answer choice (A, B, C, D, etc.).\n\n"
            "IMPORTANT: Your response must end with 'Answer: X' where X is the letter of your chosen option.\n"
            "For example: Answer: A"
    )
    
        # Generate prediction using the model
        inputs = tokenizer(query_full, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
        )

        # Decode the generated response
        raw_response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Extract the answer letter using improved function
        extracted_answer = extract_answer_letter(raw_response)
        
        # If still no answer found, log more details for debugging
        if extracted_answer is None:
            print(f"⚠️ Could not extract answer from response for row {idx+1}:")
            print(f"Response: {raw_response[:100]}...")
        
        # Create result entry
            qa_id = f"Merge Q{idx + 1}"
        result_entry = {
            "QA_ID": qa_id,
            "Origin": row.get("Origin", ""),
            "data_source": row.get("data_source_df3", ""),
            "Raw_Response": raw_response,
            "Extracted_Answer": extracted_answer
        }
        
        results.append(result_entry)
        print(f"✅ Processed {qa_id}: Answer = {extracted_answer}")
        
        # Save progress every 10 items (increased frequency for safety)
        if (idx + 1) % 10 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv("Qwen_14B_predictions_progress.csv", index=False)
            print(f"Saved progress to CSV after {idx+1} items")

    except Exception as e:
        print(f"❌ Error on row {idx}: {str(e)}")
        # Still try to save the entry with error info
        qa_id = f"Merge Q{idx + 1}"
        results.append({
            "QA_ID": qa_id,
            "Origin": row.get("Origin", ""),
            "data_source": row.get("data_source_df3", ""),
            "Raw_Response": f"ERROR: {str(e)}",
            "Extracted_Answer": None
        })

# Save final results
output_df = pd.DataFrame(results)
output_file = "Qwen_14B_predictions_final.csv"
output_df.to_csv(output_file, index=False)
print(f"Saved all predictions to {output_file}")
