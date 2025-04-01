from datasets import load_dataset
import json

# Load the dataset from Hugging Face
dataset = load_dataset("AlekseyKorshuk/persona-chat")

def extract_pairs(split_data):
    pairs = []
    for dialog in split_data:
        for utterance in dialog['utterances']:
            history = utterance['history']
            if history:
                input_text = history[-1]
                output_text = utterance['candidates'][0]
                pairs.append({ "input": input_text, "output": output_text })
    return pairs

# Extract train and validation pairs
train_pairs = extract_pairs(dataset['train'])
val_pairs = extract_pairs(dataset['validation'])

# Save to files
with open("train_pairs.json", "w", encoding="utf-8") as f:
    json.dump(train_pairs, f, indent=2)

with open("validation_pairs.json", "w", encoding="utf-8") as f:
    json.dump(val_pairs, f, indent=2)

print("✅ Saved train_pairs.json and validation_pairs.json")
