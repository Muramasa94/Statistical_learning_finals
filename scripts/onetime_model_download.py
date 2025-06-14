from transformers import AutoModel, AutoTokenizer
import os

MODEL_NAME = "distilbert/distilbert-base-uncased"
SAVE_DIR = "./models/distilbert-base-uncased"

def main():
    # Ensure the directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"⏬ Downloading tokenizer and model from: {MODEL_NAME}")

    # Load and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.save_pretrained(SAVE_DIR)
    print("✅ Tokenizer saved.")

    # Load and save model
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.save_pretrained(SAVE_DIR)
    print("✅ Model saved.")

    # Test loading the saved model
    _ = AutoTokenizer.from_pretrained(SAVE_DIR)
    _ = AutoModel.from_pretrained(SAVE_DIR)
    print("🎉 Model and tokenizer successfully saved to ./models/distilbert-base-uncased")

if __name__ == "__main__":
    main()