# Emotion Analysis Web Application

A web application for fine-grained emotion analysis of text using a transformer-based NLP model. The app (tries to) predicts 27 emotion categories (plus neutral) from user input, leveraging a model trained on the GoEmotions dataset.

---

## Tech Stack

- **Backend:** Python, Flask, Flask-RESTful
- **NLP Model:** HuggingFace Transformers (`distilbert-base-uncased` fine-tuned on GoEmotions)
- **Frontend:** HTML, Bootstrap, JavaScript
- **Data:** [GoEmotions](https://www.kaggle.com/datasets/debarshichanda/goemotions) (Reddit comments, 27 emotions + neutral)
- **Other:** PyTorch, NumPy, Pandas

---

## Features

- Sentence-level and overall emotion prediction
- Interactive web UI for text input and result visualization
- REST API for programmatic access

---

## Installation

1. **Clone the repository**
   ```sh
   git clone https://github.com/Muramasa94/Statistical_learning_finals.git
   cd Statistical_learning_finals
   ```

2. **Set up Python environment**
   - Python 3.12 recommended
   - (Optional) Create a virtual environment:
     ```sh
     python -m venv .venv
     source .venv/bin/activate  # or .venv\Scripts\activate on Windows
     ```

3. **Install dependencies**
   ```sh
   pip install ipykernel
   pip install torch==2.3.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
   pip install -r requirements.txt
   ```

4. **Download/prepare the model**
   - Download trained model folder at `https://drive.google.com/drive/folders/1NmPrm5Kpm2sCWrlVH4Pb_z7RxhA5QzV8?usp=sharing`
   - Place the model folder in the `app/model` folder, so that the directory looks like `app/model/trained_emotion_model/<6_files_in_model_folder>`.
   - If you want to re-run the report notebook, also put the model folder in the `models` directory.
   - To download a base model, you can use [`scripts/onetime_model_download.py`](scripts/onetime_model_download.py).

5. **Run the application**
   ```sh
   python run.py
   ```
   - The app will be available at `http://localhost:5000`.

---

## Usage

- Open your browser and go to `http://localhost:5000`.
- Enter or paste text in the input box and click "Analyze Emotions".
- View overall emotion breakdowns, or hover on a sentence to view its individual analysis.

---

## Project Structure

```
app/
  controllers/         # Flask REST API controllers
  model/               # Trained transformer model
  service/             # Emotion analysis logic
  static/              # Frontend JS/CSS
  templates/           # HTML templates
data/                  # GoEmotions and processed CSVs
models/                # Downloaded base model and trained model
scripts/               # Data preprocessing and model download scripts
run.py                 # App entry point
requirements.txt       # Python dependencies
Readme.md              # You are here
report.ipynb           # Report containing model evaluation details
```

---

## Transformer Model

- **Base:** `distilbert-base-uncased` ([HuggingFace link](https://huggingface.co/distilbert/distilbert-base-uncased))
- **Fine-tuned:** On an expanded version of GoEmotions dataset (27 emotions + neutral)
- **Framework:** HuggingFace Transformers, PyTorch

---

## References

- [GoEmotions Dataset](https://www.kaggle.com/datasets/debarshichanda/goemotions)
- [DistilBERT Model](https://huggingface.co/distilbert/distilbert-base-uncased)
- See [`report.ipynb`](report.ipynb) for training and evaluation details.

---

## License

This project is for educational purposes.