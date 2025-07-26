from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

EMOTIONS = ['admiration', 'amusement', 'anger', 'annoyance', 'approval',
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness',
            'optimism', 'pride', 'realization', 'relief', 'remorse',
            'sadness', 'surprise', 'neutral']

class EmotionAnalyzer:
    def __init__(self):
        model_dir = 'app/model/trained_emotion_model'
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        
    def split_text(self, text): # Split text into sentences based on common delimiters like ".", ";", "!", "?"
        # Normalize whitespace
        # Replace mid-sentence line breaks with spaces
        text = re.sub(r'(?<![.!?])\n(?!\n)', ' ', text)
        # Normalize spaces after punctuation before newline
        text = re.sub(r'([.!?]) +\n(?!\n)', r'\1\n', text)
        # Normalize paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Split sentences using regex
        sentences = re.split(r'[.!?;]', text)
        return [s.strip() for s in sentences if s.strip()]

    def analyze(self, texts):
        results = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits).cpu().detach().numpy()
            emotion_scores = {emotion: round(float(prob), 4) for emotion, prob in zip(EMOTIONS, probabilities[0])}
            results.append({
                'text': text,
                'emotions': emotion_scores
            })
        return results