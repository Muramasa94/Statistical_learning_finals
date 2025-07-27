from flask_restful import Resource
from flask import request
from app.service.AnalyzerService import EmotionAnalyzer

class AnalyzerController(Resource):
    def __init__(self):
        self.analyzer = EmotionAnalyzer()
        
    def post(self):
        data = request.get_json()
        text = data.get('text', '')
        
        # Error handling for empty text
        if not text or text.strip() == '':
            return {'error': 'No text provided for analysis'}, 400
        # Validate text type
        if not isinstance(text, str):
            return {'error': 'Text must be a string'}, 400
        
        # Split the text into sentences
        sentences = self.analyzer.split_text(text)
        
        # Analyze emotions for each sentence
        results_per_sentence, overall_emotions = self.analyzer.analyze(sentences)

        return {'per_sentence': results_per_sentence, 'overall': overall_emotions}, 200