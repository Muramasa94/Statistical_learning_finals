from flask_restful import Resource
from flask import request
from app.service.AnalyzerService import EmotionAnalyzer

class AnalyzerController(Resource):
    def __init__(self):
        self.analyzer = EmotionAnalyzer()
        
    def post(self):
        data = request.get_json()
        text = data.get('text', '')
        
        # Split the text into sentences
        sentences = self.analyzer.split_text(text)
        
        # Analyze emotions for each sentence
        results = self.analyzer.analyze(sentences)
        
        return {'results': results}, 200