from flask import Flask, render_template
from flask_restful import Api
from app.controllers.AnalyzerController import AnalyzerController

def create_app():
    app = Flask(__name__)
    api = Api(app)

    # Add resources to API
    api.add_resource(AnalyzerController, '/analyze')

    # Define routes
    @app.route("/")
    def home():
        return render_template(
            "index.html",
            customCSS=["style.css", "color.css"],
            customJS=["analyze.js"]
        )

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)