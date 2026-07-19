from flask import (Flask, render_template, send_from_directory, request)
from poke_predict import predict_pokemon
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory("uploads", filename)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["image"]
    filepath = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(filepath)

    pokemon, confidence, top5 = predict_pokemon(filepath)

    return render_template(
        "result.html",
        pokemon = pokemon,
        confidence = round(confidence, 2),
        top5 = top5,
        image_path = f"/uploads/{image.filename}"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)