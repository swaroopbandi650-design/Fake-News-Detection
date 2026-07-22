from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the TF-IDF vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# Clean input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    news_text = ""

    if request.method == "POST":

        news_text = request.form["news"]

        cleaned = clean_text(news_text)

        vector = vectorizer.transform([cleaned])

        result = model.predict(vector)

        if result[0] == 0:
            prediction = "❌ Fake News"
        else:
            prediction = "✅ Real News"

    return render_template(
        "index.html",
        prediction=prediction,
        news_text=news_text
    )


if __name__ == "__main__":
    app.run(debug=True)