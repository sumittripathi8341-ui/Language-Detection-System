from flask import Flask, render_template, request, session, jsonify
import pickle
import re

app = Flask(__name__)
app.secret_key = "secret123"   # required for session

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/", methods=["GET", "POST"])
def home():
    if "history" not in session:
        session["history"] = []

    prediction = None

    if request.method == "POST":
        text = request.form["text"]
        cleaned = clean_text(text)
        vector = cv.transform([cleaned])
        prediction = model.predict(vector)[0]

        # Save to history
        session["history"].append({
            "text": text,
            "prediction": prediction
        })

        session.modified = True

    return render_template(
        "index.html",
        prediction=prediction,
        history=session["history"]
    )


@app.route("/clear", methods=["POST"])
def clear_history():
    session.pop("history", None)
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True)