from flask import Flask, render_template, request, jsonify
from rag_engine import generate_answer
from sentence_transformers import SentenceTransformer

# Inicjalizacja modelu i danych
from main import simplified_documents, index  # Import uproszczonych dokumentów i indeksu

# Tworzenie aplikacji Flask
app = Flask(__name__)

# Zmienna globalna do przechowywania historii rozmowy
chat_history = []


@app.route("/")
def home():
    """Strona główna."""
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask_question():
    """Obsługa pytań użytkownika."""
    user_question = request.form["question"]
    if len(user_question.strip()) < 5:  # Obsługa krótkich pytań
        user_question = "Proszę kontynuować pomoc w tworzeniu gry Frogger."

    # Generowanie odpowiedzi
    answer = generate_answer(user_question, simplified_documents, index, SentenceTransformer('all-MiniLM-L6-v2'))
    chat_history.append({"user": user_question, "rag": answer})
    return jsonify({"question": user_question, "answer": answer})


@app.route("/reset", methods=["POST"])
def reset_chat():
    """Resetowanie historii rozmowy."""
    global chat_history
    chat_history = []
    return jsonify({"status": "reset"})


@app.route("/exit", methods=["POST"])
def exit_app():
    """Zamykanie aplikacji."""
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    return jsonify({"status": "exited"})


if __name__ == "__main__":
    app.run(debug=True)
