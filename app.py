import json
from flask import Flask, render_template, request, jsonify
from rag_engine import generate_answer
from sentence_transformers import SentenceTransformer

# Inicjalizacja modelu i danych
from main import simplified_documents, index

app = Flask(__name__)

# Ścieżka do pliku z zapisanymi konwersacjami
CONVERSATIONS_FILE = "conversations.json"

# Wczytaj zapisane konwersacje
def load_conversations():
    try:
        with open(CONVERSATIONS_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

# Zapisz konwersacje do pliku
def save_conversations():
    with open(CONVERSATIONS_FILE, "w") as file:
        json.dump(conversations, file, indent=4)

# Inicjalizacja globalnych zmiennych
conversations = load_conversations()
current_conversation_id = None

@app.route("/")
def home():
    """Strona główna."""
    return render_template("index.html", conversations=list(conversations.keys()))

@app.route("/new_conversation", methods=["POST"])
def new_conversation():
    """Rozpocznij nową konwersację."""
    global current_conversation_id
    current_conversation_id = str(len(conversations) + 1)
    conversations[current_conversation_id] = []
    save_conversations()
    return jsonify({"conversation_id": current_conversation_id, "conversations": list(conversations.keys())})

@app.route("/load_conversation", methods=["POST"])
def load_conversation():
    """Załaduj istniejącą konwersację."""
    global current_conversation_id
    conversation_id = request.json.get("conversation_id")
    current_conversation_id = conversation_id
    messages = conversations.get(conversation_id, [])
    return jsonify({"conversation_id": conversation_id, "messages": messages})

@app.route("/delete_conversation", methods=["POST"])
def delete_conversation():
    """Usuń istniejącą konwersację."""
    global conversations
    conversation_id = request.json.get("conversation_id")
    if conversation_id in conversations:
        del conversations[conversation_id]
        save_conversations()
    return jsonify({"status": "deleted", "conversation_id": conversation_id, "conversations": list(conversations.keys())})

@app.route("/ask", methods=["POST"])
def ask_question():
    """Obsługa pytań użytkownika."""
    global current_conversation_id
    user_question = request.form["question"]

    # Generowanie odpowiedzi
    answer = generate_answer(user_question, simplified_documents, index, SentenceTransformer('all-MiniLM-L6-v2'))

    # Dodaj wiadomość do aktualnej konwersacji
    if current_conversation_id:
        conversations[current_conversation_id].append({"user": user_question, "bot": answer})
        save_conversations()

    return jsonify({"question": user_question, "answer": answer})

@app.route("/reset", methods=["POST"])
def reset_chat():
    """Resetowanie aktualnej rozmowy."""
    global current_conversation_id
    if current_conversation_id in conversations:
        conversations[current_conversation_id] = []
        save_conversations()
    return jsonify({"status": "reset"})


@app.route("/exit", methods=["POST"])
def exit_app():
    """Zamykanie aplikacji z podsumowaniem."""
    summary = {
        "message": "Aplikacja została zamknięta. Dziękujemy za korzystanie z RAG!",
        "conversation_count": len(conversations),
        "questions_asked": sum(len(conv) for conv in conversations.values()),
    }

    # Wysyłanie odpowiedzi do przeglądarki
    response = jsonify(summary)
    response.status_code = 200

    # Wyłączenie serwera po wysłaniu odpowiedzi
    def shutdown_server():
        func = request.environ.get('werkzeug.server.shutdown')
        if func:
            func()
    shutdown_server()

    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5010)
