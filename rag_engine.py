import openai
import tiktoken
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
import os
from pdf_processor import extract_text_from_pdf

# Załaduj klucz API
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Funkcja do uproszczenia tekstu
def simplify_text(text, model="gpt-3.5-turbo"):
    """Upraszcza tekst dla dziecka uczącego się Pythona."""
    simplify_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "Simplify the following text for a 13-year-old child learning Python programming. "
                "Use simple language and clear examples where appropriate."
            )},
            {"role": "user", "content": text}
        ]
    )
    return simplify_response['choices'][0]['message']['content']

# Funkcja do liczenia tokenów
def count_tokens(text, model="gpt-3.5-turbo"):
    tokenizer = tiktoken.encoding_for_model(model)
    tokens = tokenizer.encode(text)
    return len(tokens)

# Tworzenie FAISS
def create_faiss_index(documents):
    """Tworzy indeks FAISS na podstawie listy dokumentów."""
    sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = sentence_transformer_model.encode(documents)  # Wygeneruj osadzenia dla każdego dokumentu
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # Dodaj osadzenia do indeksu
    return index, documents

# Wczytaj i przetwórz dokumenty
def preprocess_documents(pdf_path, start_page, end_page):
    """Wczytuje i przetwarza dokumenty z pliku PDF."""
    pdf_text = extract_text_from_pdf(pdf_path, start_page, end_page)
    documents = pdf_text.split("\n\n")
    simplified_documents = [simplify_text(doc) for doc in documents]
    return simplified_documents

# Generowanie odpowiedzi
def generate_answer(question, documents, index, model):
    """Generuje odpowiedź na pytanie, korzystając z FAISS i OpenAI API."""

    # Sprawdź, czy pytanie jest puste
    if len(question.strip()) == 0:
        return "Proszę zadać pytanie."

    # Prześlij pytanie i pozwól modelowi odpowiedzieć w ramach swojego kontekstu
    query_embedding = model.encode([question])
    distances, indices = index.search(query_embedding, k=2)  # Szukaj maksymalnie 2 dopasowań

    # Jeśli FAISS nie znajduje wyników, zwróć brak odpowiedzi
    if len(indices[0]) == 0 or indices[0][0] == -1:
        return "Nie znalazłem odpowiedzi w mojej bazie wiedzy. Spróbuj zadać inne pytanie."

    # Ogranicz długość fragmentów kontekstu
    max_context_length = 500
    context = " ".join([documents[i][:max_context_length] for i in indices[0] if i >= 0])

    # Generowanie odpowiedzi
    answer_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": (
                "Jesteś przyjaznym nauczycielem programowania, który pomaga 13-letniemu dziecku "
                "z podstawową wiedzą o Pythonie stworzyć grę Frogger. Wszystkie odpowiedzi udzielaj "
                "tylko w języku polskim. Wyjaśniaj prosto, krok po kroku, i używaj przykładów kodu. W przypadku pytań nie związanych z tworzeniem gier w Pythonie, odpowiadaj na nie w sposób ogólny oraz zaznacz, że nie jest związane z programowaniem gier."
            )},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )
    return answer_response['choices'][0]['message']['content']
