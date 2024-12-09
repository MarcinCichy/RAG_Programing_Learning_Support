import openai
import tiktoken
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
import os

# Załaduj klucz API
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def simplify_text(text, model="gpt-3.5-turbo"):
    """Upraszcza tekst dla dziecka uczącego się Pythona."""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "Simplify the following text for a 13-year-old child learning Python programming. "
                "Use simple language and clear examples where appropriate."
            )},
            {"role": "user", "content": text}
        ]
    )
    return response['choices'][0]['message']['content']


# Funkcja do liczenia tokenów
def count_tokens(text, model="gpt-3.5-turbo"):
    tokenizer = tiktoken.encoding_for_model(model)
    tokens = tokenizer.encode(text)
    return len(tokens)

# Tworzenie FAISS
def create_faiss_index(documents):
    """Tworzy indeks FAISS na podstawie listy dokumentów."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents)  # Wygeneruj osadzenia dla każdego dokumentu
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # Dodaj osadzenia do indeksu
    return index, documents

# Generowanie odpowiedzi
def generate_answer(question, documents, index, model):
    query_embedding = model.encode([question])
    distances, indices = index.search(query_embedding, k=2)  # Szukaj maksymalnie 2 dopasowań

    print(f"Distances: {distances}")
    print(f"Indices: {indices}")

    # Obsługa braku wyników
    if len(indices[0]) == 0 or indices[0][0] == -1:
        return "Nie znalazłem odpowiedzi w mojej bazie wiedzy. Spróbuj zadać inne pytanie."

    # Ogranicz długość fragmentów kontekstu
    max_context_length = 500
    context = " ".join([documents[i][:max_context_length] for i in indices[0] if i >= 0])

    # Generowanie odpowiedzi
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": (
                "Jesteś przyjaznym nauczycielem programowania, który pomaga 13-letniemu dziecku "
                "z podstawową wiedzą o Pythonie stworzyć grę Frogger. Wszystkie odpowiedzi udzielaj "
                "tylko w języku polskim. Wyjaśniaj prosto, krok po kroku, i używaj przykładów kodu. "
                "Bądź zachęcający i unikaj używania zbyt technicznych terminów."
            )},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )
    return response['choices'][0]['message']['content']
