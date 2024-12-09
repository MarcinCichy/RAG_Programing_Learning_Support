from pdf_processor import extract_text_from_pdf
from rag_engine import create_faiss_index, generate_answer, simplify_text
from sentence_transformers import SentenceTransformer

# Wczytaj tekst z PDF
pdf_text = extract_text_from_pdf("Docs/Code_the_Classics-book.pdf", start_page=82, end_page=114)

# Podziel tekst na fragmenty (np. akapity)
documents = pdf_text.split("\n\n")

# Uprość każdy fragment tekstu
simplified_documents = [simplify_text(doc) for doc in documents]

# Stwórz indeks FAISS na podstawie uproszczonych tekstów
model = SentenceTransformer('all-MiniLM-L6-v2')
index, documents = create_faiss_index(simplified_documents)  # Przekaż listę dokumentów
print(f"Liczba dokumentów w indeksie: {index.ntotal}")

# Komunikacja z użytkownikiem
while True:
    question = input("Zadaj pytanie (lub wpisz 'exit', aby zakonczyć): ")
    if question.lower() == 'exit':
        break

    # Obsługa krótkich pytań
    if len(question.strip()) < 5:  # Jeśli pytanie jest krótkie
        question = "Proszę kontynuować pomoc w tworzeniu gry Frogger."

    # Generowanie odpowiedzi
    answer = generate_answer(question, documents, index, model)
    print("\nOdpowiedź:", answer)