from rag_engine import create_faiss_index, preprocess_documents, generate_answer
from sentence_transformers import SentenceTransformer

# Ścieżka do pliku PDF
PDF_PATH = "Docs/Code_the_Classics-book.pdf"

# Przetwórz dokumenty
simplified_documents = preprocess_documents(PDF_PATH, start_page=82, end_page=114)

# Stwórz indeks FAISS na podstawie uproszczonych dokumentów
index, simplified_documents = create_faiss_index(simplified_documents)
print(f"Liczba dokumentów w indeksie: {index.ntotal}")

# Interfejs konsolowy
def run_console_interface():
    while True:
        question = input("Zadaj pytanie (lub wpisz 'exit', aby zakończyć): ")
        if question.lower() == 'exit':
            break
        if len(question.strip()) < 5:
            question = "Kontynuujmy tworzenie gry Frogger. Co powinienem zrobić dalej?"
        answer = generate_answer(question, simplified_documents, index, SentenceTransformer('all-MiniLM-L6-v2'))
        print("\nOdpowiedź:", answer)

if __name__ == "__main__":
    run_console_interface()
