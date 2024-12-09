import PyPDF2


def extract_text_from_pdf(pdf_path, start_page, end_page):
    """ Funkcja wczytująca tekst z pliku PDF. """
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        # Przetwarzaj tylko strony w podanym zakresie
        for page_num in range (start_page-1, end_page):
            text += reader.pages[page_num].extract_text()
        return text
