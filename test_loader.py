from pdf_loader import PDFLoader

loader = PDFLoader()

documents = loader.load_pdfs()

print("Total Pages :", len(documents))

print()

print(documents[0])