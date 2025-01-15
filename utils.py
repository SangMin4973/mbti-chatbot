from langchain.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdf(pdf_path, chunk_size=200, chunk_overlap=50):
    loader = PyPDFium2Loader(pdf_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_documents(data)

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])