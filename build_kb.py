import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

folder_path = r"C:\Users\niran\Discord Bot\llm_knowledge"
file_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith(".txt")]

all_documents = []
for path in file_paths:
    loader = TextLoader(path, encoding='utf-8')
    all_documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(all_documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.from_documents(docs, embeddings)
db.save_local("knowledge_base")

print("✅ Knowledge base index created from multiple .txt files.")
