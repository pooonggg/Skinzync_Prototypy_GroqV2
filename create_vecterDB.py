from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import fitz
import os
import json

from langchain.retrievers import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate

# ไดเรกทอรีที่เก็บไฟล์ PDF
pdf_directory = "pdf/"

# สร้างคลาสเพื่อให้แต่ละหน้าของ PDF มี .page_content และ .metadata
class PDFPage:
    def __init__(self, page_number, text, metadata=None):
        self.page_number = page_number
        self.page_content = text  # เก็บข้อความใน .page_content
        self.metadata = metadata or {}  # เก็บข้อมูลเพิ่มเติมใน .metadata

# อ่านไฟล์ PDF ด้วย PyMuPDF
def load_pdf_with_pymupdf(file_path):
    doc = fitz.open(file_path)
    data = []
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # โหลดหน้าจากไฟล์ PDF
        text = page.get_text("text")  # ดึงข้อความออกมาในรูปแบบของ text
        data.append(PDFPage(page_num + 1, text, metadata={"file_name": os.path.basename(file_path)}))  # เพิ่ม object PDFPage ลงใน list
    
    doc.close()
    return data

# อ่านไฟล์ PDF ทั้งหมดในไดเรกทอรี pdf/
def load_all_pdfs_in_directory(directory):
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    all_pdfs_data = {}

    for pdf_file in pdf_files:
        file_path = os.path.join(directory, pdf_file)
        pdf_data = load_pdf_with_pymupdf(file_path)  # อ่านข้อมูลจาก PDF
        all_pdfs_data[pdf_file] = pdf_data  # เก็บข้อมูลของแต่ละไฟล์ใน dict
    
    return all_pdfs_data

# อ่านไฟล์ PDF ทั้งหมดในไดเรกทอรี pdf/
if os.path.exists(pdf_directory):
    all_pdfs_data = load_all_pdfs_in_directory(pdf_directory)

    # สร้าง chunks จากข้อความที่อ่านได้
    text_documents = [page for pdf_data in all_pdfs_data.values() for page in pdf_data]  # Use PDFPage objects directly

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=500)
    chunks = text_splitter.split_documents(text_documents)

    print(f"Number of chunks created: {len(chunks)}")

else:
    print("Directory not found")


persist_directory = "vector_db/"
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    persist_directory=persist_directory,
    collection_name="Building-Block"
)
vector_db.persist()