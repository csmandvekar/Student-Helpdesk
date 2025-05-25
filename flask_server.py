from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI

from PyPDF2 import PdfReader
import os
import pickle
import json

from flask_cors import CORS

allow_dangerous_deserialization = True

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS globally


# OpenAI API key

API_KEY = os.getenv(API_KEY)
# Set up vector database
def create_vector_store(pdf_folder="pdfs"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

    # Read all PDFs and convert to text
    texts = []
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file_name)
            reader = PdfReader(pdf_path)
            pdf_text = "".join([page.extract_text() for page in reader.pages])
            chunks = text_splitter.split_text(pdf_text)
            texts.extend(chunks)

    # Create FAISS vector store
    vector_store = FAISS.from_texts(texts, embeddings)
    vector_store.save_local("faiss_index")
    
    
    
    return vector_store
    

def load_vector_store():
    if allow_dangerous_deserialization:
        try:
            with open("faiss_index.pkl", "rb") as file:
                vector_store = pickle.load(file)
                return vector_store
        except FileNotFoundError:
            return create_vector_store()
    else:
        raise ValueError("Dangerous deserialization is not allowed. Exiting.")
def load_vector_store():
    if allow_dangerous_deserialization:
        embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)  # Use updated embedding class
        return FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    else:
        raise ValueError("Dangerous deserialization is not allowed. Exiting.")


# Load or create vector database on startup
pdf_path = "/pdfs/datafile1.pdf"
if os.path.exists("faiss_index"):
    vector_store = FAISS.load_local("faiss_index", OpenAIEmbeddings(openai_api_key=API_KEY),allow_dangerous_deserialization=True)
else:
    vector_store = create_vector_store(pdf_path,allow_dangerous_deserialization=True)
# vector_store = load_vector_store(allow_dangerous_deserialization=True)

@app.route("/query", methods=["POST"])
def query():
    print("Enter query")
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    # Perform similarity search and answer the question
    docs = vector_store.similarity_search(user_question)
    print("Docs is working ",user_question)
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=API_KEY)
    print("llm is working ")
    
    context = "\n".join([doc.page_content for doc in docs])
    
    messages = [
    (
        "system",
        "Use the following context to answer the question: {context}.",
    ),
    ("human",user_question),
]
    
    print("pritnitng context",context)
    chat_prompt = ChatPromptTemplate.from_messages(messages)

# Format the prompt with context and question
    formatted_prompt = chat_prompt.format(context=context, question=user_question)

# Use the LLM to get a response
    response = llm.invoke(formatted_prompt)

    print("Response is", response)
    return jsonify({"answer": response.content})
   
if __name__ == "__main__":
    app.run(debug=True, port=5000)
    

