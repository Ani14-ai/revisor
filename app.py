from flask import Flask, request, jsonify, abort, render_template
from flask_cors import CORS
import pyodbc
from dotenv import load_dotenv
import os
import logging
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
load_dotenv()
app = Flask(__name__)
CORS(app, resources={"/api/*": {"origins": "*"}})
CANADA_TZ = timezone(timedelta(hours=-5))
DB_CONNECTION_STRING = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=103.239.89.99,21433;DATABASE=MyRevisorAppDB;UID=MyRevisorAppUsr;PWD=MyRetr435*$h8"
VECTOR_STORE_DIR = "persistent_vector_store"
logging.basicConfig(level=logging.INFO)
executor = ThreadPoolExecutor(max_workers=4)
vector_store = None
def initialize_vector_store():
    """Initialize the vector store from the persistent directory, if it exists."""
    global vector_store
    if os.path.exists(VECTOR_STORE_DIR):
        vector_store = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=OpenAIEmbeddings())
    else:
        vector_store = None

def save_vector_store(document_chunks):
    """Save the vector store after creating it from document chunks."""
    global vector_store
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings(), persist_directory=VECTOR_STORE_DIR)
    vector_store.persist()

def load_document_chunks(file_path):
    """Load and split documents to handle large files and multiple websites."""
    loader = PyMuPDFLoader(file_path)
    site1 = WebBaseLoader("https://myrevisor.waysdatalabs.com/")
    site2 = WebBaseLoader("https://myrevisor.waysdatalabs.com/index#aboutus")
    site3 = WebBaseLoader("https://myrevisor.waysdatalabs.com/index#properties")
    site4 = WebBaseLoader("https://myrevisor.waysdatalabs.com/index#whychooseus")
    site5 = WebBaseLoader("https://myrevisor.waysdatalabs.com/index#ourteam")
    site6 = WebBaseLoader("https://myrevisor.waysdatalabs.com/Blogs")
    site7 = WebBaseLoader("https://myrevisor.waysdatalabs.com/ContactUs")

    document1 = loader.load()
    document2 = site1.load() + site2.load() + site3.load() + site4.load() + site5.load() + site6.load() + site7.load()
    document = document1 + document2

    text_splitter = RecursiveCharacterTextSplitter()  # Adjusted chunk size
    document_chunks = text_splitter.split_documents(document)
    return document_chunks

def get_context_retriever_chain(session_id):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Based on the conversation, generate a search query to find relevant information.")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are AMI L28u, a chatbot for Revisor, an enabler for real estate buyers and developers to connect, review potential pre-construction investment opportunities, and facilitate the completion of transactions. You help users find information related to Revisor, the Canadian real estate market, and attract more customers for Revisor. You answer their queries in complete sentences, within 100 tokens, in a precise and short manner based on the context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input, session_id, start_time):
    retriever_chain = get_context_retriever_chain(session_id)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    chat_history = load_chat_history(session_id)
    formatted_chat_history = [
        {"role": "user", "content": entry["user_input"]} if i % 2 == 0 else {"role": "assistant", "content": entry["bot_response"]}
        for i, entry in enumerate(chat_history)
    ]

    response = conversation_rag_chain.invoke({
        "chat_history": formatted_chat_history,
        "input": user_input
    })

    save_chat_history(session_id, user_input, response['answer'], start_time)
    return response['answer']

def log_api_call(endpoint, status_code, response_time):
    """Log API call details to the database."""
    connection = pyodbc.connect(DB_CONNECTION_STRING)
    cursor = connection.cursor()
    cursor.execute(
        "INSERT INTO tbChatbot_APILog (api_endpoint, status_code, response_time, timestamp) VALUES (?, ?, ?, ?)",
        (endpoint, status_code, response_time, datetime.now(CANADA_TZ))
    )
    connection.commit()
    connection.close()

def authenticate_api_key(api_key):
    """Check if the provided API key is valid and active."""
    connection = pyodbc.connect(DB_CONNECTION_STRING)
    cursor = connection.cursor()
    cursor.execute("SELECT is_active FROM tbChatbot_APIkey WHERE api_key = ?", (api_key,))
    result = cursor.fetchone()
    connection.close()
    return result and result[0]

def save_chat_history(session_id, user_input, bot_response, start_time):
    """Save user input and bot response to the database."""
    connection = pyodbc.connect(DB_CONNECTION_STRING)
    cursor = connection.cursor()
    response_time = (datetime.now(CANADA_TZ) - start_time).total_seconds()
    cursor.execute(
        "INSERT INTO tbChatbot (session_id, user_input, bot_response, response_time, timestamp) VALUES (?, ?, ?, ?, ?)",
        (session_id, user_input, bot_response, response_time, datetime.now(CANADA_TZ))
    )
    connection.commit()
    connection.close()

def load_chat_history(session_id):
    """Load chat history for a given session ID."""
    connection = pyodbc.connect(DB_CONNECTION_STRING)
    cursor = connection.cursor()
    cursor.execute("SELECT user_input, bot_response FROM tbChatbot WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
    chat_history = cursor.fetchall()
    connection.close()

    # Convert each row to a dictionary
    chat_history_list = [{"user_input": row.user_input, "bot_response": row.bot_response} for row in chat_history]
    return chat_history_list

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/<path:path>')
def serve_page(path):
    return render_template(path)

@app.route('/api/upload_doc', methods=['POST'])
def upload_pdf():
    start_time = datetime.now(CANADA_TZ)
    try:
        if 'file' not in request.files:
            log_api_call('/api/upload_doc', 400, (datetime.now(CANADA_TZ) - start_time).total_seconds())
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            log_api_call('/api/upload_doc', 400, (datetime.now(CANADA_TZ) - start_time).total_seconds())
            return jsonify({"error": "No selected file"}), 400

        if file:
            file_path = f"temp_{file.filename}"
            file.save(file_path)
            executor.submit(async_load_and_save, file_path)
            log_api_call('/api/upload_doc', 200, (datetime.now(CANADA_TZ) - start_time).total_seconds())
            return jsonify({"message": "PDF is being processed. Please check back later."})

    except Exception as e:
        log_api_call('/api/upload_doc', 500, (datetime.now(CANADA_TZ) - start_time).total_seconds())
        logging.error(f"Error during PDF upload: {e}")
        return jsonify({"error": str(e)}), 500

def async_load_and_save(file_path):
    try:
        document_chunks = load_document_chunks(file_path)
        save_vector_store(document_chunks)
    finally:
        os.remove(file_path)

@app.route('/api/ask', methods=['POST'])
def ask_question():
    start_time = datetime.now(CANADA_TZ)
    api_key = request.headers.get('Authorization')
    if not api_key or not authenticate_api_key(api_key):
        log_api_call('/api/ask', 403, (datetime.now(CANADA_TZ) - start_time).total_seconds())
        abort(403, description="Invalid or missing API key")

    data = request.get_json()
    if 'question' not in data or 'session_id' not in data:
        log_api_call('/api/ask', 400, (datetime.now(CANADA_TZ) - start_time).total_seconds())
        return jsonify({"error": "Missing required fields"}), 400

    question = data['question']
    session_id = data['session_id']

    try:
        response = get_response(question, session_id, start_time)
        log_api_call('/api/ask', 200, (datetime.now(CANADA_TZ) - start_time).total_seconds())
        return jsonify({"response": response})
    except Exception as e:
        log_api_call('/api/ask', 500, (datetime.now(CANADA_TZ) - start_time).total_seconds())
        logging.error(f"Error during question processing: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    initialize_vector_store()
    app.run(debug=False)
