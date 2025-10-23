import os
import time
import json # Import JSON for parsing
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- LLM and API Key Setup ---
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file or as an environment variable.")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")


def load_retrieval_chain():
    """
    Loads the vector database and creates the retrieval chain.
    This function runs once when the server starts.
    """
    print("Loading vector database... This may take a moment.")
    
    # --- PROMPT TEMPLATE - Reverted to simple stateless version ---
    prompt_template = """
You are a friendly and helpful hotel assistant.
Your role is to provide clear, welcoming, and professional responses to guest questions.
You MUST respond in a valid JSON format.

The JSON object must have two keys:
1. "intent": (string) This will always be "qa" for this version.
2. "response": (string) Your natural language, conversational response to the user.

RULES:
- Base your answers ONLY on the provided context. If the information is not in the context,
  politely say "I'm sorry, I don't have that information, but I can connect you with our front desk for assistance."
- Do not make up information.

<context>
{context}
<context>

Question: {input}

Your JSON Response:
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    loader = PyPDFDirectoryLoader("data")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:50])
    
    vectors = FAISS.from_documents(final_documents, embeddings)
    
    print("Vector database loaded successfully.")
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # 1. Create the retriever from the vector store
    retriever = vectors.as_retriever()
    # 2. Create the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in your app

# Load the retrieval chain ONCE when the app starts
try:
    retrieval_chain = load_retrieval_chain()
except Exception as e:
    print(f"Failed to load vector database on startup: {e}")
    retrieval_chain = None

# --- NEW ROUTE TO SERVE YOUR WEBPAGE ---
@app.route("/")
def index():
    """
    Serves the index.html file from the 'templates' folder.
    """
    return render_template('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    """
    The main chat endpoint.
    Receives a JSON with "query" and returns a JSON with "intent" and "response".
    """
    if retrieval_chain is None:
        return jsonify({"error": "Vector database is not initialized. Check server logs."}), 500

    try:
        data = request.json
        user_query = data.get("query")

        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        print(f"Received query: {user_query}")

        start = time.process_time()
        
        # Invoke the chain with the user's query
        response = retrieval_chain.invoke({'input': user_query})
        
        response_time = time.process_time() - start
        print(f"Response time: {response_time:.4f} seconds")

        # --- Parse the JSON response from the LLM ---
        try:
            # The LLM's answer is in the 'answer' field
            llm_output_str = response['answer']
            # The LLM output itself is a JSON string, so we parse it.
            parsed_response = json.loads(llm_output_str)
            
            # We can also add the RAG context for debugging
            parsed_response["context"] = [doc.page_content for doc in response['context']]
            
            print(f"LLM Response: {parsed_response}")
            
            return jsonify(parsed_response)

        except json.JSONDecodeError:
            print(f"Error: LLM did not return valid JSON. Response was: {llm_output_str}")
            return jsonify({"intent": "qa", "response": "I'm sorry, I had a small glitch. Could you rephrase that?"})
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return jsonify({"intent": "qa", "response": "I'm sorry, I'm having trouble processing that request."})

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

# --- /book ENDPOINT REMOVED ---

# --- Run the Flask Server ---
if __name__ == "__main__":
    # Ensure a 'data' directory exists
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' directory. Please add your PDF files here and restart the server.")
        
    # init_db() call removed
        
    print("Starting Flask server...")
    # Running on 0.0.0.0 makes it accessible on your network, ready for EC2
    app.run(debug=True, host="0.0.0.0", port=5000)

