from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
from openai.error import OpenAIError
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI API (set your API key)
openai.api_key = "your-api-key"

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
dimension = 384  # Embedding dimension for 'all-MiniLM-L6-v2'
index = faiss.IndexFlatL2(dimension)
chunks = []  # Store chunks for reverse lookup

# Advanced Memory Simulation for MemGPT
class MemGPT:
    def __init__(self):
        self.episodic_memory = []  # Stores recent, short-term memory (raw chunks)
        self.semantic_memory = []  # Stores long-term summarized knowledge
        self.memory_limit = 10  # Limit for episodic memory before summarization

    def add_to_episodic(self, content):
        """Add content to episodic memory."""
        self.episodic_memory.append(content)
        if len(self.episodic_memory) > self.memory_limit:
            self.summarize_to_semantic()

    def summarize_to_semantic(self):
        """Summarize episodic memory and store it in semantic memory."""
        summary = " ".join(self.episodic_memory[-5:])  # Summarize last 5 memories
        self.semantic_memory.append(summary)
        self.episodic_memory = self.episodic_memory[:-5]  # Keep earlier episodic memories

    def recall_memory(self, query=None):
        """Recall memory with priority on recent episodic content."""
        if query:
            # Use a simple query similarity to prioritize memories
            relevant_episodic = [chunk for chunk in self.episodic_memory if query in chunk]
            relevant_semantic = [chunk for chunk in self.semantic_memory if query in chunk]
            return " ".join(relevant_episodic + relevant_semantic)
        return " ".join(self.episodic_memory + self.semantic_memory)

    def clear_memory(self):
        """Clear all memory."""
        self.episodic_memory = []
        self.semantic_memory = []

agent = MemGPT()

# PDF processing and embedding
@app.route('/upload', methods=['POST'])
def upload_pdf():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file provided"}), 400
    
    # Parse PDF
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_chunks = text_splitter.split_text(text)
    
    # Embed and index chunks
    global chunks
    chunks = split_chunks
    embeddings = embedding_model.encode(split_chunks)
    index.add(np.array(embeddings))
    
    return jsonify({"message": "PDF uploaded and indexed successfully", "chunks": len(split_chunks)})

@app.route('/query', methods=['POST'])
def query_document():
    data = request.json
    query = data.get('query', "")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Embed query
    query_embedding = embedding_model.encode([query])
    
    # Check if the FAISS index is empty
    if index.ntotal == 0:
        return jsonify({"error": "Index is empty. Please upload and process a document first."}), 400

    # Retrieve top-k matching chunks
    k = 5
    distances, indices = index.search(np.array(query_embedding), k)
    
    # Filter out invalid indices (-1 or indices beyond `chunks` length)
    valid_indices = [i for i in indices[0] if 0 <= i < len(chunks)]
    if not valid_indices:
        return jsonify({"error": "No relevant chunks found for the query."}), 400

    retrieved_chunks = [chunks[i] for i in valid_indices]

    # Add to MemGPT episodic memory
    for chunk in retrieved_chunks:
        agent.add_to_episodic(chunk)
    
    # Generate response using OpenAI GPT
    context = agent.recall_memory(query=query)  # Prioritize memory relevant to the query
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a legal expert."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response['choices'][0]['message']['content']
        return jsonify({"answer": answer})
    except OpenAIError as e:  # Correctly handle OpenAI API errors
        return jsonify({"error": f"Failed to generate a response: {str(e)}"}), 500

@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    agent.clear_memory()
    return jsonify({"message": "Memory cleared successfully"})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
