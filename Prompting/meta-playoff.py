import os
from typing import List
import fitz
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set your OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "..."

# Initialize OpenAI clients
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
chat_model = ChatOpenAI(model="gpt-4o-mini")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to create embeddings
def create_embeddings(texts: List[str]) -> List[List[float]]:
    return embeddings_model.embed_documents(texts)

# Function to perform RAG
def rag_query(query: str, index: faiss.IndexFlatL2, texts: List[str], k: int = 3) -> List[str]:
    query_embedding = embeddings_model.embed_query(query)
    _, indices = index.search(np.array([query_embedding]), k)
    return [texts[i] for i in indices[0]]

# Different prompting techniques
def normal_prompt(query: str, context: List[str]) -> str:
    prompt = f"Query: {query}\n\nContext:\n" + "\n".join(context) + "\n\nAnswer:"
    response = chat_model.invoke(prompt)
    return response.content

def meta_prompt(query: str, context: List[str]) -> str:
    meta_prompt = f"""
    You are an AI assistant tasked with answering questions based on given context.
    Your goal is to provide accurate and relevant answers.
    
    Query: {query}
    
    Context:
    {' '.join(context)}
    
    Please follow these steps:
    1. Analyze the query and context carefully.
    2. Identify the key information relevant to the query.
    3. Formulate a clear and concise answer.
    4. Ensure your answer is directly related to the query and supported by the context.
    
    Your answer:
    """
    response = chat_model.invoke(meta_prompt)
    return response.content

def playoff_prompt(query: str, context: List[str]) -> str:
    # Generate multiple answers
    answers = []
    for _ in range(3):
        prompt = f"Query: {query}\n\nContext:\n" + "\n".join(context) + "\n\nAnswer:"
        response = chat_model.invoke(prompt)
        answers.append(response.content)
    
    playoff_prompt = f"""
    You are tasked with evaluating multiple options in response to a given query using a "playoff" method to determine the best choice. Follow these steps carefully:
    Step 1: Generate Options
    Create a list of 6 distinct options that address the query: "{query}". Each option should be relevant and creative while addressing the core aspects of the query.
    Step 2: The Playoff Rounds
    
    Quarterfinals:
    Pair the 6 options into 3 matchups. Compare each pair and choose a winner based on:

    Relevance to the query
    Creativity and uniqueness
    Practicality or feasibility
    Other criteria specific to the query

    Provide a brief explanation for each selection.
    
    Semifinals:
    Take the 3 winners from the quarterfinals and add the highest-ranked loser as a "wild card" to create 2 matchups. Compare each pair, select a winner, and explain your reasoning in more detail.
    
    Final Round:
    Compare the final 2 options. Choose the overall best option based on how well it addresses the query, its strengths, and any other relevant factors. Provide a detailed explanation for why this option stands out as the best.
    Step 3: The Final Output
    At the end of this playoff process, present:

    The complete list of the original 6 options, ranked from best to worst.
    The final chosen option (the top-ranked option).
    A detailed explanation for why the final option was selected as the best out of all 6.
    """
    response = chat_model.invoke(playoff_prompt)
    return response.content

# Main execution
if __name__ == "__main__":
    # Load and process PDF
    pdf_path = "document.pdf"
    full_text = extract_text_from_pdf(pdf_path)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(full_text)
    
    # Create embeddings and index
    embeddings = create_embeddings(texts)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    
    # Example query
    query = "What is the scope of AI in the field of education"
    
    # Retrieve relevant chunks
    relevant_chunks = rag_query(query, index, texts)
    
    # Apply different prompting techniques
    normal_result = normal_prompt(query, relevant_chunks)
    meta_result = meta_prompt(query, relevant_chunks)
    playoff_result = playoff_prompt(query, relevant_chunks)
    
    print("\n\nNormal Prompt Result:")
    print(normal_result)
    print("\n\n\nMeta-Prompt Result:")
    print(meta_result)
    print("\n\n\nPlayoff Prompt Result:")
    print(playoff_result)
