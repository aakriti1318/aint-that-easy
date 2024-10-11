import os
import pymupdf4llm
from llama_index.core import Document, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageNode
from PIL import Image
import matplotlib.pyplot as plt
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss

# Set up OpenAI API key (replace with your actual key)
os.environ["OPENAI_API_KEY"] = "..."

# Initialize CLIP model for image embeddings
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_pdf_content(pdf_path):
    # Extract content using PyMuPDF4LLM
    docs = pymupdf4llm.to_markdown(
        doc=pdf_path,
        page_chunks=True,
        write_images=True,
        image_path="images",
        image_format="jpg",
    )
    return docs

def create_llama_documents(docs):
    llama_documents = []
    for document in docs:
        metadata = {
            "file_path": document["metadata"].get("file_path"),
            "page": str(document["metadata"].get("page")),
            "images": str(document.get("images")),
            "toc_items": str(document.get("toc_items")),
        }
        llama_document = Document(
            text=document["text"],
            metadata=metadata,
            text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
        )
        llama_documents.append(llama_document)
    return llama_documents

def create_vector_stores():
    # Create FAISS indexes
    text_index = faiss.IndexFlatL2(1536)  # OpenAI embeddings are 1536-dimensional
    image_index = faiss.IndexFlatL2(512)  # CLIP embeddings are 512-dimensional

    # Create FAISS vector stores
    text_store = FaissVectorStore(faiss_index=text_index)
    image_store = FaissVectorStore(faiss_index=image_index)
    
    return text_store, image_store

def create_multimodal_index(llama_documents, text_store, image_store, image_path):
    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )
    image_documents = SimpleDirectoryReader(image_path).load_data()
    index = MultiModalVectorStoreIndex.from_documents(
        llama_documents + image_documents,
        storage_context=storage_context,
    )
    return index

def retrieve_content(index, query, top_k=1):
    retriever = index.as_retriever(similarity_top_k=top_k, image_similarity_top_k=top_k)
    return retriever.retrieve(query)

def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)
            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])
            images_shown += 1
            if images_shown >= 9:
                break
    plt.show()

def display_source_node(node, source_length=200):
    print(f"Score: {node.score}")
    print(f"Content: {node.node.get_content()[:source_length]}...")
    print(f"Metadata: {node.node.metadata}")
    print("--------------------")

def main():
    pdf_path = "document.pdf"  # Update this path to your PDF file location
    image_path = "images"  # Update this path to your image directory

    # Extract PDF content
    docs = extract_pdf_content(pdf_path)
    print("PDF content extracted successfully.")

    # Create LlamaIndex documents
    llama_documents = create_llama_documents(docs)
    print("LlamaIndex documents created.")

    # Create vector stores
    text_store, image_store = create_vector_stores()
    print("Vector stores created.")

    # Create multimodal index
    index = create_multimodal_index(llama_documents, text_store, image_store, image_path)
    print("Multimodal index created.")

    # Example queries
    queries = [
        "Could you provide an image of the Multi-Head Attention?",
        "What are the key components of the Transformer architecture?",
        "Show me a diagram of the encoder-decoder structure.",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        retrieval_results = retrieve_content(index, query, top_k=2)

        retrieved_images = []
        for res_node in retrieval_results:
            if isinstance(res_node.node, ImageNode):
                print("Retrieved ImageNode")
                print("-----------------")
                retrieved_images.append(res_node.node.metadata["file_path"])
                display_source_node(res_node)
            else:
                print("Retrieved TextNode")
                print("-----------------")
                display_source_node(res_node)

        if retrieved_images:
            plot_images(retrieved_images)

if __name__ == "__main__":
    main()
