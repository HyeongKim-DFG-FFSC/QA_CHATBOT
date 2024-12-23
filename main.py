# main.py
from src import RAGSystem, load_and_split_documents
from dotenv import load_dotenv
import os


def main():
    # Load environment variables
    load_dotenv()

    # Initialize RAG system first
    print("Initializing RAG system...")
    rag = RAGSystem()

    # Check if data already exists in Pinecone
    stats = rag.index.describe_index_stats()
    total_records = stats.total_vector_count

    if total_records > 0:
        print(f"Found {total_records} existing records in Pinecone. Skipping data loading...")
    else:
        # Only load and store documents if Pinecone is empty
        print("Loading documents...")
        docs_path = os.path.join("data", "CompanyDocuments")
        documents = load_and_split_documents(docs_path)
        print(f"Loaded {len(documents)} document chunks")

        print("Storing embeddings...")
        rag.store_embeddings(documents)

    # Test query
    question = "What items are in the inventory?"
    print(f"\nQuestion: {question}")
    answer = rag.query(question)
    print(f"Answer: {answer}")


    question = "What are some items that Pirkko Koskitalo is likely to buy next? What incentives can I put in place to ensure he orders more?"
    print(f"\nQuestion: {question}")
    answer = rag.query(question)
    print(f"Answer: {answer}")

    # question = "What incentives can I put in place to ensure he orders more?"
    # print(f"\nQuestion: {question}")
    # answer = rag.query(question)
    # print(f"Answer: {answer}")

if __name__ == "__main__":
    main()