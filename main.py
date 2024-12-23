# main.py
from src import RAGSystem, load_and_split_documents
from dotenv import load_dotenv
import os
import sys
import time
from src import *

# main.py
def main():
    try:
        load_dotenv()

        print("Initializing RAG system...")
        rag = RAGSystem()

        # Check if documents already exist
        if not rag.check_documents_exist():
            print("\nLoading documents...")
            docs_path = os.path.join("data", "CompanyDocuments")

            if not os.path.exists(docs_path):
                print(f"Error: Document path '{docs_path}' does not exist!")
                return

            documents = load_and_split_documents(docs_path)
            if not documents:
                print("No documents were loaded!")
                return

            print(f"Loaded {len(documents)} document chunks")

            print("\nStarting embedding storage process...")
            start_time = time.time()

            rag.store_embeddings(documents)

            end_time = time.time()
            print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
        else:
            print("\nDocuments already exist in the database. Skipping document loading and embedding.")

        # Test queries
        questions = [
            "What are some items that Pirkko Koskitalo is likely to buy next? What incentives can I put in place to ensure he orders more?",
            "What high-margin products are currently in our inventory, and what personalized promotions could we create for customers who frequently purchase similar items?",
            "Which customers have decreased their order frequency in the past 3 months, and what incentives based on their previous top purchases could bring them back?",
            "Looking at our top 10 selling products, which complementary items in our inventory should we recommend to customers who buy these, and what bundle discounts would be most effective?",
            "What items do customers typically purchase together from our current inventory, and how can we optimize our loyalty program to increase these combination purchases?",
            "Based on seasonal buying patterns and current inventory levels, which customer segments should we target with early access promotions, and what minimum order values would maximize profitability?"
        ]

        for question in questions:
            print(f"\nQuestion: {question}")
            try:
                answer = rag.query(question)
                print(f"Answer: {answer}")
            except Exception as e:
                print(f"Error processing question: {str(e)}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()