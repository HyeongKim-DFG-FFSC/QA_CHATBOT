# rag.py
import os
from typing import List, Dict
from pinecone import Pinecone
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm  # Add this for progress tracking
from .embeddings import get_embeddings


class RAGSystem:
    def __init__(self):
        load_dotenv()

        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index_name = "rag-workshop"
        self.index = self.pc.Index(self.index_name)

        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

    def store_embeddings(self, documents: List[Dict], batch_size: int = 100):
        """Store document embeddings in Pinecone."""
        print(f"Processing {len(documents)} documents in batches of {batch_size}")

        for i in tqdm(range(0, len(documents), batch_size), desc="Storing batches"):
            batch = documents[i:i + batch_size]
            vectors = []

            try:
                for idx, doc in enumerate(batch):
                    embedding = get_embeddings(doc['text'])
                    vectors.append((
                        f"doc_{i + idx}",
                        embedding,
                        {'text': doc['text'], **doc['metadata']}
                    ))

                # Add error handling for upsert
                try:
                    self.index.upsert(vectors=vectors)
                except Exception as e:
                    print(f"Error upserting batch {i // batch_size}: {str(e)}")
                    continue

            except Exception as e:
                print(f"Error processing batch {i // batch_size}: {str(e)}")
                continue

        print("Embedding storage completed!")

    def query(self, question: str, top_k: int = 3) -> str:
        """Query the RAG system."""
        # Get question embedding
        question_embedding = get_embeddings(question)

        # Query Pinecone
        results = self.index.query(
            vector=question_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Prepare context from retrieved documents
        context = "\n".join([r.metadata['text'] for r in results.matches])

        # Prepare prompt
        prompt = f"""Use the following context to answer the question. If you cannot answer the question based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""

        # Generate response using Groq
        completion = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.1
        )

        return completion.choices[0].message.content