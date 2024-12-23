# rag.py
from pymongo import MongoClient
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
from sentence_transformers import SentenceTransformer
import certifi
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi



# rag.py
class RAGSystem:
    def __init__(self):
        load_dotenv()


        MONGODB_URI = (
            "mongodb+srv://hyeongsukkim94:" + os.getenv('MONGODB_API_KEY') + "@dfg-ffsc-temp.ksg4q.mongodb.net/"
            "?retryWrites=true&w=majority&appName=DFG-FFSC-TEMP"
        )

        self.client = MongoClient(
            MONGODB_URI,
            server_api=ServerApi('1'),
            tlsCAFile=certifi.where(),
        )

        try:
            self.client.admin.command('ping')
            print("\nPinged your deployment. Successfully connected to MongoDB!")
        except Exception as e:
            print("\nFailed to connect to MongoDB:", e)
            raise

        self.db = self.client['rag_workshop']
        self.collection = self.db['documents']

        # Create index for vector search if it doesn't exist
        self._ensure_index()

        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self._model_lock = threading.Lock()

    def _ensure_index(self):
        """Ensure vector search index exists."""
        try:
            index_name = "vector_search_index"
            existing_indexes = list(self.collection.list_indexes())
            index_exists = any(index.get("name") == index_name for index in existing_indexes)

            if not index_exists:
                self.collection.create_index([
                    ("embedding_vector", "2dsphere")
                ], name=index_name)
                print("Vector search index created successfully")
        except Exception as e:
            print(f"Error creating index: {str(e)}")

    def check_documents_exist(self) -> bool:
        """Check if documents already exist in the collection."""
        return self.collection.count_documents({}) > 0

    def store_embeddings(self, documents: List[Dict], batch_size: int = 100, max_workers: int = 4, force: bool = False):
        """Store document embeddings using parallel processing."""
        # Check if documents already exist
        if not force and self.check_documents_exist():
            print("\nDocuments already exist in the database. Skipping storage process.")
            print(f"Total documents in database: {self.collection.count_documents({})}")
            return

        from tqdm import tqdm

        total_documents = len(documents)
        print(f"\nProcessing {total_documents} documents in batches of {batch_size}")

        pbar = tqdm(total=total_documents, desc="Processing documents")
        processed_queue = queue.Queue()

        def process_batch(batch):
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self._process_document, doc) for doc in batch]
                for future in futures:
                    result = future.result()
                    if result:
                        results.append(result)
                        pbar.update(1)

            if results:
                try:
                    self.collection.insert_many(results)
                except Exception as e:
                    print(f"\nError inserting batch: {str(e)}")

            processed_queue.put(len(results))

        try:
            # Clear existing documents if force is True
            if force:
                print("\nForce flag is True. Clearing existing documents...")
                self.collection.delete_many({})

            for i in range(0, total_documents, batch_size):
                batch = documents[i:min(i + batch_size, total_documents)]
                process_batch(batch)

        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
        finally:
            pbar.close()
            total_processed = 0
            while not processed_queue.empty():
                total_processed += processed_queue.get()
            print(f"\nProcessing complete. Total documents processed: {total_processed}/{total_documents}")



    def query(self, question: str, top_k: int = 3) -> str:
        try:
            with self._model_lock:
                question_embedding = self.model.encode(question).tolist()

            # Using $nearSphere for vector similarity search
            similar_docs = self.collection.find(
                {
                    "embedding_vector": {
                        "$nearSphere": {
                            "$geometry": {
                                "type": "Point",
                                "coordinates": question_embedding
                            }
                        }
                    }
                }
            ).limit(top_k)

            # Convert cursor to list and extract texts
            results = list(similar_docs)

            if not results:
                return "I couldn't find any relevant information in the provided context."

            context = "\n".join([doc['text'] for doc in results])

            prompt = f"""Based ONLY on the following context, answer the question. If the context doesn't contain relevant information, say so explicitly.

Context:
{context}

Question: {question}

Answer:"""

            completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant. Only answer based on the provided context. If you can't find the information in the context, say so explicitly."},
                    {"role": "user", "content": prompt}
                ],
                model="mixtral-8x7b-32768",
                temperature=0.1
            )

            return completion.choices[0].message.content

        except Exception as e:
            print(f"Error in query: {str(e)}")
            raise

    def _process_document(self, doc):
        """Process a single document with thread-safe model access."""
        try:
            with self._model_lock:
                embedding = self.model.encode(doc['text']).tolist()
            return {
                'text': doc['text'],
                'metadata': doc['metadata'],
                'embedding_vector': embedding
            }
        except Exception as e:
            print(f"\nError processing document: {str(e)}")
            return None