# document_loader.py
from typing import List, Dict
import os
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_split_documents(directory_path: str) -> List[Dict]:
    """Load documents from a directory and split them into chunks."""
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                try:
                    loader = UnstructuredPDFLoader(file_path)
                    docs = loader.load()
                    # Split documents
                    splits = text_splitter.split_documents(docs)
                    documents.extend([{
                        'text': doc.page_content,
                        'metadata': {
                            'source': file_path,
                            **doc.metadata
                        }
                    } for doc in splits])
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return documents


