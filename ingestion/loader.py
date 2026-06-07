import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from pypdf import PdfReader

class DocumentLoader:
    """Loads text, markdown, and PDF files from directories or specific paths."""
    
    @staticmethod
    def load_file(file_path: str) -> List[Document]:
        """Loads a single file (PDF, MD, or TXT) and returns it as a list of Documents."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        _, ext = os.path.splitext(file_path.lower())
        documents = []
        
        try:
            if ext == ".pdf":
                reader = PdfReader(file_path)
                text = ""
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text:
                        # Create a Document per page to keep page tracking metadata
                        documents.append(
                            Document(
                                page_content=page_text,
                                metadata={
                                    "source": file_path,
                                    "file_name": os.path.basename(file_path),
                                    "page": page_num + 1,
                                }
                            )
                        )
            elif ext in [".md", ".txt"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": file_path,
                            "file_name": os.path.basename(file_path),
                        }
                    )
                )
            else:
                print(f"Skipping unsupported file type: {file_path}")
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            
        return documents

    @classmethod
    def load_directory(cls, directory_path: str) -> List[Document]:
        """Recursively loads all PDF, MD, and TXT files from a directory."""
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        all_documents = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file.lower())
                if ext in [".pdf", ".md", ".txt"]:
                    all_documents.extend(cls.load_file(file_path))
                    
        return all_documents
