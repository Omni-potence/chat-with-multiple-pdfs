import pymupdf as fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple, Generator
from concurrent.futures import ThreadPoolExecutor
import time
import gc

class PDFProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Dimension of the embeddings
        self.index = faiss.IndexFlatL2(self.dimension)
        self.text_chunks = []
        self.chunk_size = 300  # Reduced from 500 to 300
        self.chunk_overlap = 20  # Reduced from 50 to 20
        self.is_index_initialized = False
        self.max_chunks = 1000  # Maximum number of chunks to process at once
        self.batch_size = 10  # Number of pages to process at once

    def extract_text_from_page(self, page):
        """Extract text from a single PDF page."""
        try:
            return page.get_text()
        except Exception as e:
            print(f"Error extracting text from page: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file using batched multithreading."""
        doc = fitz.open(pdf_path)
        text = []
        
        # Process pages in batches to manage memory
        for i in range(0, len(doc), self.batch_size):
            batch_pages = [doc[j] for j in range(i, min(i + self.batch_size, len(doc)))]
            
            # Use ThreadPoolExecutor to process batch pages in parallel
            with ThreadPoolExecutor(max_workers=min(4, len(batch_pages))) as executor:
                batch_results = list(executor.map(self.extract_text_from_page, batch_pages))
                text.extend(batch_results)
            
            # Force garbage collection after each batch
            gc.collect()
        
        doc.close()
        return "".join(text)

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks with memory management."""
        chunks = []
        start = 0
        text_length = len(text)
        
        try:
            while start < text_length and len(chunks) < self.max_chunks:
                end = start + self.chunk_size
                if end > text_length:
                    end = text_length
                chunk = text[start:end]
                
                # Skip empty or whitespace-only chunks
                if chunk.strip():
                    chunks.append(chunk)
                
                start = end - self.chunk_overlap
                
                # Force garbage collection periodically
                if len(chunks) % 100 == 0:
                    gc.collect()
            
            return chunks[:self.max_chunks]  # Limit the number of chunks
        except MemoryError:
            print("Memory error during chunking - reducing chunk count")
            # If we hit a memory error, return what we have so far
            return chunks[:len(chunks)//2]

    def process_pdf(self, pdf_path: str) -> List[str]:
        """Process PDF and return text chunks with error handling."""
        try:
            text = self.extract_text_from_pdf(pdf_path)
            self.text_chunks = self.chunk_text(text)
            return self.text_chunks
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return []

    def generate_embeddings(self, chunks: List[str], progress_callback=None) -> np.ndarray:
        """Generate embeddings for text chunks with batching and progress tracking."""
        total_chunks = len(chunks)
        embeddings = []
        batch_size = 5  # Process 5 chunks at a time
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            try:
                batch_embeddings = self.model.encode(batch)
                embeddings.extend(batch_embeddings)
                
                # Report progress if callback is provided
                if progress_callback:
                    progress_callback((i + len(batch)) / total_chunks)
                
                # Force garbage collection after each batch
                gc.collect()
                
            except Exception as e:
                print(f"Error generating embeddings for batch: {e}")
                continue
        
        return np.array(embeddings)

    def build_index(self, embeddings: np.ndarray):
        """Build or update FAISS index for embeddings with error handling."""
        try:
            if not self.is_index_initialized:
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.is_index_initialized = True
            
            # Add embeddings to the index
            self.index.add(embeddings.astype('float32'))
        except Exception as e:
            print(f"Error building index: {e}")

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Search for most relevant chunks given a query with error handling."""
        try:
            query_embedding = self.model.encode([query])
            distances, indices = self.index.search(query_embedding.astype('float32'), k)
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.text_chunks):
                    results.append((self.text_chunks[idx], float(distance)))
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []