# PDF AI Assistant

A Streamlit-based AI assistant that processes PDF documents, generates embeddings, and provides intelligent responses using Llama 3 8B via the Groq API.

![PDF AI Assistant](https://raw.githubusercontent.com/Omni-potence/chat-with-multiple-pdfs/main/assets/screenshot.png)

## Features

- **PDF Upload and Processing**: Upload PDFs and extract text with multithreaded processing
- **Embedding Generation**: Create embeddings from PDF content using SentenceTransformers
- **Vector Search**: Store embeddings in FAISS for efficient retrieval
- **Conversational AI**: Generate responses using Llama 3 8B via Groq API
- **Context Retention**: Maintain conversation history for contextual awareness
- **Memory Management**: Efficient processing of large documents with batch processing and garbage collection
- **Progress Tracking**: Real-time progress indicators for embedding generation

## Technologies Used

### Core Components

- **Streamlit**: Web interface for document upload and chat
- **PyMuPDF**: Efficient PDF text extraction
- **SentenceTransformers**: Generate embeddings from text chunks
- **FAISS**: Fast vector similarity search for retrieving relevant content
- **Groq API**: Access to Llama 3 8B language model for response generation

### Optimization Features

- Multithreaded PDF processing
- Batched embedding generation
- Memory-efficient text chunking
- Streamlit caching to prevent recomputation
- Robust error handling

## Project Structure

- `app.py`: Streamlit user interface and application flow
- `utils.py`: PDF processing, embedding generation, and vector search functionality
- `groq_api.py`: Integration with Llama 3 8B via Groq API
- `requirements.txt`: Dependencies for the project

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Omni-potence/chat-with-multiple-pdfs.git
cd chat-with-multiple-pdfs
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate.bat
# On Unix or MacOS
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Upload a PDF document using the file uploader in the sidebar
3. Wait for the document to be processed (a progress bar will show embedding generation progress)
4. Ask questions about the document in the chat interface
5. The AI will search for relevant content in the document and generate responses

## Performance Considerations

- Documents are processed once and cached to avoid redundant processing
- Large documents (>50MB) are not allowed to prevent memory issues
- Text extraction and embedding generation are performed in batches to manage memory
- Garbage collection is performed periodically to free up memory

## License

MIT

## Acknowledgements

This project utilizes various open-source libraries and the Groq API to enable intelligent document interaction.