import streamlit as st
import os
from utils import PDFProcessor
from groq_api import GroqAPI
import tempfile
import time
import gc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state
if 'pdf_processor' not in st.session_state:
    st.session_state.pdf_processor = PDFProcessor()
if 'groq_api' not in st.session_state:
    # Get API key from environment variable, or prompt user to enter it
    api_key = os.getenv("GROQ_API_KEY", "")
    
    if not api_key:
        st.sidebar.warning("⚠️ No GROQ_API_KEY found in environment variables.")
        api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")
        if api_key:
            st.session_state.groq_api = GroqAPI(api_key=api_key)
        else:
            st.sidebar.error("Please enter your Groq API key to use the chat functionality.")
    else:
        st.session_state.groq_api = GroqAPI(api_key=api_key)
        
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# Set page config
st.set_page_config(page_title="PDF AI Assistant", layout="wide")

# Title
st.title("📚 PDF AI Assistant")

# Cache the PDF processing function to avoid recomputing
@st.cache_resource(show_spinner=False)
def process_pdf_once(file_path, file_name):
    """Process a PDF file once and cache the results."""
    try:
        # Show progress bar during text extraction
        status_text = st.empty()
        status_text.text("Extracting text from PDF...")
        chunks = st.session_state.pdf_processor.process_pdf(file_path)
        
        if not chunks:
            st.error("Failed to extract text from the PDF. Please try a different document.")
            return False
        
        # Show progress bar during embedding generation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress):
            progress_bar.progress(progress)
            status_text.text(f"Generating embeddings: {int(progress * 100)}% complete")
        
        start_time = time.time()
        embeddings = st.session_state.pdf_processor.generate_embeddings(chunks, progress_callback=update_progress)
        end_time = time.time()
        
        if len(embeddings) == 0:
            st.error("Failed to generate embeddings. Please try a different document.")
            return False
        
        # Update the embedding index
        st.session_state.pdf_processor.build_index(embeddings)
        
        # Add to the set of processed files
        st.session_state.processed_files.add(file_name)
        
        # Clear the progress indicators
        progress_bar.empty()
        status_text.text(f"PDF processed successfully! Processing took {end_time - start_time:.2f} seconds")
        
        # Force garbage collection
        gc.collect()
        
        return True
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
        
        # Check file size
        if file_size > 50:  # 50MB limit
            st.error("File is too large. Please upload a PDF smaller than 50MB.")
        else:
            # Check if this file has been processed before
            if file_name not in st.session_state.processed_files:
                try:
                    # Save the uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Process the PDF with progress tracking
                    success = process_pdf_once(tmp_file_path, file_name)
                    
                    # Clean up the temporary file
                    os.unlink(tmp_file_path)
                    
                    if not success:
                        st.error("Failed to process the document. Please try a different PDF.")
                except Exception as e:
                    st.error(f"Error uploading file: {str(e)}")
            else:
                st.success(f"Using cached results for {file_name}")
    
    # Display the number of documents processed
    if st.session_state.processed_files:
        st.write(f"📊 Processed documents: {len(st.session_state.processed_files)}")
        with st.expander("Documents in memory"):
            for doc in st.session_state.processed_files:
                st.write(f"- {doc}")

# Main chat interface
st.header("Chat with your PDF")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Check if Groq API is configured
if 'groq_api' not in st.session_state:
    st.warning("Please enter your Groq API key in the sidebar to enable the chat functionality.")
else:
    # Chat input
    if prompt := st.chat_input("Ask a question about your PDF"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get relevant context from the PDF
        with st.spinner("Searching for relevant information..."):
            try:
                relevant_chunks = st.session_state.pdf_processor.search(prompt)
                if relevant_chunks:
                    context = "\n".join([chunk for chunk, _ in relevant_chunks])
                else:
                    context = ""
                    st.warning("No relevant context found in the document.")
            except Exception as e:
                st.error(f"Error searching document: {str(e)}")
                context = ""
        
        # Get response from Groq API
        with st.chat_message("assistant"):
            try:
                with st.spinner("Generating response..."):
                    response = st.session_state.groq_api.get_response(prompt, context)
                    st.write(response)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Clear buttons
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        if 'groq_api' in st.session_state:
            st.session_state.groq_api.clear_history()
        st.rerun()
with col2:
    if st.button("Clear Document Cache"):
        st.session_state.processed_files = set()
        st.session_state.pdf_processor = PDFProcessor()
        # Force garbage collection
        gc.collect()
        st.rerun()