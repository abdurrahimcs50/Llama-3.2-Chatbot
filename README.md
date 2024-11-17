
```markdown
# RAG Application with Chat Interface

This project implements a **Retrieval-Augmented Generation (RAG)** application using LangChain and other components, allowing users to upload documents (PDFs and Images), process them, store the data in a vector store, and query the stored information using a chat interface. It integrates with Groq API and HuggingFace embeddings for document retrieval and generation.

## Features

- **File Upload**: Supports uploading PDFs and image files for processing.
- **Text Extraction from Images**: Uses Tesseract OCR to extract text from uploaded images.
- **PDF Processing**: Extracts text from PDFs and stores it for querying.
- **Text Splitting**: Uses LangChain's `RecursiveCharacterTextSplitter` to split documents into manageable chunks.
- **Vector Store**: Creates a vector store using FAISS for efficient document retrieval.
- **Retrieval-Augmented Generation (RAG)**: Integrates a RAG chain for answering queries based on the uploaded documents.

## Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- Tesseract OCR (for image text extraction)
- Required Python libraries:
  - `pytesseract==0.3.13`
  - `pillow==11.0.0`
  - `pypdf==5.1.0`
  - `langchain-community==0.3.7`
  - `faiss-cpu==1.9.0`
  - `langchain-huggingface==0.1.2`
  - `sentence-transformers==3.3.0`
  - `langchain-groq==0.2.1`
  - `Flask==3.1.0`
  - `dash==2.18.2`
  - `dash-bootstrap-components==1.6.0`
## Set up Groq API Key:
- Create a .env file in the root of the project.
- Add your Groq API key to the .env file:
```bash
# .env file contents
GROQ_API_KEY=your-groq-api-key-here
```
## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abdurrahimcs50/personal-assistant-chatbot.git
   cd personal-assistant-chatbot
   ```

2. **Install dependencies**:
   You can install the required Python packages using `pip`:
   ```bash
   python -m venv venv
   # on windows
   venv\Scripts\activate
   # on linux/mac
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR**:
   - Download and install Tesseract OCR from [here](https://github.com/tesseract-ocr/tesseract).
   - Update the `tesseract_cmd` variable in `helper.py` to point to your installed Tesseract location.

4. **Run the Flask and Dash app**:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000/` to interact with the app.

## How to Use

1. **Upload Documents**: 
   - Use the file upload interface to upload PDF or image files.
   - The app will extract text from images using Tesseract OCR and load text from PDFs.
   
2. **Process the Data**: 
   - Click on "Process Data" to process the uploaded documents and store them in a local vector database.
   - The documents will be split into smaller chunks and stored in a FAISS vector store for efficient search.

3. **Ask Questions**:
   - Enter your query in the chat interface and submit it.
   - The system will retrieve relevant documents and generate an answer based on the retrieved context using the RAG chain.

## Code Walkthrough

### Functions

- **`extract_text_from_image(image_path)`**:
  - Uses Tesseract OCR to extract text from images.
  
- **`load_pdf_documents(pdf_path)`**:
  - Loads and parses PDFs to extract text.

- **`split_text_documents(documents, chunk_size=1000, chunk_overlap=200)`**:
  - Splits large documents into smaller chunks to improve retrieval performance.

- **`create_vector_store(splits, model_name="sentence-transformers/all-mpnet-base-v2", device='cpu')`**:
  - Creates a vector store using HuggingFace embeddings and FAISS.

- **`load_and_search_vector_store(VECTOR_STORE_DB_NAME)`**:
  - Loads the saved vector store and sets up a retriever for querying.

- **`create_rag_chain(groq_api_key, user_query, VECTOR_STORE_DB_NAME)`**:
  - Creates a RAG chain that retrieves documents based on a user query and generates a response using Groq's LLM.

### Dash & Flask Interface

The Dash app provides a simple user interface for uploading files and interacting with the RAG-powered chatbot. Flask is used to handle file uploads and serve the app.

## Contributing

Feel free to fork this repository, submit issues, and create pull requests to contribute to the project. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **LangChain**: A framework for building LLM-powered applications, used for text splitting, vector stores, and RAG.
- **Tesseract OCR**: An open-source OCR engine used for extracting text from images.
- **FAISS**: A library for efficient similarity search, used to create the vector store.
- **Groq API**: Used for language model-based query answering.
```