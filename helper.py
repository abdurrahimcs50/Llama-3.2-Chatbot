from PIL import Image, UnidentifiedImageError
import pytesseract
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def extract_text_from_image(image_path):
    """Extract text from an image using Tesseract OCR."""
    image = Image.open(image_path)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    return pytesseract.image_to_string(image)

def load_pdf_documents(pdf_path):
    """Load documents from a PDF using PyPDFLoader."""
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def split_text_img_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split text documents using RecursiveCharacterTextSplitter."""
    print("Splitting text documents...")
    print(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents([documents])
    return docs

def split_text_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split text documents using RecursiveCharacterTextSplitter."""
    print("Splitting text documents...")
    print(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def create_vector_store(splits, model_name="sentence-transformers/all-mpnet-base-v2", device='cpu',):
    """Create a vector store with FAISS and HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device}, encode_kwargs={'normalize_embeddings': False})
    index = faiss.IndexFlatL2(len(embeddings.embed_query("Rag App")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    uuids = [str(uuid4()) for _ in range(len(splits))]
    vector_store.add_documents(documents=splits, ids=uuids)
    vector_store.save_local("My_Test_App_Data")
    #check local db created or not
    if os.path.exists("My_Test_App_Data"):
        print("Local db created")
        return 'Local db created'
    else:
        print("Local db not created")
        return 'Local db not created'

def load_and_search_vector_store(VECTOR_STORE_DB_NAME,):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': False})
    vector_store = FAISS.load_local(VECTOR_STORE_DB_NAME, embeddings, allow_dangerous_deserialization=True)
    retriver = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10})
    return retriver

def create_rag_chain(groq_api_key, user_query, VECTOR_STORE_DB_NAME):
    """Create a retrieval-augmented generation (RAG) chain."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': False})
    vector_store = FAISS.load_local(VECTOR_STORE_DB_NAME, embeddings, allow_dangerous_deserialization=True)
    retriver = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.2-90b-vision-preview")
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Provide the most relevant and direct answer "
        "based on the retrieved context Focus on accuracy and clarity, "
        "avoiding unnecessary elaborationand and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriver, question_answer_chain)
    response = rag_chain.invoke({"input": user_query})
    print(response["answer"])
    return response

# def main():
#     image_text = extract_text_from_image('F:\\DevWorkSpace\\WSP-2024\\Mohammad\\data\\img2.png')
#     # print("Extracted Text:\n", image_text)
    
#     docs = load_pdf_documents("./resources/resources/folder2/engineeringguidebook.pdf")
#     splits = split_text_documents(docs)
    
#     vector_store = create_vector_store(splits)
#     vector_store.save_local("My_Test_App_Data")
    
#     new_docs, new_retriver = load_and_search_vector_store("My_Test_App_Data", vector_store.embedding_function, "what is engineering?")
#     new_documents = new_retriver.invoke("what is engineering?")
    
#     rag_chain = create_rag_chain(new_retriver, "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")
#     response = rag_chain.invoke({"input": "What is engineering?"})
#     print(response["answer"])

# if __name__ == "__main__":
#     main()
