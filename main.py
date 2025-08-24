import os
import json
import fitz
import re
import datetime
from dotenv import load_dotenv
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
import pymongo
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.docstore.document import Document
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
class KnowledgeRequest(BaseModel):
    topic: str
    index_name: str

class ChatRequest(BaseModel):
    chat_id: str
    index_name: str
    user_message: str

app = FastAPI(
    title="AI Tutor - Full API",
    description="Handles PDF processing, graph generation, summaries, and chat."
)
origins = [
    "http://127.0.0.1:5500", # The address of your local HTML file
    "http://localhost:5500", # Another common address for local files
    "null", # This is important for requests from local files (origin: null)
    # Add your future frontend's deployed URL here later
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

try:
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
    )
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_version=os.getenv("EMBEDDING_MODEL_VERSION"),
    )
    print("Successfully initialized Azure OpenAI clients.")
except Exception as e:
    print(f" Error initializing Azure OpenAI clients: {e}")
    llm = None
    embeddings = None

try:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print(" Successfully connected to Pinecone!")
except Exception as e:
    print(f"Error connecting to Pinecone: {e}")
    pc = None

try:
    MONGO_URI = os.getenv("MONGO_URI")
    mongo_client = pymongo.MongoClient(MONGO_URI)
    db = mongo_client.get_database("ai_tutor_db")
    chat_history_collection = db.get_collection("chat_histories")
    mongo_client.server_info()
    print(" Successfully connected to MongoDB!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    mongo_client = None

def sanitize_filename(filename: str) -> str:
    return "".join(c if c.isalnum() else '-' for c in filename).lower()

def extract_json_from_string(text: str) -> str:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        raise ValueError("No valid JSON object found in the LLM output.")

def generate_graph_from_text(document_text: str) -> str:
    global llm
    prompt_template = """
    Based on the following text, identify the main topics and their relationships.
    Generate a JSON object with two keys: "nodes" and "edges".
    - "nodes" should be a list of objects, each with an "id" and "label".
    - "edges" should be a list of objects, each with a "source" and a "target" id.
    IMPORTANT: Your response MUST be ONLY the JSON object. Do not include any extra text,
    explanations, or markdown formatting like ```json. The response must start with a '{{'
    and end with a '}}'.
    Here is the text:
    ---
    {text_chunk}
    ---
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    llm_raw_output = chain.invoke({"text_chunk": document_text})
    print(f"LLM RAW OUTPUT: {llm_raw_output}")
    return extract_json_from_string(llm_raw_output)

def add_message_to_history(chat_id: str, user_message: str, ai_response: str):
    if not mongo_client: return
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    chat_history_collection.insert_many([
        {"chat_id": chat_id, "role": "user", "content": user_message, "created_at": timestamp},
        {"chat_id": chat_id, "role": "assistant", "content": ai_response, "created_at": timestamp}
    ])

def get_chat_history(chat_id: str) -> list:
    if not mongo_client: return []
    history = []
    messages = chat_history_collection.find({"chat_id": chat_id}).sort("created_at", 1)
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    return history

@app.post("/api/process-pdf")
async def process_pdf_and_create_graph(file: UploadFile = File(...)):
    if not all([pc, llm, embeddings]):
        raise HTTPException(status_code=500, detail="A required service is not initialized.")
    
    try:
        pdf_bytes = await file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        docs = [
            Document(page_content=page.get_text(), metadata={"page": i + 1})
            for i, page in enumerate(pdf_document) if page.get_text()
        ]
        pdf_document.close()

        if not docs:
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(docs)

        index_name = "nerv"
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=3072,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        
        PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
        
        full_text = "".join([doc.page_content for doc in docs])
        GRAPH_GENERATION_MAX_LENGTH = 50000
        truncated_text = full_text[:GRAPH_GENERATION_MAX_LENGTH]
        json_string_from_ai = generate_graph_from_text(truncated_text)
        graph_data = json.loads(json_string_from_ai)
        
        return {
            "message": "Document processed successfully.",
            "index_name": index_name,
            "graph_data": graph_data
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/get-summary")
async def get_summary(request: KnowledgeRequest):
    if not all([pc, llm, embeddings]):
        raise HTTPException(status_code=500, detail="A required service is not initialized.")

    try:
        vector_store = PineconeVectorStore.from_existing_index(request.index_name, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(request.topic)
        context_text = "\n\n".join([
            f"Source (Page {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}" 
            for doc in relevant_docs
        ])

        summary_prompt_template = "Based *only* on the following text, write a concise summary of the topic: '{topic}'.\n\nText:\n---\n{context}\n---"
        prompt = ChatPromptTemplate.from_template(summary_prompt_template)
        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({"topic": request.topic, "context": context_text})
        
        return {"summary": summary}
    except Exception as e:
        print(f"An error occurred during summarization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_with_topic(request: ChatRequest):
    if not all([pc, llm, embeddings, mongo_client]):
        raise HTTPException(status_code=500, detail="A required service is not initialized.")
    
    try:
        history = get_chat_history(request.chat_id)
        
        vector_store = PineconeVectorStore.from_existing_index(request.index_name, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(request.user_message)
        context_text = "\n\n".join([
            f"Source (Page {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}" 
            for doc in relevant_docs
        ])
        
        system_prompt = f"""
        You are an expert AI tutor for the topic of "{request.chat_id}".
        Your goal is to provide the best possible answer. Base your answer on the user's conversation history and the relevant context from the document provided below. Prioritize the document's information.
        CONTEXT FROM DOCUMENT:
        ---
        {context_text}
        ---
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            *history,
            HumanMessage(content=request.user_message)
        ]
        
        ai_response = llm.invoke(messages).content
        
        add_message_to_history(request.chat_id, request.user_message, ai_response)
        
        return {"ai_response": ai_response}
    except Exception as e:
        print(f"An error occurred during chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))