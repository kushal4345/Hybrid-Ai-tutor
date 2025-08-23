# --- 1. IMPORTS & SETUP ---

import os
import json
import fitz  # PyMuPDF
import re
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, HTTPException
from pinecone import Pinecone, ServerlessSpec # This is from the new 'pinecone' package

# LangChain Imports for Azure
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

# Load all environment variables from .env file
load_dotenv()

# --- 2. INITIALIZE SERVICES ---

app = FastAPI(
    title="AI Tutor - Document Processing API",
    description="Processes a PDF to generate a knowledge graph and a searchable knowledge base using Azure OpenAI."
)

# Initialize Azure OpenAI Models
try:
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
    )
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_version=os.getenv("EMBEDDING_MODEL_VERSION"),
    )
    print("✅ Successfully initialized Azure OpenAI clients.")
except Exception as e:
    print(f"❌ Error initializing Azure OpenAI clients: {e}")
    llm = None
    embeddings = None

# Initialize the Pinecone client
try:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    # NOTE: The 'environment' parameter is deprecated and no longer needed in pinecone-client v3+
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("✅ Successfully connected to Pinecone!")
except Exception as e:
    print(f"❌ Error connecting to Pinecone: {e}")
    pc = None

def sanitize_filename(filename: str) -> str:
    """Sanitize to meet Pinecone's index name rules."""
    return "".join(c if c.isalnum() else '-' for c in filename).lower()

def extract_json_from_string(text: str) -> str:
    """
    Finds and extracts the first valid JSON object from a string.
    The LLM often wraps the JSON in Markdown backticks (```json ... ```).
    """
    # This regex pattern looks for a string that starts with '{' and ends with '}'
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        # This will help us debug if the LLM output is completely broken
        raise ValueError("No valid JSON object found in the LLM output.")
# --- 3. THE API ENDPOINT ---

@app.post("/api/process-pdf")
async def process_pdf_and_create_graph(file: UploadFile = File(...)):
  
    if not pc or not llm or not embeddings:
        raise HTTPException(status_code=500, detail="A required service (Pinecone or Azure OpenAI) is not initialized.")
    
    print(f"Processing file: {file.filename}")
    
    try:
        # 1. EXTRACT TEXT FROM PDF
        pdf_bytes = await file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = "".join(page.get_text() for page in pdf_document)
        pdf_document.close()

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF.")
        print(f"📄 Extracted {len(full_text)} characters.")

        # 2. SPLIT TEXT INTO MANAGEABLE CHUNKS
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_text(full_text)
        print(f"Splitting text into {len(chunks)} chunks.")

        # 3. CREATE OR CONNECT TO A PINECONE INDEX
        index_name = "nerv"
        print(f"Using Pinecone index: '{index_name}'")
        
        if index_name not in pc.list_indexes().names():
            print(f"Index '{index_name}' does not exist. Creating...")
            pc.create_index(
                name=index_name,
                dimension=3072,  # Dimension for Azure's text-embedding-3-small
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            print("✅ Index created successfully.")
        
        # 4. POPULATE THE INDEX (CREATE AND STORE EMBEDDINGS)
        print("Creating embeddings and upserting to Pinecone... (This may take a moment)")
        PineconeVectorStore.from_texts(
            texts=chunks, 
            embedding=embeddings, 
            index_name=index_name
        )
        print(f"✅ Successfully upserted {len(chunks)} chunks to the index.")

        # 5. GENERATE THE KNOWLEDGE GRAPH JSON
        print("🤖 Generating knowledge graph from full text...")
        json_string_from_ai = generate_graph_from_text(full_text)
        print("LLM RAW OUTPUT:", json_string_from_ai)  
        graph_data = json.loads(json_string_from_ai)
        print("✅ Knowledge graph generated.")
        
        # 6. RETURN THE GRAPH DATA AND THE INDEX NAME FOR FUTURE USE
        return {
            "message": "Document processed successfully.",
            "index_name": index_name,
            "graph_data": graph_data
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_graph_from_text(document_text: str) -> str:
    """Uses an LLM to create the relational JSON for the knowledge graph."""
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

    # The JSON extraction function is still needed, as the LLM might still add markdown
    return extract_json_from_string(llm_raw_output)