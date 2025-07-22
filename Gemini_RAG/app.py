# Import necessary libraries
import os
import glob
import gradio as gr
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import google.generativeai as genai
from load_dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables from .env file
load_dotenv()

# Specify the Google Generative AI model to use
MODEL = "gemini-1.5-flash"
# Define the name of the vector database
db_name = "vector_db"

# Configure the Google Generative AI API
genai.configure()
# Initialize the generative AI model
model = genai.GenerativeModel(MODEL)


# Get a list of folders in the knowledge-base directory
folders = glob.glob("knowledge-base/*")
# Define keyword arguments for the TextLoader
text_loader_kwargs = {'encoding': 'utf-8'}

# Initialize an empty list to store documents
documents =[]
# Iterate over each folder in the knowledge-base
for folder in folders:
    # Get the name of the folder
    doc_type = os.path.basename(folder)
    # Load documents from the folder
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    # Add document type metadata to each document and append to the documents list
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

# Create a text splitter to split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# Split the documents into chunks
chunks = text_splitter.split_documents(documents)

# Initialize embeddings using Google Generative AI
embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001',
                                          google_api_key=os.getenv("GOOGLE_API_KEY"))

# Delete the existing vector database if it exists
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

# Create a vector database from the chunks
vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=db_name)
# Print the number of documents in the vector database
print(f"Vectorstore created with {len(vector_db.get()["documents"])} documents")

# Get a sample embedding to determine its dimensions
sample = vector_db.get(include=["embeddings"], limit=1)
dimensions = len(sample["embeddings"][0])
# Print the dimensions of the embeddings
print(f"Vectorstore dimensions: {dimensions}")

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history",
                                  return_messages=True,
                                  output_key="answer")

# Initialize the language model using Google Generative AI
llm = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Create a retriever to fetch relevant documents from the vector database
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Create a conversational retrieval chain
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# Define a sample query
query = "Can you describe Insurellm in a few sentences"
# Get the result from the conversational retrieval chain
result = conversation_chain.invoke({"question":query})
# Print the answer
print(result["answer"])