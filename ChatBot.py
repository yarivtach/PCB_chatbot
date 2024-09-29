from langchain_openai import ChatOpenAI, OpenAIEmbeddings # module for loading openai models
import google.generativeai as genai
from langchain_community.vectorstores import Qdrant
from dotenv import load_dotenv
import chromadb
from langchain.embeddings import OllamaEmbeddings
import os

from data_manipulated import Data_manipulated


class ChatBot:
    def __init__(self):
        load_dotenv()
        print(f"check at data_manipulated.py\n")
        self.model_name = os.getenv("MODEL_NAME")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.qdrant_url = os.getenv("QDRANT_URL")
        genai.configure(api_key=os.environ["GEMINI_API_KEY"]) 
        self.path_data_pdf = os.getenv("PATH_DATA_PDF")
        print(f"create chatbot inside ChatBot.py\n")
        self.model = genai.GenerativeModel(self.model_name)
        self.data_manipulated = Data_manipulated()
        self.chunks = self.data_manipulated.return_chunks_from_path_data_pdf()
        if self.chunks is None:
            print("No data to process. Please ensure PDF files are present in the data directory.")
            return
        
        # self.vector_RAG_database = self.data_manipulated.prepare_data_for_chatbot()



    def generate_response(self, query):
        response = self.model.generate_content(query)
        return response.text
    
    def create_vector_database_for_quaries(self):
        qdrant_client = Qdrant.from_documents(
            documents=self.chunks,
            embedding=self.vector_RAG_database,
            url=self.qdrant_url,
            collection_name="chatbot_collection",
            api_key=self.qdrant_api_key
        )
        return qdrant_client

        
        #convert the chunks into a vector database - basicly for user input
    def convert_chunks_to_vector_database(self):    
        text_from_chunks = []
        for idx, chunk in enumerate(self.chunks):
            # Print the chunk to debug its content and type
            print(f"Processing chunk {idx}: type {type(chunk)}, content: {chunk}")
            
            # Extract the text from the chunk based on its type
            if isinstance(chunk, dict):
                # If chunk is a dictionary, extract the 'text' key
                text = chunk.get('text', '')
            elif isinstance(chunk, tuple):
                # If chunk is a tuple, decide which element contains the text
                # Assuming the text is the second element
                if len(chunk) > 1:
                    text = chunk[1]
                else:
                    text = ''
            elif isinstance(chunk, str):
                # If chunk is already a string, use it directly
                text = chunk
            else:
                # For any other type, convert it to a string
                text = str(chunk)
            
            # Ensure the text is a string and not empty
            if text and isinstance(text, str):
                text_from_chunks.append(text)
            else:
                print(f"Skipping chunk {idx} due to invalid text content.")
        
        # Now text_from_chunks is a list of strings
        # Use a valid model name for embedding
        embeddings_response = genai.embed_content(
            model="models/embedding-gecko-001",
            content=text_from_chunks,
            task_type="retrieval_document"
        )
        
        # Extract embeddings from the response
        embeddings = [item.embedding.value for item in embeddings_response]
        
        return embeddings

        
    def RAG(self):
        client = chromadb.Client()
        texts = list(self.chunks)
        collection = client.create_collection("chatbot_collection")
        collection.add(
            documents=texts,
            ids=[str(i) for i in range(len(self.chunks))],
            embeddings=self.convert_chunks_to_vector_database()
        )
        return collection
