from langchain_openai import ChatOpenAI, OpenAIEmbeddings # module for loading openai models
import google.generativeai as genai
from langchain_community.vectorstores import Qdrant
from dotenv import load_dotenv
import os

from data_manipulated import Data_manipulated


class ChatBot:
    def __init__(self):
        load_dotenv()
        self.model_name = os.getenv("MODEL_NAME")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.qdrant_url = os.getenv("QDRANT_URL")
        genai.configure(api_key=os.environ["GEMINI_API_KEY"]) 
        self.path_data_pdf = os.getenv("PATH_DATA_PDF")
        self.model = genai.GenerativeModel(self.model_name)
        self.data_manipulated = Data_manipulated()
        self.chanks = self.data_manipulated.return_chunks_from_path_data_pdf()
        self.vector_RAG_database = self.data_manipulated.prepare_data_for_chatbot()



    def generate_response(self, query):
        response = self.model.generate_content(query)
        return response.text
    
    def create_vector_database_for_quaries(self):
        qdrant_client = Qdrant.from_documents(
            documents=self.chanks,
            embedding=self.vector_RAG_database,
            url=self.qdrant_url,
            collection_name="chatbot_collection",
            api_key=self.qdrant_api_key
        )
        return qdrant_client

        
        
        