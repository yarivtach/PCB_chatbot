from langchain_openai import ChatOpenAI, OpenAIEmbeddings # module for loading openai models
import google.generativeai as genai
from langchain_community.vectorstores import Qdrant
from dotenv import load_dotenv
import chromadb
from langchain.embeddings import OllamaEmbeddings
from sentence_transformers import SentenceTransformer
import os
import requests
from data_manipulated import Data_manipulated


class ChatBot:
    def __init__(self):
        load_dotenv()
        self.model_name = os.getenv("MODEL_NAME")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.model_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.gemini_api_key}"
        genai.configure(api_key=os.environ["GEMINI_API_KEY"]) 
        self.path_data_pdf = os.getenv("PATH_DATA_PDF")
        self.collection = None
        self.model = genai.GenerativeModel(self.model_name)
        self.data_manipulated = Data_manipulated()
        
        self.chunks = self.data_manipulated.return_chunks_from_path_data_pdf()
        if self.chunks is None:
            print("No data to process. Please ensure PDF files are present in the data directory.")
            return
        
            
        
        
            
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
        text_chunks = [docs.page_content for docs in self.chunks]
        
        
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(text_chunks)
        #print(f"embeddings: {embeddings}\n")
        return embeddings

        
    def RAG(self, query):
        #query the collection
        response = self.collection.query(query_texts=[query],n_results=1)
                                    
        relevant_document = response.get("documents")[0][0]
        answer = self.generate_response_gemini(query, relevant_document)
        
        return answer
    
    
    def parse_response(self, response_json):
        try:
            candidates = response_json.get('candidates', [])
            if candidates:
                content = candidates[0].get('content', {})
                parts = content.get('parts', [])
                if parts:
                    return parts[0].get('text', '')
        except (IndexError, KeyError, TypeError) as e:
            print(f"Error parsing response: {e}")
        

    def generate_response_gemini(self, query, context):
        pre_prompt = os.getenv("PRE_PROMPT", "")
        
        # Construct the full prompt by adding the pre-prompt, context, and query
        prompt = f"""
        {pre_prompt}
        context: {context}\n\n
        question: {query}\n\n
        answer:
        """
        
        # Prepare the data for the API request
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt  # The full prompt including pre-prompt, context, and query
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,  # Adjust this value as needed
                "maxOutputTokens": 500,  # Adjust this value as needed

    }
        }

        headers = {
            "Content-Type": "application/json"
        }

        try:
            # Make the API request to Gemini
            response = requests.post(self.model_url, json=data, headers=headers)

            # Print the response for debugging
            #print(f"Response: {response}\n")

            # Check for a successful response
            if response.status_code == 200:
                response_json = response.json()
                
                # Print the response JSON for debugging
                # print(f"Response JSON: {response_json}\n")

                # Extract the generated text from the response
                generated_text = self.parse_response(response_json)
                
               # print(f"generated_text: {generated_text}\n")
                return generated_text
            else:
                # Log and return error message if the API call failed
                print(f"Error: Received status code {response.status_code}")
                print(f"Response: {response.text}")
                return "Sorry, something went wrong while generating the response."

        except Exception as e:
            # Log and return error message if an exception occurred
            print(f"Error generating response: {e}")
            return "Sorry, I couldn't generate a response. Please try again."
