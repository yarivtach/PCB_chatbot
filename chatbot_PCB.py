import os
from dotenv import load_dotenv
from langchain import hub # module for loading models
from langchain.text_splitter import RecursiveCharacterTextSplitter # module for splitting text
from langchain_community.document_loaders import WebBaseLoader # module for loading web data
from langchain_community.vectorstores import Chroma # module for storing data in a vector database
# from langchain_core.output_parsers import StrOutputParse # module for parsing output
from langchain_core.runnables import RunnablePassthrough # module for passing through data
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # module for loading openai models
import google.generativeai as genai
from pdf2image import convert_from_path

from ChatBot import ChatBot
from data_manipulated import Data_manipulated



load_dotenv()

# Set the path to the Poppler bin directory
poppler_path = r"c:\\Program Files\\poppler\\poppler-24.07.0\\Library\\bin\\"
os.environ["PATH"] += os.pathsep + poppler_path #

# ... rest of your existing code ...

#set up user_agent
user_agent = os.getenv("USER_AGENT")

# load the data from the pdf files
data_class = Data_manipulated()

#set up the chat model
model_name = os.getenv("MODEL_NAME")
chatbot = ChatBot()
if chatbot.chunks is None:
    print("Exiting due to lack of data.")
    exit()
pre_prompt = os.getenv("PRE_PROMPT")

#create the vector database for the chatbot
# qdrant_client = data_class.create_vector_database_for_quaries()

user_input = input("Enter your question, say 'exit' to end the conversation: ")
# response = qdrant_client.similarity_search(user_input)


collection = chatbot.RAG()
answer = collection.query(query_texts=[user_input], n_results=1)
print(answer)
