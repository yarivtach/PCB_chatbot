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
os.environ["PATH"] += os.pathsep + poppler_path 
#set up user_agent
user_agent = os.getenv("USER_AGENT")


# load the data from the pdf files
# data_class = Data_manipulated()

#set up the chat model
model_name = os.getenv("MODEL_NAME")
chatbot = ChatBot()
if chatbot.chunks is None:
    print("Exiting due to lack of data.")
    exit()

#create the vector database for the chatbot
# qdrant_client = data_class.create_vector_database_for_quaries()

while True:
    user_input = input("Enter your question, say 'exit' to end the conversation: ")
    if user_input == "exit":
        break
    else:
        chat_answer = chatbot.RAG(user_input)
        print(str(chat_answer))

