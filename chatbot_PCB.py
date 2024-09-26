import os
from dotenv import load_dotenv
from langchain import hub # module for loading models
from langchain.text_splitter import RecursiveCharacterTextSplitter # module for splitting text
from langchain_community.document_loaders import WebBaseLoader # module for loading web data
from langchain_community.vectorstores import Chroma # module for storing data in a vector database
from langchain_core.output_parsers import StrOutputParse # module for parsing output
from langchain_core.runnables import RunnablePassthrough # module for passing through data
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # module for loading openai models
import google.generativeai as genai


from ChatBot import ChatBot
from data_manipulated import Data_manipulated



load_dotenv()


#set up the chat model
model_name = os.getenv("MODEL_NAME")
chatbot = ChatBot(model_name)
pre_prompt = os.getenv("PRE_PROMPT")

# load the data from the pdf files
Path_data_pdf = os.getenv("LOCAL_DATA_PDF")
data_class = Data_manipulated(Path_data_pdf)

#create the vector database for the chatbot
qdrant_client = data_class.create_vector_database_for_quaries()

user_input = input("Enter your question, say 'exit' to end the conversation: ")
response = qdrant_client.similarity_search(user_input)
print(response)
# #conversation loop
# while user_input != "exit":
#     response = chatbot.generate_response(user_input)
#     print(response)