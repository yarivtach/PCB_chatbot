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
import chromadb

from pdf2image import convert_from_path

from ChatBot import ChatBot
from data_manipulated import Data_manipulated

        

def process_data(chatbot, client):
    if chatbot.chunks is None:
        print("Exiting due to lack of data.")
        return -1
    try:
        collection = client.get_collection(name="chatbot_collection")
        print("Existing collection found.")
    except chromadb.errors.InvalidCollectionException:
        print("Collection does not exist. Creating new collection...")
        collection = client.create_collection(name="chatbot_collection")
        print("New collection created.")
        
    my_embeddings = chatbot.convert_chunks_to_vector_database()
    my_documents = []
    my_ids = []
    #my_metadata = []
    for i, doc in enumerate(chatbot.chunks):
        my_ids.append(str(i))
        my_documents.append(doc.page_content)
        # my_metadata.append({"source": doc.metadata.get("source", "")})
        
    collection.add(
        documents=my_documents,
        ids=my_ids,
        embeddings=my_embeddings,
        # metadata=my_metadata
    )
    return collection


        
def main():
    load_dotenv()

    # Set the path to the Poppler bin directory
    poppler_path = r"c:\\Program Files\\poppler\\poppler-24.07.0\\Library\\bin\\"
    os.environ["PATH"] += os.pathsep + poppler_path 
    #set up user_agent
    user_agent = os.getenv("USER_AGENT")
    #set up the chat model
    model_name = os.getenv("MODEL_NAME")
    chatbot = ChatBot()
    client = chromadb.Client()

    collection = process_data(chatbot, client)
    if collection == -1:
        print("Exiting due to lack of data.")
        return
    if collection.count() == 0:
        print("populating collection with data")
        
    chatbot.collection = collection
    
    while True:
        user_input = input("Enter your question, say 'exit' to end the conversation: ")
        if user_input == "exit":
            break
        else:
            chat_answer = chatbot.RAG(user_input)
            print(str(chat_answer))
    

# main function
if __name__ == "__main__":
    main()
