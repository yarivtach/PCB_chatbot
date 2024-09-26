import os
from langchain_community.document_loaders import unstructuredPDFLoader
from langchain_community.document_loaders import onlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
from dotenv import load_dotenv

class Data_manipulated:
    def __init__(self, path_data_pdf):
        load_dotenv()
        self.path_data_pdf = path_data_pdf
        self.model_name = os.getenv("MODEL_NAME")

# load all pdf files from the path
    def load_pdf_files(self):
        array_pdf = []
        for file in os.listdir(self.path_data_pdf):
            if file.endswith(".pdf"):
                array_pdf.append(file)
        return array_pdf
    
#upload all pdf files to the database
    def upload_pdf_files(self, array_pdf):
        if len(array_pdf) != 0:
            for pdf in array_pdf:
                loader = unstructuredPDFLoader(self.path_data_pdf + pdf)
                data = loader.load()
                return data
        else:
            print("No pdf files found")
            return None

#extract the text from the pdf files
    def extract_text_from_pdf(self, data):
        text = ""
        for page in data:
            text += page.page_content
        return text


        
#split the text into chunks
    def split_text_into_chunks(self, data):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(data)
        return chunks
    
    #return chanks from path data pdf
    def return_chunks_from_path_data_pdf(self):
        data = self.upload_pdf_files(self.load_pdf_files())
        chunks = self.split_text_into_chunks(data)
        return chunks
        
#convert text to vectors
    def convert_text_to_vectors(self, text, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

#convert the chunks into a vector database - basicly for user input
    def convert_chunks_to_vector_database(self, chunks, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        embeddings = []
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True) #tokenize the text return tensor "pt" means pytorch
            outputs = model(**inputs) #pass the tokenized text to the model the ** is used to unpack the dictionary
            embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy()) #append the embeddings to the list, detach the numpy array from the graph and convert it to a numpy array
        return embeddings

#store the embeddings in a vector database
    def store_embeddings_in_vector_database(self, embeddings):
        vector_database = faiss.IndexFlatL2(embeddings[0].shape[1]) 
        vector_database.add(embeddings)
        return vector_database
    


#prepare data for the chatbot
    def prepare_data_for_chatbot(self):
        array_pdf = self.load_pdf_files()
        data = self.upload_pdf_files(array_pdf)
        chunks = self.split_text_into_chunks(data)
        vector_store = self.convert_chunks_to_vector_database(chunks, model_name=self.model_name)
        vector_database = self.store_embeddings_in_vector_database(vector_store)
        return vector_database

