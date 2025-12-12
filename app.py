from flask import Flask, render_template, jsonify, request
from src.helper import text_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
# for our model
from PIL import Image
import io
import pickle  # Or your model's library (e.g., tensorflow, torch)



app = Flask(__name__)

model = None
try:
    with open('path/to/your/model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Model file not found. Upload will return a message.")

def predict_eye_disease(image):
    if model is None:
        return None  # Indicate no model
    # Preprocess and predict (replace with your logic)
    return {"disease": "CNV", "confidence": 0.95}

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


embeddings = text_embedding()

index_name = "eye-disease-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/upload", methods=["POST"])
def upload():
    if 'image' not in request.files:
        return "No image uploaded", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    image = Image.open(io.BytesIO(file.read()))
    prediction = predict_eye_disease(image)
    if prediction is None:
        return "We don't have the model to classify it."
    # Create query for chatbot
    query = f"What is {prediction['disease']}? Confidence: {prediction['confidence']}"
    response = rag_chain.invoke({"input": query})
    return str(response["answer"])

    
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)