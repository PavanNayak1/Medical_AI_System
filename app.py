from flask import Flask, render_template, jsonify, request
import google.generativeai as genai
# from google.generativeai.types import Blob
from src.helper import text_embedding
from src.model import predict_eye_disease
from src.brain import predict_brain_tumor
from src.lungs import predict_chest_xray
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from services.voice import speech_to_text, text_to_speech
from openai import OpenAI
from dotenv import load_dotenv
from src.prompt import *
from PIL import Image
import uuid # For generating unique filenames(openai TTS required)
import io
import os
import asyncio
import edge_tts
from flask import send_from_directory # Add this to your flask imports
import re




app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


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

#cleaning text for TTS
def clean_text_for_tts(text):
    # Remove any unwanted characters or formatting that might affect TTS
    clean = re.sub(r'[*_]', '', text)
    clean = re.sub(r'[#+]','', clean) 
    return clean.strip() 
# This is the helper function to generate the voice
async def generate_neural_voice(text, filename):
    text = clean_text_for_tts(text)
    communicate = edge_tts.Communicate(text, "en-US-AvaNeural", rate="-5%", pitch="-15Hz")
    await communicate.save(filename)



@app.route("/")
def index():
    return render_template('chat.html')

# Allows Flask to serve the audio files to the frontend
@app.route('/static/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory('static/audio', filename)


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return "No image uploaded", 400

    file = request.files["image"]
    organ_type = request.form.get("organ")

    if file.filename == "":
        return "No selected file", 400

    # Read image safely
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    # --- SMART VALIDATION: Did the user upload the wrong image? ---
    try:
        validation_model = genai.GenerativeModel("gemini-1.5-flash")
        # We ask Gemini to do a quick sanity check
        val_prompt = "Look at this medical image. Is it an eye scan (retina/OCT), a brain MRI, a chest X-ray, or a skin image? Reply with ONLY one word: eye, brain, chest, skin, or unknown."
        val_response = validation_model.generate_content([val_prompt, image])
        detected_type = val_response.text.strip().lower()
        expected_type = {"eye": "eye", "brain": "brain", "lungs": "chest"}.get(organ_type)
        selected_label = {"eye": "Eye", "brain": "Brain", "lungs": "Chest"}.get(organ_type, "Unknown")

        # Check if Gemini thinks the image doesn't match their selected setting
        if expected_type and detected_type in ["eye", "brain", "chest", "skin"] and detected_type != expected_type:
            error_msg = f"Wait! It looks like you uploaded a **{detected_type.capitalize()}** image, but your current target is set to **{selected_label}**. Please change the target in the sidebar."
            
            # Generate audio for the error so the bot speaks it out loud!
            audio_file = f"error_{uuid.uuid4()}.mp3"
            audio_path = os.path.join("static/audio", audio_file)
            asyncio.run(generate_neural_voice(error_msg, audio_path))

            return jsonify({
                "prediction": {"disease": "Error", "confidence": 0},
                "chatbot_response": error_msg,
                "audio_url": f"/static/audio/{audio_file}"
            })
    except Exception as e:
        print("Validation error bypassed:", e)
        pass # If Gemini fails for some reason, just proceed to the ML model anyway
    # -----------------------------------------------------------------
    # If it passes validation, route it to your TensorFlow models!
    scan_name = "scan"
    if organ_type == "eye":
        prediction = predict_eye_disease(image)
        scan_name = "Retinal OCT scan"
    elif organ_type == "brain":
        prediction = predict_brain_tumor(image)
        scan_name = "Brain MRI scan"
    elif organ_type == "lungs":
        prediction = predict_chest_xray(image)
        scan_name = "Chest X-ray"
    else:
        return jsonify({"error": "Invalid organ type selected"}), 400

    # # 🔥 Model prediction happens HERE
    # prediction = predict_eye_disease(image)

    if prediction is None:
        return "Model could not classify the image", 500

    # 🔥 Send result to chatbot
    query = (
        f"The {scan_name} suggests {prediction['disease']} "
        f"with {prediction['confidence']*100:.1f}% confidence. "
        "Explain briefly in a patient-friendly way."
    )

    response = rag_chain.invoke({"input": query})

    # Generate Voice
    audio_file = f"speech_{uuid.uuid4()}.mp3"
    audio_path = os.path.join("static/audio", audio_file)
    os.makedirs("static/audio", exist_ok=True)
    asyncio.run(generate_neural_voice(response["answer"], audio_path))

    # Return Prediction, Text Reply, and Audio
    return jsonify({
        "prediction": prediction,
        "chatbot_response": response["answer"],
        "audio_url": f"/static/audio/{audio_file}"
    })
    
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

    # Generate Voice
    audio_file = f"speech_{uuid.uuid4()}.mp3"
    audio_path = os.path.join("static/audio", audio_file)
    os.makedirs("static/audio", exist_ok=True)
    asyncio.run(generate_neural_voice(reply_text, audio_path))

    # Return BOTH text and audio
    return jsonify({
        "reply": reply_text, 
        "audio_url": f"/static/audio/{audio_file}"
    })

@app.route("/voice", methods=["POST"])
def voice_route():

    print("📥 /voice route triggered")

    if "audio" not in request.files:
        print("❌ No audio file received")
        return jsonify({"error": "No audio file received"}), 400

    audio_file = request.files["audio"]

    input_path = f"temp_{uuid.uuid4()}.webm"
    audio_file.save(input_path)

    print("💾 Audio saved:", input_path)

    # Read audio bytes
    with open(input_path, "rb") as f:
        audio_bytes = f.read()

    print("⚡ Sending audio to Gemini STT...")

   # Gemini STT (Your existing logic)
    model = genai.GenerativeModel("gemini-2.5-flash")
    stt_response = model.generate_content(["Transcribe this audio to text only.", {"mime_type": "audio/webm", "data": audio_bytes}])
    user_text = stt_response.text

    # Get RAG Answer
    rag_response = rag_chain.invoke({"input": user_text})
    reply_text = str(rag_response["answer"])

    # Generate Voice
    out_audio_file = f"speech_{uuid.uuid4()}.mp3"
    audio_path = os.path.join("static/audio", out_audio_file)
    os.makedirs("static/audio", exist_ok=True)
    asyncio.run(generate_neural_voice(reply_text, audio_path))

    # Return Transcript, Text Reply, and Audio
    return jsonify({
        "transcript": user_text,
        "reply": reply_text,
        "audio_url": f"/static/audio/{out_audio_file}"
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8085, debug= True)
