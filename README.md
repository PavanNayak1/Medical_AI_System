# 🏥 QuickMed AI – Medical Imaging & Chat Assistant
MedInsight AI is an AI-powered medical assistant that analyzes medical images and answers health-related questions in a simple and structured way.

It supports:

👁 Eye (Retinal OCT)

🧠 Brain (MRI)

🫁 Chest (X-ray)

💬 Text-based medical questions

🎤 Voice interaction

This project is for educational and assistive use only.
<hr>

**🔍 What It Does**
🖼 Image Analysis

You can upload a medical scan, and the system will:

Predict the disease using a deep learning model

Show a confidence score

Explain the condition in clear language

Provide a voice response (optional)

Supported Conditions

**Eye (OCT):**

CNV

DME

DRUSEN

NORMAL

**Brain (MRI):**

Glioma

Meningioma

Pituitary Tumor

No Tumor

**Lungs (X-ray):**

Covid-19

Emphysema

Normal

Pneumonia (Bacterial)

Pneumonia (Viral)

Tuberculosis

**💬 Medical Chatbot**

The assistant can:

Explain diseases

Describe symptoms

Discuss causes

Talk about general management

Answer follow-up questions

It uses a medical knowledge base (RAG) to give accurate and relevant information.
<hr>

**🛠️ Technologies Used**


TensorFlow / Keras – Model inference

Flask – Backend

LangChain – RAG pipeline

Pinecone – Vector database

Gemini API – Language model

Edge TTS – Voice generation

HTML / CSS / JavaScript – Frontend
<hr>

**📁 Project Structure**


```
Medical_AI_System/
│
├── app.py
├── src/
│   ├── model.py
│   ├── brain.py
│   ├── heart.py
│   ├── helper.py
│   └── prompt.py
│
│
├── templates/
│   └── chat.html
│
├── static/
│   ├── style.css
│   └── audio/
│
├── models/
├── .env
├── requirements.txt
└── README.md
```


<hr>

****⚙️ Setup Instructions****

1️⃣ Clone the repository

git clone <your-github-repo-link>
cd Medical_AI_System

2️⃣ Create and activate a virtual environment
```
python -m venv medchatenv
source medchatenv/bin/activate   # Linux / Mac
# medchatenv\Scripts\activate    # Windows
```

3️⃣ Install dependencies
```
pip install -r requirements.txt
```
4️⃣ Configure environment variables

Create a .env file in the root directory:

```
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_gemini_api_key
```

5️⃣ Run the application
```
python app.py
```

The app will start at:

http://127.0.0.1:8080

<hr>

**⚠️ Important Disclaimer**

This project is for educational and assistive purposes only.
It is not a diagnostic tool and should not be used for medical decisions.

Always consult a qualified eye-care professional for diagnosis and treatment.
