import streamlit as st
from PIL import Image
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests
import json

# -----------------------------
# Gemini API wrapper for LangChain
# -----------------------------
class GeminiChatLLM:
    """
    Simple wrapper to use Gemini model with LangChain workflow
    """
    def __init__(self, api_key, model="models/gemini-2.5-pro"):
        self.api_key = api_key
        self.model = model
        self.api_url = f"https://gemini.googleapis.com/v1/models/{model}:generateMessage"

    def __call__(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "prompt": prompt,
            "temperature": 0.7,
            "maxOutputTokens": 500
        }
        response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            return result.get("candidates", [{}])[0].get("content", "No response from Gemini.")
        else:
            return f"Error: {response.status_code}, {response.text}"

# -----------------------------
# OCR Placeholder
# -----------------------------
def extract_text_from_image(image):
    # Replace with pytesseract or other OCR service
    return "Extracted text from uploaded image"

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AI IT Support with LangChain & Gemini Model")

st.header("Step 1: Enter your query")
user_question = st.text_input("Describe your issue or question:")

st.header("Step 2: Upload a screenshot or error photo (optional)")
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

image_text = None
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_text = extract_text_from_image(image)

# -----------------------------
# LangChain Prompt Template
# -----------------------------
template = """
You are a friendly IT support assistant. Solve the user's problem clearly and step-by-step.
If there is an image with text, use it to understand the problem better.
Detect the user's preferred language. If uncertain, respond in English.
Provide concise and easy-to-follow steps.

User's question: {user_question}
{image_text}
"""

prompt = PromptTemplate(
    input_variables=["user_question", "image_text"],
    template=template
)

# -----------------------------
# LangChain + Gemini Execution
# -----------------------------
API_KEY = "AIzaSyBpthm2RiMSSHBRDxwvccDsuI_174bIRbs"
gemini_llm = GeminiChatLLM(API_KEY)

if st.button("Get IT Support Response"):
    if user_question.strip() == "":
        st.warning("Please enter your question!")
    else:
        # Fill prompt template
        filled_prompt = prompt.format(user_question=user_question, image_text=image_text or "")
        # Get response from Gemini via LangChain workflow
        response = gemini_llm(filled_prompt)
        st.subheader("IT Support Response")
        st.write(response)
