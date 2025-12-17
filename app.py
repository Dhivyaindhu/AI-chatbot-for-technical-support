import streamlit as st
from PIL import Image
from langchain_core.prompts import PromptTemplate
import requests
import json
# Gemini API wrapper
# -----------------------------
class GeminiChatLLM:
    def __init__(self, api_key, model="models/gemini-2.5-flash"):
        self.api_key = api_key
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent"

    def __call__(self, prompt):
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        response = requests.post(
            f"{self.api_url}?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )

        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return f"Error {response.status_code}: {response.text}"


# -----------------------------
# OCR Placeholder
# -----------------------------
def extract_text_from_image(image):
    return "Extracted text from uploaded image"


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI IT Support", page_icon="🛠️")
st.title("🛠️ AI IT Support Assistant (Gemini)")

user_question = st.text_input("Describe your issue:")

uploaded_file = st.file_uploader(
    "Upload screenshot (optional)",
    type=["png", "jpg", "jpeg"]
)

image_text = ""
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)
    image_text = extract_text_from_image(image)


# -----------------------------
# Prompt Template
# -----------------------------
prompt_template = """
You are a friendly IT support assistant.
Explain the solution step-by-step in simple language.

User problem:
{user_question}

Image text (if any):
{image_text}
"""

prompt = PromptTemplate(
    input_variables=["user_question", "image_text"],
    template=prompt_template
)


# -----------------------------
# Run Gemini (API key directly here)
# -----------------------------
API_KEY = "AIzaSyBpthm2RiMSSHBRDxwvccDsuI_174bIRbs"
llm = GeminiChatLLM(API_KEY)

if st.button("Get Solution"):
    if not user_question.strip():
        st.warning("Please enter a problem.")
    else:
        final_prompt = prompt.format(
            user_question=user_question,
            image_text=image_text
        )
        answer = llm(final_prompt)
        st.subheader("Solution")
        st.write(answer)

