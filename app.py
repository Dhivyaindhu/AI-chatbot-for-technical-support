import streamlit as st
from PIL import Image
from langchain_core.prompts import PromptTemplate
from groq import Groq

# -----------------------------
# Groq LLM Wrapper
# -----------------------------
class GroqChatLLM:
    def __init__(self, api_key, model="llama3-8b-8192"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def __call__(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a friendly IT support assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


# -----------------------------
# OCR Placeholder
# -----------------------------
def extract_text_from_image(image):
    return "Extracted text from uploaded image"


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI IT Support", page_icon="🛠️")
st.title("🛠️ AI IT Support Assistant (Groq)")

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
# Run Groq
# -----------------------------
API_KEY = st.secrets["GROQ_API_KEY"]
llm = GroqChatLLM(API_KEY)

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
