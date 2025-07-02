import streamlit as st
import boto3
import os
import json
import time
import traceback
import logging
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3
from datetime import datetime
from PIL import Image
import pytesseract

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AskRockAI")

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = os.getenv("BEDROCK_INFERENCE_PROFILE_ARN")
DEFAULT_MAX_RETRIES = 7
DEFAULT_RETRY_DELAY_SECONDS = 3
DEFAULT_MAX_TOKENS = 25000

# Assume IAM Role for credentials
ROLE_ARN = "arn:aws:iam::207567766326:role/Workmates-SSO-L2SupportRole"
SESSION_NAME = "AskRockAISession"

try:
    logger.info(f"Assuming IAM Role: {ROLE_ARN} ...")
    base_session = boto3.Session()
    sts_client = base_session.client("sts", region_name=AWS_REGION)
    assumed_role = sts_client.assume_role(
        RoleArn=ROLE_ARN,
        RoleSessionName=SESSION_NAME,
        DurationSeconds=3600,
    )

    credentials = assumed_role['Credentials']

    bedrock_runtime = boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )

    logger.info("Successfully assumed role and created bedrock_runtime client.")

except Exception as e:
    logger.error(f"Failed to assume role: {e}")
    raise

def microphone_available() -> bool:
    try:
        sr.Microphone()
        return True
    except OSError as e:
        logger.warning("Microphone check failed: %s", e)
        return False

# Streamlit page setup
st.set_page_config(page_title="🪨🎙️ AskRock AI with Chat", page_icon="🪨", layout="wide")
st.title("🪨🎙️ AskRock AI: Bedrock Chat Assistant")
st.caption("_Now with Chat History, Save, and Optional Image OCR!_")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    max_tokens = st.slider("Max tokens:", 500, 30000, DEFAULT_MAX_TOKENS, step=500)
    max_retries = st.slider("Max retries:", 1, 10, DEFAULT_MAX_RETRIES)
    retry_delay = st.slider("Retry delay (s):", 1, 10, DEFAULT_RETRY_DELAY_SECONDS)
    auto_speak = st.checkbox("🔊 Auto-read answer aloud", value=True)
    theme = st.radio("🎨 Theme:", ["Light", "Dark"], index=0)
    st.divider()

if theme == "Dark":
    st.markdown(
        """
        <style>
        body { background-color: #0e1117; color: #fafafa; }
        </style>
        """,
        unsafe_allow_html=True,
    )

topic = st.text_input("🔎 Optional topic (e.g., Finance, Science):", value="General")
col1, col2 = st.columns(2)
with col1:
    st.subheader("💬 Type your question")
    user_query = st.text_area(" ", placeholder="Enter your question here...")

    st.subheader("🖼️ Optionally upload an image with your question")
    uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    extracted_text = ""
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        try:
            img = Image.open(uploaded_image)
            extracted_text = pytesseract.image_to_string(img)
            if extracted_text.strip():
                st.info(f"📝 Extracted text from image:\n\n{extracted_text.strip()}")
            else:
                st.warning("⚠️ No text could be extracted from the image.")
        except Exception as e:
            st.error(f"Failed to parse image: {e}")

with col2:
    st.subheader("🎤 Record your question")
    if microphone_available():
        if st.button("🎙️ Start Recording"):
            try:
                recognizer = sr.Recognizer()
                mic = sr.Microphone()
                with mic as source:
                    st.info("Listening... speak clearly.")
                    audio = recognizer.listen(source, timeout=10, phrase_time_limit=20)
                try:
                    text = recognizer.recognize_google(audio)
                    user_query = text
                    st.success(f"📝 Transcribed: {text}")
                except sr.UnknownValueError:
                    st.error("Could not understand audio. Please try again.")
                except sr.RequestError as e:
                    st.error(f"Speech recognition error: {e}")
            except OSError as e:
                st.error("🎤 No microphone detected on this device/environment. Please use text input instead.")
                logger.error("Microphone initialization error: %s", str(e))
    else:
        st.info("🎤 Microphone not available in this environment. Please use text input or upload an image.")

def build_prompt(question: str, topic: str, ocr_text: str) -> str:
    prompt = f"Topic: {topic}\n\nQuestion: {question}"
    if ocr_text.strip():
        prompt += f"\n\nAdditional context extracted from image:\n{ocr_text.strip()}"
    return prompt

def format_output(text: str) -> str:
    return text.strip()

def speak_text(text: str):
    engine = pyttsx3.init()
    engine.setProperty("rate", 180)
    engine.say(text)
    engine.runAndWait()

def attempt_model_invoke(payload: dict, retries: int, delay: int) -> str:
    last_exception = None
    for attempt in range(1, retries + 1):
        try:
            logger.info("Attempt %d: Sending request to Bedrock...", attempt)
            start_time = time.time()
            response = bedrock_runtime.invoke_model(
                modelId=MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload).encode("utf-8"),
            )
            elapsed = time.time() - start_time
            logger.info("Response received in %.2fs.", elapsed)
            raw_response = response["body"].read().decode("utf-8")
            logger.debug("Raw response: %s", raw_response)
            result = json.loads(raw_response)
            messages = result.get("content", [])
            if messages and isinstance(messages, list):
                return messages[0].get("text", "").strip()
            else:
                logger.warning("Unexpected response structure: %s", result)
                return None
        except Exception as e:
            logger.error("Attempt %d failed: %s", attempt, str(e))
            last_exception = e
            if attempt < retries:
                time.sleep(delay)
    if last_exception:
        raise last_exception

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("🚀 Get Answer"):
    if not user_query.strip() and not extracted_text.strip():
        st.error("Please type/record your question or upload an image with text.")
    else:
        combined_prompt = build_prompt(user_query, topic, extracted_text)
        with st.expander("🔍 Prompt Details"):
            st.code(combined_prompt, language="markdown")
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": combined_prompt}],
        }

        try:
            with st.spinner("Thinking... ⏳"):
                answer = attempt_model_invoke(payload, max_retries, retry_delay)
            if answer:
                formatted_answer = format_output(answer)
                st.success("✅ **Answer:**")
                st.markdown(f"### {formatted_answer}")
                st.session_state.chat_history.append(
                    {"question": user_query or '[Image-only question]', "answer": formatted_answer}
                )
                if auto_speak:
                    speak_text(formatted_answer)
            else:
                st.error("⚠️ No response received after retries.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.text(traceback.format_exc())

if st.session_state.chat_history:
    st.markdown("---")
    st.header("🗂️ Chat History")
    for i, entry in enumerate(reversed(st.session_state.chat_history), 1):
        st.markdown(f"**Q{i}:** {entry['question']}")
        st.markdown(f"**A{i}:** {entry['answer']}")

    if st.download_button(
        label="💾 Download Chat History",
        data="\n\n".join(
            f"Q: {e['question']}\nA: {e['answer']}" for e in st.session_state.chat_history
        ),
        file_name=f"askrock_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    ):
        st.success("✅ Chat history saved successfully!")
