from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import boto3
import pyttsx3
import pytesseract
import speech_recognition as sr
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# ─────────────────────── 1. Logging & ENV ────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("AskRockAI")

load_dotenv()

AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID: str = os.getenv("BEDROCK_INFERENCE_PROFILE_ARN", "")
DEFAULT_MAX_RETRIES = 7
DEFAULT_RETRY_DELAY_SECONDS = 3
DEFAULT_MAX_TOKENS = 25_000

ROLE_ARN = "arn:aws:iam::207567766326:role/Workmates-SSO-L2SupportRole"
SESSION_NAME = "AskRockAISession"

# ───────────────────── 2. AWS client bootstrap ───────────────────

def _session_from_streamlit_secrets() -> Optional[boto3.Session]:
    """Return a boto3.Session built from `st.secrets['aws']` if present."""
    secret_block = st.secrets.get("aws", None)
    if not secret_block:
        return None
    key = secret_block.get("access_key_id")
    sec = secret_block.get("secret_access_key")
    tok = secret_block.get("session_token")  # optional
    if key and sec:
        logger.info("🔑 Using AWS credentials from st.secrets")
        return boto3.Session(
            aws_access_key_id=key,
            aws_secret_access_key=sec,
            aws_session_token=tok,
            region_name=AWS_REGION,
        )
    return None

def get_bedrock_client() -> boto3.client:  # type: ignore[return-value]
    """Return a Bedrock Runtime client after assuming ROLE_ARN."""
    boto_session = _session_from_streamlit_secrets() or boto3.Session(region_name=AWS_REGION)

    if boto_session.get_credentials() is None:
        msg = (
            "🛑  No AWS credentials were found.\n\n"
            "Add them either:\n"
            " • In Streamlit Cloud as secrets: `aws.access_key_id`, `aws.secret_access_key`\n"
            " • Environment vars AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY\n"
            " • An AWS profile (AWS_PROFILE) or instance role."
        )
        logger.error(msg)
        st.error(msg)
        st.stop()

    try:
        sts = boto_session.client("sts")
        caller = sts.get_caller_identity()["Arn"]
        logger.info("🔐 Base identity: %s", caller)
        logger.info("Assuming role %s …", ROLE_ARN)
        assumed = sts.assume_role(
            RoleArn=ROLE_ARN,
            RoleSessionName=SESSION_NAME,
            DurationSeconds=3600,
        )["Credentials"]

        return boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=assumed["AccessKeyId"],
            aws_secret_access_key=assumed["SecretAccessKey"],
            aws_session_token=assumed["SessionToken"],
        )

    except Exception as exc:
        logger.exception("Failed to assume role or create Bedrock client")
        st.error(f"Unexpected AWS error: {exc}")
        st.stop()

bedrock_runtime = get_bedrock_client()
logger.info("Bedrock client initialised ✔")

# ─────────────────────── 3. Helper functions ─────────────────────

def microphone_available() -> bool:
    try:
        sr.Microphone()
        return True
    except OSError as e:
        logger.warning("Microphone check failed: %s", e)
        return False

def build_prompt(question: str, topic: str, ocr_text: str) -> str:
    prompt = f"Topic: {topic}\n\nQuestion: {question}"
    if ocr_text.strip():
        prompt += f"\n\nAdditional context extracted from image:\n{ocr_text.strip()}"
    return prompt

def format_output(text: str) -> str:
    return text.strip()

def speak_text(text: str) -> None:
    engine = pyttsx3.init()
    engine.setProperty("rate", 180)
    engine.say(text)
    engine.runAndWait()

def attempt_model_invoke(payload: Dict[str, Any], retries: int, delay: int) -> Optional[str]:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            logger.info("Attempt %d: sending request to Bedrock …", attempt)
            start = time.time()
            response = bedrock_runtime.invoke_model(
                modelId=MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload).encode("utf-8"),
            )
            logger.info("Response in %.2fs", time.time() - start)
            raw = response["body"].read().decode("utf-8")
            logger.debug("Raw response: %s", raw)
            data = json.loads(raw)
            messages = data.get("content", [])
            if isinstance(messages, list) and messages:
                return messages[0].get("text", "").strip()
            logger.warning("Unexpected response structure: %s", data)
            return None
        except Exception as exc:
            logger.error("Attempt %d failed: %s", attempt, exc)
            last_exc = exc
            if attempt < retries:
                time.sleep(delay)
    if last_exc:
        raise last_exc
    return None

# ───────────────────────── 4. UI – Streamlit ─────────────────────

st.set_page_config(page_title="🪨🎙️ AskRock AI with Chat", page_icon="🪨", layout="wide")
st.title("🪨🎙️ AskRock AI: Bedrock Chat Assistant")
st.caption("_Now with Chat History, Save, and Optional Image OCR!_")

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
    user_query = st.text_area(" ", placeholder="Enter your question here…")

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
        except Exception as exc:
            st.error(f"Failed to parse image: {exc}")

with col2:
    st.subheader("🎤 Record your question")
    if microphone_available():
        if st.button("🎙️ Start Recording"):
            try:
                recognizer = sr.Recognizer()
                mic = sr.Microphone()
                with mic as source:
                    st.info("Listening… speak clearly.")
                    audio = recognizer.listen(source, timeout=10, phrase_time_limit=20)
                try:
                    text = recognizer.recognize_google(audio)
                    user_query = text
                    st.success(f"📝 Transcribed: {text}")
                except sr.UnknownValueError:
                    st.error("Could not understand audio. Please try again.")
                except sr.RequestError as exc:
                    st.error(f"Speech recognition error: {exc}")
            except OSError as exc:
                st.error("🎤 No microphone detected on this device/environment.")
                logger.error("Microphone init error: %s", exc)
    else:
        st.info("🎤 Microphone not available in this environment. Please use text input or image upload.")

# ─────────────────── 5. Chat handling & history ──────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, str]] = []

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
            with st.spinner("Thinking… ⏳"):
                answer = attempt_model_invoke(payload, max_retries, retry_delay)

            if answer:
                formatted = format_output(answer)
                st.success("✅ **Answer:**")
                st.markdown(f"### {formatted}")

                st.session_state.chat_history.append({
                    "question": user_query or "[Image-only question]",
                    "answer": formatted,
                })

                if auto_speak:
                    speak_text(formatted)
            else:
                st.warning("⚠️ Model did not return a valid response.")
        except Exception as exc:
            logger.exception("Failed to invoke model")
            st.error(f"❌ Error while generating answer: {exc}")

if st.session_state.chat_history:
    st.subheader("🗃️ Chat History")
    for idx, entry in enumerate(reversed(st.session_state.chat_history), start=1):
        with st.expander(f"💬 Q{idx}: {entry['question']}"):
            st.markdown(f"**Answer:** {entry['answer']}")
