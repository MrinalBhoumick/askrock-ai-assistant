# 🪨🎙️ AskRock AI: Bedrock Chat Assistant

AskRock AI is a multimodal Q&A assistant built with Streamlit, AWS Bedrock, speech recognition, and OCR.

It lets you:
✅ Ask questions by typing, speaking, or uploading an image containing text.  
✅ Automatically extracts text from uploaded images using OCR (Tesseract) and includes it in your question prompt.  
✅ Uses Amazon Bedrock to generate answers with Anthropic Claude/Titan models.  
✅ Keeps a chat history you can review or download.
✅ Optionally reads answers aloud.

---

## 🚀 Features

- 🔎 **Typed, spoken, or image-based questions**
- 🖼️ **OCR support** with pytesseract for text extraction from images
- 🎤 **Speech recognition** using Google Speech API
- 🪨 **Bedrock-powered answers** with Anthropic models
- 🗂️ **Chat history** with save/download feature
- 🎧 **Optional text-to-speech** to read answers aloud
- 🌗 **Light/Dark theme**

---

## 🛠️ Requirements

- Python 3.8+
- Tesseract OCR installed and in your PATH
- AWS credentials with access to Amazon Bedrock (configured via environment or AWS CLI)

---

## 🔧 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/MrinalBhoumick/askrock-ai-assistant
   cd askrock-ai-assistant
````

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract**

   * **Linux (Debian/Ubuntu):**

     ```bash
     sudo apt update && sudo apt install tesseract-ocr
     ```
   * **macOS:**

     ```bash
     brew install tesseract
     ```
   * **Windows:**

     * Download from [UB Mannheim Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki).
     * Add the install folder (e.g., `C:\Program Files\Tesseract-OCR`) to your system PATH.

4. **Set environment variables**

   Create a `.env` file in the project root with:

   ```
   BEDROCK_INFERENCE_PROFILE_ARN=your-bedrock-model-id
   AWS_REGION=your-region
   ```

   Make sure your AWS credentials are configured (e.g., using `aws configure`).

---

## ▶️ Running the App

```bash
streamlit run app.py
```

---

## 🎨 Screenshots

![AskRock App](./screenshot.png)

---

## 🤖 Usage

1. Type your question, speak it, or upload an image containing text.
2. Click **🚀 Get Answer**.
3. View the answer, listen to it if text-to-speech is enabled, and see your chat history.
4. Download the chat log if needed.

---

## 📚 Tech Stack

* [Streamlit](https://streamlit.io/) for the UI
* [AWS Bedrock](https://aws.amazon.com/bedrock/) for inference
* [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) for voice input
* [pytesseract](https://pypi.org/project/pytesseract/) for OCR
* [pyttsx3](https://pypi.org/project/pyttsx3/) for text-to-speech

---

## 📝 License

MIT License

---

## 🙏 Credits

Created by [Mrinal Bhoumick](https://github.com/MrinalBhoumick).