import os
import requests
import torch
import numpy as np
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
import traceback
import soundfile as sf
from scipy.io.wavfile import write
from kokoro import KPipeline

# Load environment variables
load_dotenv()

# Constants
GROQ_API_KEY = "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"
URL = "https://api.groq.com/openai/v1/audio/transcriptions"

os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Convolabs AI Assistant"

# Global variables
model = None
tokenizer = None
messages = []

### 🚀 **Load LLaMA 3.2 1B with LoRA Fine-Tuned Weights**
def initialize_chat_model():
    """Load LLaMA 3.2-1B with LoRA fine-tuned weights."""
    global model, tokenizer, messages

    base_model_name = "unsloth/Llama-3.2-1B-Instruct"
    lora_model_path = "./myapp/lora_model"  # Path to your LoRA fine-tuned weights

    print("🔹 Loading Base Model:", base_model_name)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,  # Use FP16 for efficiency
        load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
        device_map="auto"  # Automatically selects GPU/CPU
    )

    # Load LoRA fine-tuned weights
    print("🔹 Loading LoRA Weights from:", lora_model_path)
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    # Merge LoRA with base model for inference
    model = model.merge_and_unload()

    print("✅ Model Loaded Successfully!")

    # messages = [
    #     {"role": "system", "content": "You are a professional AI assistant Named ConvoLabs AI for Customer Care Support"
    #                                   "Please keep your responses concise and to the point. "
    #                                   "Limit your replies to 2-3 sentences unless necessary."},
    #     {"role": "assistant", "content": "Hi, how can I help you today?"}
    # ]
    messages = [
    {"role": "system", "content": (
        "You are ConvoLabs AI, a professional AI assistant for Customer Care Support. "
        "Respond extremely concisely and accurately, following the examples."
    )},
    {"role": "user", "content": "What does ConvolabsAI do?"},
    {"role": "assistant", "content": (
        "ConvolabsAI provides AI agents for customer service, data analysis, and automation to streamline business operations."
    )},
    {"role": "user", "content": "How can your AI help with customer service?"},
    {"role": "assistant", "content": (
        "Our AI handles queries, automates responses, and predicts customer needs—available 24/7 across voice, email, and text."
    )},
    {"role": "user", "content": "How fast are your AI agents?"},
    {"role": "assistant", "content": (
        "They are lightning-fast, using models like Deepseek and Llama3, cutting wait times to seconds and improving over time."
    )},
    {"role": "user", "content": "How do they integrate with my tools?"},
    {"role": "assistant", "content": (
        "They easily integrate via APIs and sync quickly with most platforms. What platform are you using?"
    )}
]

initialize_chat_model()

### 🎙️ **Speech-to-Text (Transcription)**
def transcribe_audio_file(audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            audio_data = audio_file.read()

        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        files = {"file": ("audio.wav", audio_data, "audio/wav"), "model": (None, "whisper-large-v3")}

        response = requests.post(URL, headers=headers, files=files)

        if response.status_code == 200:
            return response.json().get("text", "")

        print("Transcription API Error:", response.status_code, response.text)  # Debug API response
        return None

    except Exception as e:
        print("Error in transcription:", e)
        return None

### 🤖 **LLM Inference**
def chat_with_model(user_input):
    """Chat with the fine-tuned model."""
    global messages

    messages.append({"role": "user", "content": user_input})

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(input_ids=inputs, max_new_tokens=256, temperature=1.0, min_p=0.1)
    ai_response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1].strip()

    messages.append({"role": "assistant", "content": ai_response})
    return ai_response

### 🔊 **Text-to-Speech**
def text_to_combined_audio(text, output_filename, lang_code='a', voice='af_bella'):
    """Convert text to speech."""
    pipeline = KPipeline(lang_code=lang_code)
    
    text_chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
    audio_files = []

    for i, text_chunk in enumerate(text_chunks):
        generator = pipeline(text_chunk, voice=voice, speed=1, split_pattern=r'\n+')
        for j, (gs, ps, audio) in enumerate(generator):
            filename = f"chunk_{i}_{j}.wav"
            sf.write(filename, audio, 24000)
            audio_files.append(filename)

    audio_data = [sf.read(file)[0] for file in audio_files]
    combined_audio = np.concatenate(audio_data)
    sf.write(output_filename, combined_audio, 24000)

    for file in audio_files:
        os.remove(file)

### 🌍 **Django API for Processing Requests**
@csrf_exempt
def process_audio_request(request):
    try:
        if request.method != "POST":
            return JsonResponse({"error": "Only POST requests allowed"}, status=405)

        if "audio" not in request.FILES:
            return JsonResponse({"error": "No audio file provided."}, status=400)

        # Save uploaded voice
        audio_file = request.FILES["audio"]
        audio_path = "./uploaded_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())

        # 🎙️ Transcribe
        transcription = transcribe_audio_file(audio_path)
        if not transcription:
            return JsonResponse({"error": "Transcription failed."}, status=500)

        # 🤖 Run LoRA Model
        ai_response = chat_with_model(transcription)
        if not ai_response:
            return JsonResponse({"error": "No response from AI."}, status=500)

        # 🔊 Convert to Speech
        text_to_combined_audio(ai_response, "final_audio.wav")

        # 📂 Read final_audio.wav and encode as base64
        with open("final_audio.wav", "rb") as f:
            wav_data = f.read()
        encoded_wav = base64.b64encode(wav_data).decode("utf-8")

        return JsonResponse({
            "user_transcription": transcription,
            "ai_text_response": ai_response,
            "audio_base64": encoded_wav
        })

    except Exception as e:
        error_message = traceback.format_exc()  # Get detailed error trace
        print("Server Error:", error_message)  # Print error to terminal
        return JsonResponse({"error": "Server error occurred", "details": str(e)}, status=500)
