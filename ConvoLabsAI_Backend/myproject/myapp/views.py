# myapp/views.py

import os
import torch
import numpy as np
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
import traceback
import soundfile as sf
from kokoro import KPipeline # Your TTS library
import whisper # Offline Whisper STT

# Load environment variables from .env file
load_dotenv()

# --- Global Model Variables ---
llm_model = None
llm_tokenizer = None
llm_messages = [] # Stores conversation history for the LLM
stt_model = None    # For Whisper STT
tts_pipeline = None # For Kokoro TTS

# --- Configuration ---
# Correct Mistral Base Model for your LoRA fine-tune
BASE_LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
LORA_MODEL_PATH = "./myapp/lora_model"  # Path to your LoRA adapter directory

# Whisper STT model size ("tiny", "base", "small", "medium", "large-v3")
WHISPER_MODEL_SIZE = "base" # Start with "base" for speed; increase for accuracy if needed

# Kokoro TTS Configuration (Adjust as needed)
KOKORO_LANG_CODE = 'en' # Assuming English, check Kokoro docs for other languages
KOKORO_VOICE = 'en_us_jessica' # Example voice, check Kokoro docs for available voices for your lang_code
KOKORO_TTS_SPEED = 1.0

# --- Determine Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using AI device: {DEVICE}")

# --- Helper to check if 4-bit loading should be attempted ---
# bitsandbytes is primarily for CUDA. It might have experimental CPU support but can be slow/problematic.
# Unsloth's benefits are most pronounced with its specialized kernels, often on CUDA.
# If LORA_MODEL_PATH contains an adapter trained with 4-bit quantization (e.g. via Unsloth's FastLanguageModel),
# the base model here should also be loaded with 4-bit compatibility if possible.
# For simplicity, if on CPU, we'll default to float32/float16 to avoid potential bitsandbytes CPU issues.
USE_4BIT_LOADING = (DEVICE == "cuda") # Enable 4-bit only on CUDA by default for stability

### 🚀 **Initialize Fine-Tuned LLM (Mistral 7B Instruct v0.3 + LoRA)**
def initialize_chat_model():
    global llm_model, llm_tokenizer, llm_messages
    if llm_model is not None:
        print("✅ LLM (Mistral + LoRA) already loaded.")
        return

    print(f"🔹 Loading Base LLM: {BASE_LLM_MODEL_NAME}")
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(BASE_LLM_MODEL_NAME)
        # Mistral tokenizers might not have a pad_token set. Common practice is to use eos_token.
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
            print("ℹ️ Tokenizer pad_token set to eos_token.")

        model_kwargs = {
            "torch_dtype": torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if DEVICE == "cuda" else torch.float32),
            "device_map": "auto"
        }
        
        if USE_4BIT_LOADING:
            print("Attempting to load base model with 4-bit quantization.")
            # This requires bitsandbytes correctly installed and compatible with your CUDA version.
            # For Unsloth-trained LoRAs, the base model should be loaded in a way compatible with how Unsloth
            # expects it for applying the adapter. Unsloth itself handles the complexities if you use
            # FastLanguageModel.from_pretrained to load BOTH base and adapter.
            # When loading separately like this, ensure base model loading matches Unsloth's typical setup.
            model_kwargs["load_in_4bit"] = True
            # You might need bnb_4bit_compute_dtype if using 4-bit:
            # model_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16 
        else:
            print(f"ℹ️ 4-bit quantization not enabled for LLM on {DEVICE}.")


        base_model_for_peft = AutoModelForCausalLM.from_pretrained(
            BASE_LLM_MODEL_NAME,
            **model_kwargs
        )

        print(f"🔹 Loading LoRA Adapter from: {LORA_MODEL_PATH}")
        if not os.path.isdir(LORA_MODEL_PATH): # Check if it's a directory
            raise FileNotFoundError(f"LoRA model path is not a directory or not found: {LORA_MODEL_PATH}")

        # Load the LoRA adapter. PEFT will automatically infer the PEFT config from the adapter_config.json.
        llm_model = PeftModel.from_pretrained(base_model_for_peft, LORA_MODEL_PATH)
        
        print("🔹 Merging LoRA adapter with base model for inference...")
        llm_model = llm_model.merge_and_unload() # Merge for faster inference
        llm_model.eval() # Set to evaluation mode

        print("✅ LLM (Mistral + LoRA) Loaded Successfully!")

        # Conversation history initialization
        llm_messages = [
            {"role": "system", "content": (
                "You are ConvoLabs AI, a professional AI assistant for Customer Care Support, powered by Mistral. "
                "Your responses should be concise, accurate, and directly answer the user's questions, "
                "similar to the examples provided. Aim for 1-3 sentences unless more detail is essential."
            )},
            # Add a few high-quality examples relevant to your fine-tuning
            {"role": "user", "content": "What services does ConvolabsAI offer?"},
            {"role": "assistant", "content": "ConvolabsAI specializes in AI-driven agents for customer service, data analytics, and business process automation."},
            {"role": "user", "content": "How quickly can the AI agents respond?"},
            {"role": "assistant", "content": "Our AI agents provide near-instantaneous responses, typically within seconds, ensuring minimal customer wait times."}
        ]
    except Exception as e:
        print(f"❌ Error initializing LLM (Mistral + LoRA): {e}")
        print(traceback.format_exc())
        llm_model = None # Ensure model is None if loading failed

### 🎙️ **Initialize Offline Speech-to-Text (Whisper)**
def initialize_stt_model():
    global stt_model
    if stt_model is not None:
        print("✅ STT Model (Whisper) already loaded.")
        return
    try:
        print(f"🔹 Loading STT Model (Whisper {WHISPER_MODEL_SIZE}) to {DEVICE}...")
        stt_model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)
        print(f"✅ STT Model (Whisper {WHISPER_MODEL_SIZE}) Loaded Successfully!")
    except Exception as e:
        print(f"❌ Error initializing STT model: {e}")
        print(traceback.format_exc())
        stt_model = None

### 🔊 **Initialize Offline Text-to-Speech (Kokoro)**
def initialize_tts_pipeline():
    global tts_pipeline
    if tts_pipeline is not None:
        print("✅ TTS Pipeline (Kokoro) already initialized.")
        return
    try:
        print(f"🔹 Initializing TTS Pipeline (Kokoro - Lang: {KOKORO_LANG_CODE}, Voice: {KOKORO_VOICE})...")
        tts_pipeline = KPipeline(lang_code=KOKORO_LANG_CODE)
        # You might need to pre-load or check the specific voice if Kokoro requires it.
        # For now, assuming KPipeline loads defaults or the voice is applied during generation.
        print("✅ TTS Pipeline (Kokoro) Initialized Successfully!")
    except Exception as e:
        print(f"❌ Error initializing TTS pipeline (Kokoro): {e}")
        print(traceback.format_exc())
        tts_pipeline = None

# --- Call initializers on Django startup (or first request) ---
models_initialized_flag = False
def ensure_models_initialized():
    global models_initialized_flag
    if not models_initialized_flag:
        print("--- Initializing AI Models (STT, LLM, TTS) ---")
        initialize_stt_model()    # STT first (often smaller)
        initialize_chat_model()   # LLM next (can be large)
        initialize_tts_pipeline() # TTS last
        models_initialized_flag = True
        print("--- AI Models Initialization Attempt Complete ---")
    # Sanity check after initialization attempt
    if not stt_model: print("⚠️ STT model failed to load or is not available.")
    if not llm_model: print("⚠️ LLM (Mistral+LoRA) failed to load or is not available.")
    if not tts_pipeline: print("⚠️ TTS pipeline (Kokoro) failed to load or is not available.")

# This ensures models are loaded when Django starts and this module is imported.
# For more robust Django app loading, consider using AppConfig.ready().
ensure_models_initialized()


### 🎙️ **Offline Speech-to-Text (Transcription with Whisper)**
def transcribe_audio_offline(audio_file_path):
    if not stt_model:
        print("❌ STT Model (Whisper) not loaded. Cannot transcribe.")
        return None
    try:
        print(f"🎙️ Transcribing (Whisper offline): {audio_file_path}")
        # Forcing English but Whisper can auto-detect. fp16 useful on CUDA.
        result = stt_model.transcribe(audio_file_path, language="en", fp16=(DEVICE=="cuda"))
        transcription = result["text"]
        print(f"📝 Transcription: {transcription}")
        return transcription
    except Exception as e:
        print(f"❌ Error in offline STT (Whisper): {e}")
        print(traceback.format_exc())
        return None

### 🤖 **LLM Inference (Mistral + LoRA)**
def chat_with_model(user_input_text):
    global llm_messages # Use the module-level conversation history
    if not llm_model or not llm_tokenizer:
        print("❌ LLM (Mistral+LoRA) or Tokenizer not loaded. Cannot generate response.")
        return "I am currently unable to process your request due to a model loading issue."

    # Prepare conversation for the prompt by appending the new user message
    conversation_for_prompt = llm_messages + [{"role": "user", "content": user_input_text}]
    
    try:
        # Apply chat template (Mistral Instruct v0.3 uses a specific format)
        # The tokenizer for "mistralai/Mistral-7B-Instruct-v0.3" should handle this.
        inputs = llm_tokenizer.apply_chat_template(
            conversation_for_prompt,
            tokenize=True,
            add_generation_prompt=True, # Ensures the model knows to generate a response
            return_tensors="pt"
        ).to(DEVICE)

        # Generate response using the fine-tuned model
        with torch.no_grad(): # Disable gradient calculations for inference
            outputs = llm_model.generate(
                input_ids=inputs,
                max_new_tokens=150,  # Keep responses concise as per system prompt
                temperature=0.6,     # Slightly creative but mostly factual
                top_p=0.9,
                do_sample=True,      # Enable sampling
                pad_token_id=llm_tokenizer.eos_token_id # Crucial for proper generation
            )
        
        # Decode only the newly generated part of the output
        input_token_len = inputs.shape[1]
        generated_tokens = outputs[0][input_token_len:]
        ai_response_text = llm_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        print(f"🤖 LLM AI Response: {ai_response_text}")

        # Update conversation history
        llm_messages.append({"role": "user", "content": user_input_text})
        llm_messages.append({"role": "assistant", "content": ai_response_text})
        
        # Optional: Truncate history to prevent excessive context length
        MAX_HISTORY_PAIRS = 5 # Number of user/assistant turn pairs (e.g. 5 pairs = 10 messages + system prompt)
        if len(llm_messages) > (MAX_HISTORY_PAIRS * 2 + 1):
             # Keep system prompt + last MAX_HISTORY_PAIRS turns
            llm_messages = [llm_messages[0]] + llm_messages[-(MAX_HISTORY_PAIRS * 2):]

        return ai_response_text
    except Exception as e:
        print(f"❌ Error during LLM (Mistral+LoRA) chat generation: {e}")
        print(traceback.format_exc())
        return "An error occurred while formulating my response."

### 🔊 **Offline Text-to-Speech (Kokoro)**
def text_to_speech_offline(text_to_speak, output_filepath, lang_code=KOKORO_LANG_CODE, voice=KOKORO_VOICE, speed=KOKORO_TTS_SPEED):
    if not tts_pipeline:
        print("❌ TTS Pipeline (Kokoro) not loaded. Cannot synthesize speech.")
        return False

    print(f"🔊 Synthesizing TTS (Kokoro) for: '{text_to_speak[:60]}...'")
    try:
        # Kokoro's KPipeline returns a generator.
        # Each item from the generator is (graphemes, phonemes, audio_numpy_array)
        # We need to collect all audio segments and concatenate them.
        
        audio_segments = []
        # Process text in chunks if extremely long, though Kokoro might handle this.
        # For simplicity, let's assume Kokoro's internal handling or moderate text length.
        # If very long texts, your original chunking logic for KPipeline might be needed.
        generator = tts_pipeline(text_to_speak, voice=voice, speed=speed, split_pattern=r'\n+') # Use configured voice and speed
        
        for _graphemes, _phonemes, audio_segment_np_array in generator:
            audio_segments.append(audio_segment_np_array)

        if not audio_segments:
            print("⚠️ No audio segments generated by Kokoro TTS for the given text.")
            return False

        combined_audio_data = np.concatenate(audio_segments)
        
        # Save the combined audio. Kokoro's default sample rate is typically 24000 Hz. Verify this.
        sf.write(output_filepath, combined_audio_data, samplerate=24000)
        print(f"✅ TTS audio saved to {output_filepath}")
        return True
    except Exception as e:
        print(f"❌ Error in TTS (Kokoro) synthesis: {e}")
        print(traceback.format_exc())
        return False

### 🌍 **Django API View for Processing Voice Input (Offline)**
@csrf_exempt # Add CSRF protection if your frontend sends CSRF tokens and you have sessions
def process_voice_input_offline(request):
    # Ensure models are loaded (handles first request or reloads if server restarts)
    # Note: ensure_models_initialized() is called at module load, but this is a safeguard.
    if not models_initialized_flag: # Or re-call ensure_models_initialized()
        print("🔁 Models were not initialized, attempting again...")
        ensure_models_initialized() 

    # Check if essential models are actually available after initialization attempt
    if not stt_model or not llm_model or not tts_pipeline:
         return JsonResponse({
             "error": "One or more core AI models are not available. Please check server logs.",
             "stt_ready": bool(stt_model),
             "llm_ready": bool(llm_model),
             "tts_ready": bool(tts_pipeline)
         }, status=503) # 503 Service Unavailable

    try:
        if request.method != "POST":
            return JsonResponse({"error": "Only POST requests are allowed for this endpoint."}, status=405)

        if "audio" not in request.FILES:
            return JsonResponse({"error": "No audio file ('audio' field) found in the request."}, status=400)

        audio_file_from_request = request.FILES["audio"]
        
        # Create a temporary directory for audio files if it doesn't exist
        temp_audio_storage_dir = "./temp_audio_files" 
        os.makedirs(temp_audio_storage_dir, exist_ok=True)
        
        # Define paths for temporary audio files
        uploaded_audio_path = os.path.join(temp_audio_storage_dir, "user_uploaded_audio.wav")
        tts_response_audio_path = os.path.join(temp_audio_storage_dir, "ai_response_audio.wav")
        
        # Save uploaded audio file
        with open(uploaded_audio_path, "wb") as f_out:
            for chunk in audio_file_from_request.chunks():
                f_out.write(chunk)
        print(f"💾 User audio saved temporarily to: {uploaded_audio_path}")

        # 1. Speech-to-Text (Offline)
        user_transcription = transcribe_audio_offline(uploaded_audio_path)
        if user_transcription is None: # Check specifically for None for transcription failure
            return JsonResponse({"error": "Failed to transcribe user audio."}, status=500)

        # 2. LLM Inference (Offline with Mistral+LoRA)
        ai_text_response = chat_with_model(user_transcription)
        if not ai_text_response or "unable to process" in ai_text_response.lower() or "error occurred" in ai_text_response.lower():
            # Check for generic error messages from chat_with_model
            return JsonResponse({"error": "AI model could not generate a response.", "details": ai_text_response}, status=500)

        # 3. Text-to-Speech (Offline with Kokoro)
        tts_succeeded = text_to_speech_offline(ai_text_response, tts_response_audio_path)
        if not tts_succeeded:
            return JsonResponse({"error": "Failed to convert AI text response to speech."}, status=500)

        # 4. Read generated TTS audio and encode as base64 for JSON response
        with open(tts_response_audio_path, "rb") as f_tts_wav:
            tts_wav_data = f_tts_wav.read()
        encoded_tts_audio_base64 = base64.b64encode(tts_wav_data).decode("utf-8")

        # Clean up temporary audio files after successful processing
        try:
            if os.path.exists(uploaded_audio_path): os.remove(uploaded_audio_path)
            if os.path.exists(tts_response_audio_path): os.remove(tts_response_audio_path)
        except OSError as e_cleanup:
            print(f"⚠️ Warning: Could not delete temporary audio files: {e_cleanup}")

        # Return successful JSON response
        return JsonResponse({
            "user_transcription": user_transcription,
            "ai_text_response": ai_text_response,
            "audio_base64": encoded_tts_audio_base64 # This is the AI's speech
        })

    except Exception as e_main:
        detailed_error_trace = traceback.format_exc()
        print(f"❌ Critical Server Error in 'process_voice_input_offline': {detailed_error_trace}")
        return JsonResponse({"error": "An unexpected critical server error occurred.", "details": str(e_main)}, status=500)