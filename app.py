# app.py
import os
import json
import requests
import base64
import torch
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Union
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from PIL import Image
from io import BytesIO

# Configuración de los modelos desde variables de entorno
API_KEY = os.getenv("API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
VL_LLM_MODEL_NAME = os.getenv("VL_LLM_MODEL_NAME")
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME")
MODEL_LOADING_STRATEGY = os.getenv("MODEL_LOADING_STRATEGY", "DYNAMIC") # Default a DYNAMIC

# Variables globales para los modelos
llm_model, llm_tokenizer = None, None
vl_llm_model, vl_llm_processor = None, None
asr_pipe = None

app = FastAPI()

# Configuración de cuantización
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# --- LÓGICA DE CARGA DE MODELOS ---
if MODEL_LOADING_STRATEGY == "PARALLEL":
    print("Modo de carga: PARALELO. Cargando todos los modelos al inicio.")
    try:
        print(f"Cargando modelo de lenguaje: {LLM_MODEL_NAME}")
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            device_map="auto",
            quantization_config=quantization_config
        )
        llm_tokenizer = AutoProcessor.from_pretrained(LLM_MODEL_NAME)

        print(f"Cargando modelo de visión: {VL_LLM_MODEL_NAME}")
        vl_llm_model = AutoModelForCausalLM.from_pretrained(
            VL_LLM_MODEL_NAME,
            device_map="auto",
            quantization_config=quantization_config
        )
        vl_llm_processor = AutoProcessor.from_pretrained(VL_LLM_MODEL_NAME)

        print(f"Cargando modelo ASR: {ASR_MODEL_NAME}")
        asr_pipe = pipeline("automatic-speech-recognition", model=ASR_MODEL_NAME, device=0)
        
    except Exception as e:
        print(f"Error al cargar los modelos en modo paralelo: {e}.")
        print("Esto podría deberse a una limitación de VRAM. Considera usar el modo 'DYNAMIC'.")
        # Abortar la app si falla
        raise RuntimeError("No se pudieron cargar todos los modelos en modo PARALELO.")

else: # Por defecto, MODEL_LOADING_STRATEGY == "DYNAMIC"
    print("Modo de carga: DINÁMICO. Los modelos se cargarán bajo demanda.")

# --- LÓGICA DE CARGA DINÁMICA DE MODELOS ---
def get_llm_model():
    global llm_model, llm_tokenizer
    if llm_model is None:
        print(f"Cargando modelo de lenguaje: {LLM_MODEL_NAME}")
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME, device_map="auto", quantization_config=quantization_config
        )
        llm_tokenizer = AutoProcessor.from_pretrained(LLM_MODEL_NAME)
    return llm_model, llm_tokenizer

def get_vl_llm_model():
    global vl_llm_model, vl_llm_processor
    if vl_llm_model is None:
        print(f"Cargando modelo de visión: {VL_LLM_MODEL_NAME}")
        vl_llm_model = AutoModelForCausalLM.from_pretrained(
            VL_LLM_MODEL_NAME, device_map="auto", quantization_config=quantization_config
        )
        vl_llm_processor = AutoProcessor.from_pretrained(VL_LLM_MODEL_NAME)
    return vl_llm_model, vl_llm_processor

def get_asr_pipe():
    global asr_pipe
    if asr_pipe is None:
        print(f"Cargando modelo ASR: {ASR_MODEL_NAME}")
        asr_pipe = pipeline("automatic-speech-recognition", model=ASR_MODEL_NAME, device=0)
    return asr_pipe

# --- ENDPOINTS Y LÓGICA DE SOLICITUD ---
class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, str]]]

class RequestPayload(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = 2048
    temperature: float = 0.7

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, payload: RequestPayload):
    # 1. Autenticación
    auth_header = request.headers.get("Authorization")
    if not auth_header or auth_header.split(" ")[1] != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 2. Procesamiento de los mensajes
    processed_input = ""
    is_multimodal = False
    
    for msg in payload.messages:
        if isinstance(msg.content, str):
            processed_input += f"\n{msg.role}: {msg.content}"
        elif isinstance(msg.content, list):
            is_multimodal = True
            for part in msg.content:
                if "image_base64" in part:
                    vl_llm_model, vl_llm_processor = get_vl_llm_model()
                    image_data = base64.b64decode(part["image_base64"])
                    image = Image.open(BytesIO(image_data))
                    prompt = f"{processed_input}\nUSER: [IMAGEN]"
                    inputs = vl_llm_processor(text=prompt, images=image, return_tensors="pt")
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    output = vl_llm_model.generate(**inputs, max_new_tokens=200)
                    generated_text = vl_llm_processor.decode(output[0])
                    processed_input = generated_text.split("ASSISTANT:")[-1].strip()

                elif "audio_base64" in part:
                    asr_pipe = get_asr_pipe()
                    audio_data = base64.b64decode(part["audio_base64"])
                    with open("temp_audio.wav", "wb") as f:
                        f.write(audio_data)
                    transcription = asr_pipe("temp_audio.wav")["text"]
                    os.remove("temp_audio.wav")
                    processed_input += f"\n[TRANSCRIPCIÓN DE AUDIO]: {transcription}"
    
    # 3. Generar la respuesta final
    final_response = "Estoy procesando la solicitud."
    if not is_multimodal:
        llm_model, llm_tokenizer = get_llm_model()
        prompt = f"USER: {processed_input}\nASSISTANT:"
        inputs = llm_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        output = llm_model.generate(**inputs, max_new_tokens=200)
        final_response = llm_tokenizer.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
    elif is_multimodal:
        final_response = processed_input
        
    # 4. Construir respuesta en formato OpenAI
    response = {
        "id": "chatcmpl-local-dev-123",
        "object": "chat.completion",
        "created": 1715000000,
        "model": "local-llm",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": final_response},
            "finish_reason": "stop"
        }]
    }
    return response

    print(f"Cargando modelo de visión: {VL_LLM_MODEL_NAME}")
    vl_llm_model = AutoModelForCausalLM.from_pretrained(
        VL_LLM_MODEL_NAME,
        device_map="auto",
        quantization_config=quantization_config
    )
    vl_llm_processor = AutoProcessor.from_pretrained(VL_LLM_MODEL_NAME)

    print(f"Cargando modelo ASR: {ASR_MODEL_NAME}")
    asr_pipe = pipeline("automatic-speech-recognition", model=ASR_MODEL_NAME, device=0)

except Exception as e:
    print(f"Error al cargar los modelos: {e}.")
    # Manejo de error o carga parcial

app = FastAPI()

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, str]]]

class RequestPayload(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = 2048
    temperature: float = 0.7

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, payload: RequestPayload):
    # 1. Autenticación
    auth_header = request.headers.get("Authorization")
    if not auth_header or auth_header.split(" ")[1] != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 2. Procesamiento de los mensajes
    processed_input = ""
    is_multimodal = False
    
    for msg in payload.messages:
        if isinstance(msg.content, str):
            processed_input += f"\n{msg.role}: {msg.content}"
        elif isinstance(msg.content, list):
            is_multimodal = True
            for part in msg.content:
                if part.get("type") == "image_url":
                    # Este bloque es para la compatibilidad con la API de OpenAI (no es Base64)
                    image_url = part.get("image_url", {}).get("url")
                    if image_url:
                        response = requests.get(image_url)
                        image = Image.open(BytesIO(response.content))
                        prompt = f"{processed_input}\nUSER: [IMAGEN]"
                        inputs = vl_llm_processor(text=prompt, images=image, return_tensors="pt")
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        output = vl_llm_model.generate(**inputs, max_new_tokens=200)
                        generated_text = vl_llm_processor.decode(output[0])
                        processed_input = generated_text.split("ASSISTANT:")[-1].strip()
                elif "image_base64" in part:
                    image_data = base64.b64decode(part["image_base64"])
                    image = Image.open(BytesIO(image_data))
                    prompt = f"{processed_input}\nUSER: [IMAGEN]"
                    inputs = vl_llm_processor(text=prompt, images=image, return_tensors="pt")
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    output = vl_llm_model.generate(**inputs, max_new_tokens=200)
                    generated_text = vl_llm_processor.decode(output[0])
                    processed_input = generated_text.split("ASSISTANT:")[-1].strip()
                elif "audio_base64" in part:
                    audio_data = base64.b64decode(part["audio_base64"])
                    # Guardar temporalmente el archivo de audio para que `soundfile` lo lea
                    with open("temp_audio.wav", "wb") as f:
                        f.write(audio_data)
                    # Transcribir el audio
                    transcription = asr_pipe("temp_audio.wav")["text"]
                    os.remove("temp_audio.wav") # Eliminar archivo temporal
                    processed_input += f"\n[TRANSCRIPCIÓN DE AUDIO]: {transcription}"
    
    # 3. Generar la respuesta final
    final_response = "Estoy procesando la solicitud."
    if not is_multimodal:
        # Aquí se usa el LLM de solo texto para inputs de texto puro
        # Por simplicidad, se concatena el input para esta demostración
        prompt = f"USER: {processed_input}\nASSISTANT:"
        inputs = llm_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        output = llm_model.generate(**inputs, max_new_tokens=200)
        final_response = llm_tokenizer.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
    elif is_multimodal:
        # En el caso multimodal, ya se generó la respuesta en el paso anterior
        final_response = processed_input
        
    # 4. Construir respuesta en formato OpenAI
    response = {
        "id": "chatcmpl-local-dev-123",
        "object": "chat.completion",
        "created": 1715000000,
        "model": "local-llm",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": final_response},
            "finish_reason": "stop"
        }]
    }
    return response

