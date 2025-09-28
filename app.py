# app.py
import os
import json
import requests
import base64
import torch
import tempfile
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Union
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    pipeline
)
from PIL import Image
from io import BytesIO

# --- Configuración desde variables de entorno ---
API_KEY = os.getenv("API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
VL_LLM_MODEL_NAME = os.getenv("VL_LLM_MODEL_NAME")
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME")
MODEL_LOADING_STRATEGY = os.getenv("MODEL_LOADING_STRATEGY", "DYNAMIC")

# --- Variables globales para modelos ---
llm_model, llm_tokenizer = None, None
vl_llm_model, vl_llm_processor = None, None
asr_pipe = None

app = FastAPI()

# --- Configuración de cuantización (4-bit) ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# --- Carga en modo PARALELO (con manejo de errores) ---
if MODEL_LOADING_STRATEGY == "PARALLEL":
    print("Modo de carga: PARALELO. Cargando todos los modelos al inicio.")
    try:
        print(f"Cargando modelo de lenguaje: {LLM_MODEL_NAME}...")
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME, device_map="auto", quantization_config=quantization_config
        )
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        print("✔ Modelo de lenguaje cargado con éxito.")

        print(f"Cargando modelo de visión: {VL_LLM_MODEL_NAME}...")
        vl_llm_model = AutoModelForCausalLM.from_pretrained(
            VL_LLM_MODEL_NAME, device_map="auto", quantization_config=quantization_config
        )
        vl_llm_processor = AutoProcessor.from_pretrained(VL_LLM_MODEL_NAME)
        print("✔ Modelo de visión cargado con éxito.")

        print(f"Cargando modelo ASR: {ASR_MODEL_NAME}...")
        asr_pipe = pipeline("automatic-speech-recognition", model=ASR_MODEL_NAME, device=0)
        print("✔ Modelo ASR cargado con éxito.")
    except Exception as e:
        print(f"❌ Error al cargar los modelos en modo PARALELO: {e}")
        raise RuntimeError("Fallo crítico en modo PARALELO. Usa modo DYNAMIC.")

else:
    print("Modo de carga: DINÁMICO. Los modelos se cargarán bajo demanda.")


# --- Funciones de carga dinámica con manejo de errores y logs ---
def get_llm_model():
    global llm_model, llm_tokenizer
    if llm_model is None:
        print(f"Cargando modelo de lenguaje: {LLM_MODEL_NAME}...")
        try:
            llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME, device_map="auto", quantization_config=quantization_config
            )
            llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
            print("✔ Modelo de lenguaje cargado con éxito.")
        except Exception as e:
            print(f"❌ Error al cargar el modelo de lenguaje: {e}")
            raise HTTPException(status_code=500, detail=f"Error al cargar LLM: {str(e)}")
    return llm_model, llm_tokenizer


def get_vl_llm_model():
    global vl_llm_model, vl_llm_processor
    if vl_llm_model is None:
        print(f"Cargando modelo de visión: {VL_LLM_MODEL_NAME}...")
        try:
            vl_llm_model = AutoModelForCausalLM.from_pretrained(
                VL_LLM_MODEL_NAME, device_map="auto", quantization_config=quantization_config
            )
            vl_llm_processor = AutoProcessor.from_pretrained(VL_LLM_MODEL_NAME)
            print("✔ Modelo de visión cargado con éxito.")
        except Exception as e:
            print(f"❌ Error al cargar el modelo de visión: {e}")
            raise HTTPException(status_code=500, detail=f"Error al cargar VL-LLM: {str(e)}")
    return vl_llm_model, vl_llm_processor


def get_asr_pipe():
    global asr_pipe
    if asr_pipe is None:
        print(f"Cargando modelo ASR: {ASR_MODEL_NAME}...")
        try:
            asr_pipe = pipeline("automatic-speech-recognition", model=ASR_MODEL_NAME, device=0)
            print("✔ Modelo ASR cargado con éxito.")
        except Exception as e:
            print(f"❌ Error al cargar el modelo ASR: {e}")
            raise HTTPException(status_code=500, detail=f"Error al cargar ASR: {str(e)}")
    return asr_pipe


# --- Modelos Pydantic para la API ---
class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, str]]]


class RequestPayload(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = 2048
    temperature: float = 0.7


# --- Endpoint principal ---
@app.post("/v1/chat/completions")
async def chat_completions(request: Request, payload: RequestPayload):
    # 1. Autenticación
    auth_header = request.headers.get("Authorization")
    if not auth_header or auth_header.split(" ")[1] != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 2. Procesar mensajes
    processed_input = ""
    is_multimodal = False

    for msg in payload.messages:
        if isinstance(msg.content, str):
            processed_input += f"\n{msg.content}"
        elif isinstance(msg.content, list):
            is_multimodal = True
            for part in msg.content:
                # Soporte para formato OpenAI: image_url
                if part.get("type") == "image_url":
                    vl_llm_model, vl_llm_processor = get_vl_llm_model()
                    image_url = part.get("image_url", {}).get("url", "")
                    if image_url:
                        try:
                            response = requests.get(image_url, timeout=10)
                            response.raise_for_status()
                            image = Image.open(BytesIO(response.content)).convert("RGB")
                            prompt = f"{processed_input}\nUSER: Describe esta imagen."
                            inputs = vl_llm_processor(text=prompt, images=image, return_tensors="pt").to("cuda")
                            output = vl_llm_model.generate(**inputs, max_new_tokens=200)
                            generated_text = vl_llm_processor.decode(output[0], skip_special_tokens=True)
                            processed_input = generated_text.split("ASSISTANT:")[-1].strip()
                        except Exception as e:
                            raise HTTPException(status_code=400, detail=f"Error al procesar imagen desde URL: {e}")

                # Soporte para image_base64
                elif "image_base64" in part:
                    vl_llm_model, vl_llm_processor = get_vl_llm_model()
                    try:
                        image_data = base64.b64decode(part["image_base64"])
                        image = Image.open(BytesIO(image_data)).convert("RGB")
                        prompt = f"{processed_input}\nUSER: Describe esta imagen."
                        inputs = vl_llm_processor(text=prompt, images=image, return_tensors="pt").to("cuda")
                        output = vl_llm_model.generate(**inputs, max_new_tokens=200)
                        generated_text = vl_llm_processor.decode(output[0], skip_special_tokens=True)
                        processed_input = generated_text.split("ASSISTANT:")[-1].strip()
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"Error al decodificar imagen base64: {e}")

                # Soporte para audio_base64
                elif "audio_base64" in part:
                    asr_pipe = get_asr_pipe()
                    try:
                        audio_data = base64.b64decode(part["audio_base64"])
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            tmp.write(audio_data)
                            tmp_path = tmp.name
                        try:
                            transcription = asr_pipe(tmp_path)["text"]
                            processed_input += f"\n[TRANSCRIPCIÓN]: {transcription}"
                        finally:
                            os.unlink(tmp_path)
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"Error al procesar audio: {e}")

    # 3. Generar respuesta final (siempre con el LLM principal)
    llm_model, llm_tokenizer = get_llm_model()
    messages_for_llm = [{"role": "user", "content": processed_input.strip()}]
    prompt = llm_tokenizer.apply_chat_template(
        messages_for_llm, tokenize=False, add_generation_prompt=True
    )
    inputs = llm_tokenizer(prompt, return_tensors="pt").to("cuda")
    output = llm_model.generate(**inputs, max_new_tokens=200)
    decoded = llm_tokenizer.decode(output[0], skip_special_tokens=True)

    # Extraer solo la respuesta del asistente (para Llama 3)
    if "<|start_header_id|>assistant<|end_header_id|>" in decoded:
        final_response = decoded.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    else:
        final_response = decoded

    # 4. Formato OpenAI
    return {
        "id": "chatcmpl-local-123",
        "object": "chat.completion",
        "created": 1715000000,
        "model": "local-llm",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": final_response},
            "finish_reason": "stop"
        }]
    }