# app.py
import os
import requests
import base64
import torch
import tempfile
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Union, Optional # Añadir Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    pipeline
)
from TTS.api import TTS # Importar TTS
from PIL import Image
from io import BytesIO

# --- Configuración desde variables de entorno ---
API_KEY = os.getenv("API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
VL_LLM_MODEL_NAME = os.getenv("VL_LLM_MODEL_NAME")
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME")
TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME") # <-- Añadir esta línea
MODEL_LOADING_STRATEGY = os.getenv("MODEL_LOADING_STRATEGY", "DYNAMIC")

# --- Variables globales para modelos ---
llm_model, llm_tokenizer = None, None
vl_llm_model, vl_llm_processor = None, None
asr_pipe = None
tts_model = None # <-- Añadir esta línea

app = FastAPI()

# --- Configuración de cuantización (4-bit) ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# --- Funciones para descargar modelos ---
def download_llm():
    print(f"Descargando modelo de lenguaje: {LLM_MODEL_NAME}...")
    # Usar AutoTokenizer para LLM
    AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME, device_map="auto", quantization_config=quantization_config
    )
    AutoTokenizer.from_pretrained(LLM_MODEL_NAME) # <-- Cambiado
    print("✔ Modelo de lenguaje descargado.")

def download_vl_llm():
    print(f"Descargando modelo de visión: {VL_LLM_MODEL_NAME}...")
    AutoModelForCausalLM.from_pretrained(
        VL_LLM_MODEL_NAME, device_map="auto", quantization_config=quantization_config
    )
    AutoProcessor.from_pretrained(VL_LLM_MODEL_NAME)
    print("✔ Modelo de visión descargado.")

def download_asr():
    print(f"Descargando modelo ASR: {ASR_MODEL_NAME}...")
    pipeline("automatic-speech-recognition", model=ASR_MODEL_NAME)
    print("✔ Modelo ASR descargado.")

def download_tts(): # <-- Añadir esta función si TTS está integrado
    if TTS_MODEL_NAME:
        print(f"Descargando modelo TTS: {TTS_MODEL_NAME}...")
        TTS(model_name=TTS_MODEL_NAME, progress_bar=False, gpu=False) # gpu=False para descarga
        print("✔ Modelo TTS descargado.")
    else:
        print("⚠ No se especificó TTS_MODEL_NAME, omitiendo descarga de TTS.")

# --- Endpoint para descargar modelos ---
@app.post("/download-models")
async def download_models(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or auth_header.split(" ")[1] != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key for download endpoint")

    try:
        download_llm()
        download_vl_llm()
        download_asr()
        download_tts() # <-- Añadir esta línea si TTS está integrado
        return {"message": "Todos los modelos descargados exitosamente en el volumen."}
    except Exception as e:
        print(f"❌ Error al descargar modelos: {e}")
        raise HTTPException(status_code=500, detail=f"Error al descargar modelos: {str(e)}")

# --- Carga en modo PARALELO (con manejo de errores) ---
if MODEL_LOADING_STRATEGY == "PARALLEL":
    print("Modo de carga: PARALELO. Cargando todos los modelos al inicio.")
    try:
        print(f"Cargando modelo de lenguaje: {LLM_MODEL_NAME}...")
        # Usar AutoTokenizer para LLM de texto puro
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

        # Cargar TTS en paralelo también (si se desea)
        print(f"Cargando modelo de síntesis de voz: {TTS_MODEL_NAME}...")
        # Cargar con cuantización si es posible
        try:
            # Intentar cargar el modelo interno con cuantización
            # Este enfoque es complejo y específico. Para VITS, la API estándar es más directa.
            # La biblioteca TTS internamente usa PyTorch, pero no expone directamente el modelo de transformers.
            # Por lo tanto, la cuantización directa es difícil a través de la API estándar.
            # Sin embargo, para VITS, que es ligero, puede ser manejable con DYNAMIC o en PARALLEL si hay VRAM.
            # La estrategia aquí es cargarlo normalmente, pero elegir un modelo VITS ligero.
            tts_model = TTS(model_name=TTS_MODEL_NAME, progress_bar=False, gpu=torch.cuda.is_available())
            print("✔ Modelo de síntesis de voz cargado con éxito.")
        except Exception as e_tts_load:
            print(f"⚠️  No se pudo cargar TTS con cuantización directa: {e_tts_load}")
            print("   Cargando con la API estándar. Puede usar más VRAM.")
            tts_model = TTS(model_name=TTS_MODEL_NAME, progress_bar=False, gpu=torch.cuda.is_available())
            print("✔ Modelo de síntesis de voz (sin cuantización directa) cargado con éxito.")

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
            # Asegúrate de que el modelo ya esté descargado antes de cargarlo
            llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME, device_map="auto", quantization_config=quantization_config
            )
            llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME) # <-- Cambiado
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

def get_tts_model():
    global tts_model
    if tts_model is None:
        print(f"Cargando modelo de síntesis de voz: {TTS_MODEL_NAME}...")
        try:
            # Intentar cargar con cuantización si es posible (mismo comentario que en PARALLEL)
            # Para VITS, la API estándar es la forma más directa, aunque no cuantice directamente.
            # La eficiencia de VRAM dependerá del modelo VITS elegido.
            tts_model = TTS(model_name=TTS_MODEL_NAME, progress_bar=False, gpu=torch.cuda.is_available())
            print("✔ Modelo de síntesis de voz cargado con éxito.")
        except Exception as e:
            print(f"❌ Error al cargar el modelo de síntesis de voz: {e}")
            raise HTTPException(status_code=500, detail=f"Error al cargar TTS: {str(e)}")
    return tts_model # <-- Añadir esta línea


# --- Modelos Pydantic para la API ---
class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, str]]]
    audio_content: Optional[str] = None # <-- Añadir campo para audio base64


class RequestPayload(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = 2048
    temperature: float = 0.7
    # Añadir un campo para solicitar audio (opcional)
    request_audio_response: bool = False # <-- Nuevo campo


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
    
    # Aplicar el chat template con fallback para Llama 3
    if llm_tokenizer.chat_template is None:
        # Usar el template oficial de Llama 3 si no está definido
        llm_tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    
    messages_for_llm = [{"role": "user", "content": processed_input.strip()}]
    prompt = llm_tokenizer.apply_chat_template(
        messages_for_llm, tokenize=False, add_generation_prompt=True
    )
    inputs = llm_tokenizer(prompt, return_tensors="pt").to("cuda")
    output = llm_model.generate(**inputs, max_new_tokens=200)
    decoded = llm_tokenizer.decode(output[0], skip_special_tokens=True)

    # Extraer solo la respuesta del asistente (para Llama 3)
    if "<|start_header_id|>assistant<|end_header_id|>" in decoded:
        final_response_text = decoded.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    else:
        final_response_text = decoded

    # 3.1. Generar audio (TTS) si se solicitó
    audio_base64 = None
    if payload.request_audio_response: # <-- Comprobar si se solicitó audio
        tts_model = get_tts_model() # <-- Cargar modelo TTS
        # Generar el audio y guardarlo temporalmente
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_out:
            tts_model.tts_to_file(text=final_response_text, file_path=tmp_audio_out.name)
            tmp_audio_out_path = tmp_audio_out.name

        try:
            # Leer el archivo de audio generado
            with open(tmp_audio_out_path, "rb") as f:
                audio_bytes = f.read()
            # Codificar el audio a base64
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        finally:
            # Eliminar el archivo temporal de audio generado
            os.unlink(tmp_audio_out_path)


    # 4. Construir respuesta en formato OpenAI extendido
    response_message = {
        "role": "assistant",
        "content": final_response_text, # Texto
    }
    if audio_base64:
        response_message["audio_content"] = audio_base64 # Agregar audio si se generó

    return {
        "id": "chatcmpl-local-123",
        "object": "chat.completion",
        "created": 1715000000,
        "model": "local-llm",
        "choices": [{
            "index": 0,
            "message": response_message, # Usar el diccionario construido
            "finish_reason": "stop"
        }]
    }
