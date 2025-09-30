# download_models.py
import os
from transformers import AutoModel, AutoProcessor, pipeline
from huggingface_hub import login

# Obtener el token del entorno
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    print("Iniciando sesión en Hugging Face...")
    login(token=hf_token)
    print("Sesión iniciada.")
else:
    print("Advertencia: No se proporcionó HF_TOKEN. Es posible que no se pueda descargar el modelo restringido.")

models_to_download = [
    ('meta-llama/Llama-3.1-8b-Instruct', 'AutoModel', 'AutoProcessor', {'trust_remote_code': True}),
    ('llava-hf/llava-1.5-7b-hf', 'AutoModel', 'AutoProcessor', {'trust_remote_code': True}),
    ('openai/whisper-tiny', 'pipeline', 'automatic-speech-recognition', {}),
]

for model_name, model_class, processor_class, kwargs in models_to_download:
    print(f"Descargando {model_name}...")
    try:
        if model_class == 'pipeline':
            # Caso especial para pipeline
            pipeline(task=processor_class, model=model_name, **kwargs.get('pipeline_kwargs', {}))
        else:
            # Caso general para AutoModel y AutoProcessor
            model_cls = getattr(__import__('transformers', fromlist=[model_class]), model_class)
            proc_cls = getattr(__import__('transformers', fromlist=[processor_class]), processor_class)
            model_cls.from_pretrained(model_name, **kwargs)
            proc_cls.from_pretrained(model_name, **kwargs)
        print(f"✓ {model_name} descargado.")
    except Exception as e:
        print(f"✗ Error descargando {model_name}: {e}")
        # Decidir si detener o continuar
        # raise # Descomentar para detener el build si falla una descarga

print("Descarga de modelos completada.")