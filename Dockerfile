# Dockerfile

# Usa una imagen base con drivers CUDA preinstalados
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Instala dependencias del sistema y Python
RUN apt-get update && apt-get install -y python3-pip git ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Copiar archivos y instalar dependencias de Python
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copiar la aplicación
COPY . .

# Exponer el puerto del servidor
EXPOSE 8000

# Comando para iniciar el servidor
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
