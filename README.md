# Servidor LLM Multimodal Local

## 🚀 Visión General

Este proyecto implementa un servidor LLM (Large Language Model) multimodal local, diseñado para procesar mensajes de texto, imágenes y audio desde conversaciones de WhatsApp a través de **EvolutionAPI**. El servidor emula la API de **OpenAI** para permitir una integración fluida con **n8n** sin necesidad de modificar los flujos de trabajo existentes. El objetivo es ofrecer una solución de IA privada y de bajo coste, ideal para el desarrollo y las pruebas antes de pasar a servicios de pago.

---

## ✨ Características Principales

-   **Compatibilidad con n8n**: Emula la API de OpenAI, lo que permite el uso de nodos nativos de n8n.
-   **Procesamiento Multimodal**: Maneja imágenes (Base64), audio (Base64) y texto.
-   **Síntesis de Voz (TTS)**: *Opcionalmente*, puede responder con voz, generando audio en base64.
-   **Eficiencia de Hardware**: Optimizado para funcionar en una **RTX 3060 (12GB)** mediante la cuantización de modelos.
-   **Arquitectura Escalable**: Configurado con `docker-compose` para una transición sin problemas a hardware más potente (ej. 4x RTX 4090) con solo cambiar un archivo de entorno.
-   **Autenticación Segura**: Utiliza una API Key para proteger el endpoint.

---

## 🛠️ Despliegue Paso a Paso sistemas Windows

### **Variante A: Compilación Local (`docker-compose up --build`)**

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/dfedez920912/agente-llm-multimodal.git
    cd agente-llm-multimodal
    ```
2.  **Configurar las variables de entorno:**
    Copia el archivo de plantilla `.env.base` a `.env` y edítalo con tus datos.
    ```bash
    cp .env.base .env
    ```
    Ahora, edita el nuevo archivo `.env` con tu clave de API y otros parámetros si lo deseas.
3.  **Iniciar el servicio con Docker Compose (compilando localmente):**
    Asegúrate de tener Docker Desktop y los drivers de NVIDIA configurados. Luego, ejecuta:
    ```bash
    docker-compose up --build -d
    ```
    El servicio estará disponible en `http://localhost:8000`.

### **Variante B: Descarga de Imagen Precompilada desde GitHub (`docker-compose up`)**

1.  **Clonar el repositorio (o descargar los ficheros `docker-compose.yml` y `.env`):**
    ```bash
    git clone https://github.com/dfedez920912/agente-llm-multimodal.git
    cd agente-llm-multimodal
    ```
2.  **Configurar las variables de entorno:**
    Copia el archivo de plantilla `.env.base` a `.env` y edítalo con tus datos.
    ```bash
    cp .env.base .env
    ```
    Ahora, edita el nuevo archivo `.env` con tu clave de API y otros parámetros si lo deseas.
3.  **Modificar `docker-compose.yml` para usar la imagen precompilada:**
    Edita el archivo `docker-compose.yml` y **reemplaza** la línea `build: .` por `image: ghcr.io/dfedez920912/agente-llm-multimodal:main` (o la etiqueta deseada). Asegúrate de que la sección `deploy` esté reemplazada por `runtime: nvidia` (como se describe en las mejoras).
    ```yaml
    # docker-compose.yml (fragmento)
    services:
      llm-agent:
        image: ghcr.io/dfedez920912/agente-llm-multimodal:main # <-- Usar imagen desde GHCR
        # build: . # <-- Comentar o eliminar esta línea
        restart: always
        environment:
          - API_KEY=${API_KEY}
          - LLM_MODEL_NAME=${LLM_MODEL_NAME}
          - VL_LLM_MODEL_NAME=${VL_LLM_MODEL_NAME}
          - ASR_MODEL_NAME=${ASR_MODEL_NAME}
          - TTS_MODEL_NAME=${TTS_MODEL_NAME} # <-- Asegúrate de tener esta si TTS está integrado
          - MODEL_LOADING_STRATEGY=${MODEL_LOADING_STRATEGY}
          - NVIDIA_VISIBLE_DEVICES=all # <-- Asegúrate de tener esta línea
        ports:
          - "8000:8000"
        volumes:
          - model-cache:/root/.cache/huggingface
        runtime: nvidia # <-- Crucial para Windows/WSL2
    ```
4.  **Iniciar el servicio con Docker Compose (descargando imagen):**
    Asegúrate de tener Docker Desktop y los drivers de NVIDIA configurados. Luego, ejecuta:
    ```bash
    docker-compose up -d
    ```
    Docker Compose descargará la imagen desde GitHub Container Registry y la ejecutará. El servicio estará disponible en `http://localhost:8000`.

---

## 🛠️ Despliegue Paso a Paso sistemas Linux

### **Variante A: Compilación Local (`docker-compose up --build`)**

1.   **Actualizar el Sistema:**
     Antes de instalar nada, es fundamental que actualices la lista de paquetes y las versiones de tu sistema:
    ```bash
     sudo apt update
     sudo apt upgrade -y
    ```

2.  **Instalar Docker y Docker Compose:**
    Sigue las instrucciones oficiales de Docker para instalar `docker-ce`, `docker-ce-cli` y `docker-compose-plugin` en Ubuntu.

3.  **Instalar los drivers de NVIDIA:**
    *   **Identificar el Driver Recomendado:**
        Usa el siguiente comando para que Ubuntu te muestre los drivers disponibles y te sugiera cuál es el mejor para tu GPU:
        ```bash
        ubuntu-drivers devices
        ```
        La salida te mostrará una lista de drivers, con una recomendación entre paréntesis, como (recommended).
    *   **Instalar el Driver Recomendado:**
        Ahora, usa el comando autoinstall para que Ubuntu instale automáticamente el driver recomendado y las dependencias necesarias:
        ```bash
        sudo ubuntu-drivers autoinstall
        ```
        Este proceso se encarga de todo, incluso de instalar el kit de herramientas CUDA si es necesario para el driver.
    *   **Reiniciar el Servidor:**
        Para que los cambios surtan efecto y el kernel cargue los nuevos controladores, debes reiniciar el servidor.
        ```bash
        sudo reboot
        ```
    *   **Verificar la Instalación:**
        Una vez que el servidor se haya reiniciado, vuelve a iniciar sesión y ejecuta el comando nvidia-smi.
        ```bash
        nvidia-smi
        ```
        Si la instalación fue exitosa, verás una tabla con la información de tu GPU, el uso de VRAM y la versión del driver.  Si el comando no funciona, algo salió mal durante la instalación y tendrás que revisar los logs.

4.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/dfedez920912/agente-llm-multimodal.git
    cd agente-llm-multimodal
    ```
5.  **Configurar las variables de entorno:**
    Copia el archivo de plantilla `.env.base` a `.env` y edítalo con tus datos.
    ```bash
    cp .env.base .env
    ```
    Ahora, edita el nuevo archivo `.env` con tu clave de API y otros parámetros si lo deseas.
6.  **Iniciar el servicio con Docker Compose (compilando localmente):**
    ```bash
    docker-compose up --build -d
    ```
    El servicio estará disponible en `http://localhost:8000`.

### **Variante B: Descarga de Imagen Precompilada desde GitHub (`docker-compose up`)**

Sigue los pasos 1 a 3 de la Variante A (Actualizar sistema, Instalar Docker, Instalar drivers NVIDIA).

4.  **Clonar el repositorio (o descargar los ficheros `docker-compose.yml` y `.env`):**
    ```bash
    git clone https://github.com/dfedez920912/agente-llm-multimodal.git
    cd agente-llm-multimodal
    ```
5.  **Configurar las variables de entorno:**
    Copia el archivo de plantilla `.env.base` a `.env` y edítalo con tus datos.
    ```bash
    cp .env.base .env
    ```
    Ahora, edita el nuevo archivo `.env` con tu clave de API y otros parámetros si lo deseas.
6.  **Modificar `docker-compose.yml` para usar la imagen precompilada:**
    Edita el archivo `docker-compose.yml` y **reemplaza** la línea `build: .` por `image: ghcr.io/dfedez920912/agente-llm-multimodal:main` (o la etiqueta deseada). Asegúrate de que la sección `deploy` esté reemplazada por `runtime: nvidia`.
    ```yaml
    # docker-compose.yml (fragmento)
    services:
      llm-agent:
        image: ghcr.io/dfedez920912/agente-llm-multimodal:main # <-- Usar imagen desde GHCR
        # build: . # <-- Comentar o eliminar esta línea
        restart: always
        environment:
          - API_KEY=${API_KEY}
          - LLM_MODEL_NAME=${LLM_MODEL_NAME}
          - VL_LLM_MODEL_NAME=${VL_LLM_MODEL_NAME}
          - ASR_MODEL_NAME=${ASR_MODEL_NAME}
          - TTS_MODEL_NAME=${TTS_MODEL_NAME} # <-- Asegúrate de tener esta si TTS está integrado
          - MODEL_LOADING_STRATEGY=${MODEL_LOADING_STRATEGY}
          - NVIDIA_VISIBLE_DEVICES=all # <-- Asegúrate de tener esta línea
        ports:
          - "8000:8000"
        volumes:
          - model-cache:/root/.cache/huggingface
        runtime: nvidia # <-- Crucial para Linux con GPU
    ```
7.  **Iniciar el servicio con Docker Compose (descargando imagen):**
    ```bash
    docker-compose up -d
    ```
    Docker Compose descargará la imagen desde GitHub Container Registry y la ejecutará. El servicio estará disponible en `http://localhost:8000`.

---

## 🧩 Explicación de los Cambios Realizados

### **1. Mejora de `app.py`**

*   **Uso correcto de `AutoTokenizer`:** Se cambió `AutoProcessor.from_pretrained` por `AutoTokenizer.from_pretrained` para el modelo de lenguaje (LLM), ya que `AutoProcessor` no es el adecuado para el flujo de texto puro con Llama 3.
*   **Aplicación de `chat_template`:** Se añadió el uso de `llm_tokenizer.apply_chat_template()` para aplicar el formato oficial de chat de Llama 3, mejorando la calidad y coherencia de las respuestas de texto.
*   **Manejo seguro de archivos temporales:** Se reemplazó el uso de `temp_audio.wav` por `tempfile.NamedTemporaryFile` para manejar de forma segura los archivos de audio decodificados, evitando conflictos en concurrencia.
*   **Manejo de errores y logs:** Se añadieron bloques `try/except` y mensajes de log (`print`) claros (`✔` / `❌`) para la carga de modelos, facilitando la depuración.
*   **Integración de Síntesis de Voz (TTS):** Se añadieron importaciones, variables globales, funciones de carga dinámica y lógica en el endpoint `/v1/chat/completions` para generar audio a partir de la respuesta de texto si se solicita (mediante un campo opcional en la solicitud).
*   **Actualización de Pydantic:** Se añadió el campo `audio_content` opcional al modelo `Message` para devolver el audio generado en la respuesta.
*   **Soporte para `image_url` (OpenAI):** Se integró el manejo del formato estándar de OpenAI para imágenes en URL, junto con el formato personalizado `image_base64`.

### **2. Corrección de `docker-compose.yml`**

*   **Compatibilidad con Windows/WSL2 y Linux:** Se reemplazó la sección `deploy: ...` (diseñada para Docker Swarm) por `runtime: nvidia`. Esta es la forma correcta de asignar la GPU a un contenedor en Docker Compose standalone, lo que es esencial para que el contenedor funcione correctamente en Windows con WSL2 o en Linux con drivers NVIDIA instalados.
*   **Uso de `NVIDIA_VISIBLE_DEVICES`:** Se añadió `NVIDIA_VISIBLE_DEVICES=all` como variable de entorno para asegurar que el contenedor vea todas las GPUs NVIDIA disponibles.

### **3. Integración de TTS en `.env`**

*   **Nueva variable `TTS_MODEL_NAME`:** Se añadió `TTS_MODEL_NAME` al archivo `.env` para permitir configurar el modelo de síntesis de voz desde las variables de entorno.

### **4. Actualización de `requirements.txt`**

*   **Adición de `TTS`:** Se añadió la biblioteca `TTS==0.22.0` para permitir la funcionalidad de síntesis de voz.

### **5. Workflow de GitHub Actions (`docker-build.yml`)**

*   **Automatización del Build y Push:** Se configuró un workflow para que, con cada push a la rama `main`, se construya la imagen Docker y se suba al GitHub Container Registry (GHCR) con etiquetas útiles (`latest`, `sha-commit`, `timestamp`).

---

## 🤝 Contribuidores

¡Gracias a estos increíbles colaboradores! 

<a href="https://github.com/dfedez920912  "><img src="https://github.com/tu-usuario-1.png  " width="50" alt="Daniel Fernandez Sotolongo"></a>
---

## ❤️ Apoyar el Proyecto

Si este proyecto te ha sido útil, por favor considera apoyar su desarrollo. ¡Cada donación ayuda a cubrir los costos y a seguir mejorando!

* **PayPal**: [daniel920912](https://paypal.me/dfedez?locale.x=es_XC&country.x=US)
* **EnZona (Cuba)**:
    <br>
    <img src="assets/enzona_qr.png" alt="Código QR EnZona para donar" width="200" height="200"> <!-- Ajusta el nombre del archivo y el tamaño -->
    <br>
* **Transfermóvil (Cuba)**: 
    <br>
    <img src="assets/transfermovil_qr.jpg" alt="Código QR EnZona para donar" width="200" height="200"> <!-- Ajusta el nombre del archivo y el tamaño -->
    <br>
* **QvaPay**: [dfedez920912](https://qvapay.com/payme/daniel920912)