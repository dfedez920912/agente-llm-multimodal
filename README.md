# Servidor LLM Multimodal Local

## 🚀 Visión General

Este proyecto implementa un servidor LLM (Large Language Model) multimodal local, diseñado para procesar mensajes de texto, imágenes y audio desde conversaciones de WhatsApp a través de **EvolutionAPI**. El servidor emula la API de **OpenAI** para permitir una integración fluida con **n8n** sin necesidad de modificar los flujos de trabajo existentes. El objetivo es ofrecer una solución de IA privada y de bajo coste, ideal para el desarrollo y las pruebas antes de pasar a servicios de pago.

---

## ✨ Características Principales

-   **Compatibilidad con n8n**: Emula la API de OpenAI, lo que permite el uso de nodos nativos de n8n.
-   **Procesamiento Multimodal**: Maneja imágenes (Base64), audio (Base64) y texto.
-   **Eficiencia de Hardware**: Optimizado para funcionar en una **RTX 3060 (12GB)** mediante la cuantización de modelos.
-   **Arquitectura Escalable**: Configurado con `docker-compose` para una transición sin problemas a hardware más potente (ej. 4x RTX 4090) con solo cambiar un archivo de entorno.
-   **Autenticación Segura**: Utiliza una API Key para proteger el endpoint.

---

## 🛠️ Despliegue Paso a Paso sistemas Windows

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/dfedez920912/agente-llm-multimodal.git](https://github.com/dfedez920912/agente-llm-multimodal.git)
    cd agente-llm-multimodal
    ```
2.  **Configurar las variables de entorno:**
    Copia el archivo de plantilla `.env.base` a `.env` y edítalo con tus datos.
    ```bash
    cp .env.base .env
    ```
    Ahora, edita el nuevo archivo `.env` con tu clave de API y otros parámetros si lo deseas.
3.  **Iniciar el servicio con Docker Compose:**
    Asegúrate de tener Docker Desktop y los drivers de NVIDIA configurados. Luego, ejecuta:
    ```bash
    docker-compose up --build -d
    ```
    El servicio estará disponible en `http://localhost:8000`.

    Actualizar el Sistema:

## 🛠️ Despliegue Paso a Paso sistemas Linux
1.   **Actualizar el Sistema:**
     Antes de instalar nada, es fundamental que actualices la lista de paquetes y las versiones de tu sistema:
    ```bash
     sudo apt update
     sudo apt upgrade -y
    ```


2.  **Identificar el Driver Recomendado:**
    Usa el siguiente comando para que Ubuntu te muestre los drivers disponibles y te sugiera cuál es el mejor para tu GPU:
    ```bash
    ubuntu-drivers devices
    ```
    La salida te mostrará una lista de drivers, con una recomendación entre paréntesis, como (recommended).

3.  **Instalar el Driver Recomendado:**
    Ahora, usa el comando autoinstall para que Ubuntu instale automáticamente el driver recomendado y las dependencias necesarias:
    ```bash
    sudo ubuntu-drivers autoinstall
    ```
    Este proceso se encarga de todo, incluso de instalar el kit de herramientas CUDA si es necesario para el driver.

4.  **Reiniciar el Servidor:**
    Para que los cambios surtan efecto y el kernel cargue los nuevos controladores, debes reiniciar el servidor.

    ```bash
    sudo reboot
    ```
5. **Verificar la Instalación:**
    Una vez que el servidor se haya reiniciado, vuelve a iniciar sesión y ejecuta el comando nvidia-smi.

    ```bash
    nvidia-smi
    ```
    Si la instalación fue exitosa, verás una tabla con la información de tu GPU, el uso de VRAM y la versión del driver.  Si el comando no funciona, algo salió mal durante la instalación y tendrás que revisar los logs.

---

## 🤝 Contribuidores

¡Gracias a estos increíbles colaboradores! 

<a href="https://github.com/dfedez920912"><img src="https://github.com/tu-usuario-1.png" width="50" alt="Daniel Fernandez Sotolongo"></a>
---

## ❤️ Apoyar el Proyecto

Si este proyecto te ha sido útil, por favor considera apoyar su desarrollo. ¡Cada donación ayuda a cubrir los costos y a seguir mejorando! [attachment_0](attachment).

* **PayPal**: [daniel920912](https://paypal.me/dfedez?locale.x=es_XC&country.x=US)
* **EnZona (Cuba)**: Escanea el código QR para donar. 
* **Transfermóvil (Cuba)**: Escanea el código QR para donar. 
* **QvaPay**: [dfedez920912]"https://qvapay.com/payme/daniel920912"
