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

## 🛠️ Despliegue Paso a Paso

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/agente-llm-multimodal.git](https://github.com/tu-usuario/agente-llm-multimodal.git)
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

---

## 🤝 Contribuidores

¡Gracias a estos increíbles colaboradores! 

<a href="https://github.com/dfedez920912"><img src="https://github.com/tu-usuario-1.png" width="50" alt="Daniel Fernandez Sotolongo"></a>
---

## ❤️ Apoyar el Proyecto

Si este proyecto te ha sido útil, por favor considera apoyar su desarrollo. ¡Cada donación ayuda a cubrir los costos y a seguir mejorando! [attachment_0](attachment).

* **PayPal**: [daniel920912@gmail.com]
* **EnZona (Cuba)**: Escanea el código QR para donar. 
* **Transfermóvil (Cuba)**: Envía tu donación a mi número de teléfono. 
* **QvaPay**: [dfedez920912] 
