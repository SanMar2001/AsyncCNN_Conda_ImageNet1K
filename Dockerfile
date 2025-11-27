FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Instalar Python 3.10, pip y herramientas básicas
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Crear entorno virtual con Python 3.10
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Actualizar pip
RUN pip install --upgrade pip

# Instalar PyTorch con CUDA 12.9
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Instalar librerías adicionales útiles
RUN pip install numpy scikit-learn docker kaggle python-dotenv

# Definir directorio de trabajo
WORKDIR /app

# Copiar el código de tu proyecto al contenedor
COPY . /app

# Comando por defecto para ejecutar tu script principal
CMD ["python", "cnn.py"]
