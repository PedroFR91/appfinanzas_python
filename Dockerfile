# Usa una imagen base ligera de Python
FROM python:3.11-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos del proyecto al contenedor
COPY . /app

# Asegúrate de que pip esté actualizado e instala las dependencias
RUN python -m ensurepip --upgrade && \
    python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Expone el puerto 8080 (requerido por App Runner)
EXPOSE 8080

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
