# Gunakan base image Python
FROM python:3.9-slim

# Buat direktori untuk aplikasi
WORKDIR /app

# Salin file requirements.txt ke dalam container
COPY requirements.txt requirements.txt

# Install semua dependensi yang diperlukan
RUN pip3 install -r requirements.txt

# Salin semua file aplikasi ke dalam container
COPY . .

# Expose port untuk FastAPI
EXPOSE 8000

# Jalankan aplikasi FastAPI menggunakan Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
