# 🚀 Space Cargo Management System

This is a backend + frontend project for managing space cargo, designed as part of the Space Cargo Management System challenge.

## 📂 Project Structure

```
.
├── main.py             # Backend API in Flask
├── Dockerfile          # Docker setup
├── requirements.txt    # Python dependencies
├── templates/          # HTML Frontend
```

## 🔧 Backend (Flask API)

The backend is written in Python using Flask. It exposes the following API endpoints:

- `POST /add_cargo` — Add a cargo record
- `GET /get_cargo/<id>` — Retrieve cargo by ID
- `GET /all_cargo` — List all cargo
- `DELETE /delete_cargo/<id>` — Delete cargo by ID

## 🌐 Frontend

A simple HTML + JavaScript frontend is provided to interact with the API.

## 🚀 Run Locally

### 1. Using Python (Development)

```bash
pip install -r requirements.txt
python main.py
```

### 2. Using Docker (Production-like)

```bash
docker build -t space-cargo .
docker run -p 8000:8000 space-cargo
```

## 📬 API Testing

Use Postman or the frontend UI to test endpoints.
