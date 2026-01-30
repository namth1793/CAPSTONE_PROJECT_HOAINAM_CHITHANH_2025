# Classroom Management Database Server

## Overview

This project is a **FastAPI-based database server** for a Classroom Management and AI-assisted Learning System. It provides RESTful APIs to manage **users, students, attendance, emotions, behaviors, focus/engagement metrics, and student feedback (text & voice)**. The system is designed to support AI-driven classroom analytics, including engagement estimation and speech-to-text (STT) processing for voice feedback.

The server uses **SQLite + SQLAlchemy ORM**, supports **role-based authentication**, and exposes **OpenAPI documentation** for easy integration with frontend dashboards or AI pipelines.

---

## Project Folder Structure

```text
BackEnd/
├── __pycache__/              # Python bytecode cache
├── database/                 # Database-related files and ORM utilities
├── database_backups/         # Backup files for the SQLite database
├── detected_faces_samples/   # Sample images captured by face detection modules
├── feedback_audio/            # Stored audio files for voice feedback (STT input)
├── classroom_ai.db            # Main SQLite database file
├── main.py                    # FastAPI application entry point (API & database server)
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies

```

## Key Features

* **User Management**

  * Admin & regular user roles
  * Secure password hashing (SHA-256)
  * Token-based authentication (mock bearer tokens)

* **Student & Class Management**

  * Persistent student roster (`class_students`)
  * Student metadata: class, gender, DOB, parents, contact info
  * Active/inactive enrollment tracking

* **Attendance & Classroom Analytics**

  * Daily attendance (check-in / check-out)
  * Emotion recognition results with confidence scores
  * Behavior and engagement scoring
  * Focus and concentration level tracking

* **Student Feedback System**

  * Text feedback API
  * Voice feedback API (base64 audio upload)
  * Automatic Speech-to-Text (STT)

    * Google Speech Recognition (free)
    * Whisper (local, optional)
    * Multi-level fallback strategy

* **Real-Time Communication**

  * WebSocket support for broadcasting updates

* **API Documentation**

  * Swagger UI: `/api/docs`
  * OpenAPI JSON: `/api/openapi.json`

---

## Technology Stack

* **Backend Framework**: FastAPI
* **Database**: SQLite
* **ORM**: SQLAlchemy
* **Data Validation**: Pydantic
* **Speech-to-Text**:

  * SpeechRecognition (Google Web API)
  * OpenAI Whisper (optional, local)
* **Audio Processing**:

  * pydub
  * ffmpeg

---

## Database Schema

### Core Tables

* **users**

  * Authentication & authorization
  * Admin vs regular user

* **student_data**

  * Attendance
  * Emotion & behavior recognition
  * Focus and engagement metrics

* **class_students**

  * Official class roster
  * Persistent student information

* **student_feedback**

  * Text & voice feedback
  * Audio metadata and STT results

All tables are auto-created at startup:

```python
Base.metadata.create_all(bind=engine)
```

---

## Installation

### 1. Create virtual environment (recommended)

```bash
conda create -n database python=3.10 -y
conda activate database
```

### 2. Install dependencies

```bash
pip install -r requirement.txt
```
---

## Running the Server

```bash
python main.py
```

Access:

* API Docs: [http://localhost:8000/api/docs](http://localhost:8000/api/docs)
* Redoc: [http://localhost:8000/api/redoc](http://localhost:8000/api/redoc)

---

## Authentication Flow

1. User logs in → receives `access_token`
2. Token is passed as query parameter:

```
?token=YOUR_TOKEN
```

3. Role-based protection:

   * Admin-only endpoints
   * Authenticated user endpoints

> **Note**: Token system is mock-based for research/demo. Replace with JWT + Redis for production.

---

## Main API Endpoints (Summary)

### Feedback

* `POST /api/feedback/text`
* `POST /api/feedback/voice`
* `GET /api/feedback`
* `GET /api/feedback/stats`
* `POST /api/feedback/process-stt/{id}`

### Students & Classes

* `GET /api/students/list`
* `POST /api/students`
* `PUT /api/students/{student_id}`

### Real-Time

* WebSocket manager for live updates

---

## Audio & Speech-to-Text Pipeline

Voice feedback processing pipeline:

1. Receive base64 audio
2. Save to `feedback_audio/`
3. Convert to WAV (16kHz, mono)
4. Run STT with fallback order:

   * Google Speech Recognition
   * Whisper (local)
   * Silent / error detection
5. Store transcription & confidence score

This design ensures **robust STT even on low-resource systems**.

---

## Intended Use Cases

* AI-based classroom engagement analysis
* Smart attendance systems
* Student-centered feedback collection
* Educational research & thesis projects
* Backend for AI-powered classroom dashboards

---

## Notes & Limitations

* SQLite is suitable for **development and research**, not large-scale deployment
* Token authentication is **not production-grade**
* Whisper models may be CPU-intensive on low-end machines

---

## License

This project is intended for **academic, research, and educational use**.

---

## Author

Developed as part of an **AI-powered Classroom Engagement & Feedback System**.
