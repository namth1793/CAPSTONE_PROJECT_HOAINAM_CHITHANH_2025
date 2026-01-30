# Analyzing Learning Engagement System using AI-Based Real-Time Emotion and Behavior Detection and Interactive Feedback Collection in K–12 STEAM Classrooms

## 1. Introduction

In recent years, the integration of **Artificial Intelligence (AI)** into education has created new opportunities for understanding and improving student learning experiences. One of the major challenges in K–12 STEAM classrooms is the **objective measurement of student engagement**, which is traditionally dependent on subjective observation and manual assessment by teachers.

This **Capstone Project (2025)** proposes and develops an **AI-based learning engagement analysis system** that leverages **real-time emotion recognition, behavior detection, and interactive feedback mechanisms**. The system analyzes classroom video and audio data to infer students’ emotional states, learning behaviors, and overall engagement levels, thereby supporting data-driven instructional decisions and improving teaching effectiveness.

## 2. Research Objectives

The primary objectives of this project are:

1. Design and implement a **real-time facial emotion recognition system** for classroom monitoring and attendance.
2. Detect and classify **student behaviors and emotional states** associated with engagement and attention.
3. Compute **engagement scores** at both individual and classroom levels.
4. Develop an **interactive web-based dashboard** for engagement visualization and analysis.
5. Ensure **ethical, privacy-aware, and responsible AI usage** in educational environments.

## 3. System Architecture Overview

The proposed system follows a **modular and layered architecture** designed to ensure scalability, interpretability, and maintainability.

<img width="1071" height="432" alt="Image" src="https://github.com/user-attachments/assets/6827822c-da59-4301-820b-84c7cddd4d20" />

### 3.1 Input Layer
- Captures real-time classroom video streams from tracking cameras  
- Collects audio signals via microphones and textual inputs from users  
- Provides raw multimodal data for subsequent processing stages  

### 3.2 Perception Layer
- Performs face detection and multi-object tracking on video streams  
- Preprocesses visual, audio, and textual data into structured representations  
- Prepares modality-specific inputs for deep learning models  

### 3.3 Analysis Layer
- Integrates identity, emotion, and behavior outputs from multiple modalities  
- Aligns multimodal data temporally and semantically  
- Computes student engagement scores and attention states based on predefined indicators  

### 3.4 Feedback & Attendance Layer
- Infers attendance status based on identity persistence and temporal presence  
- Analyzes textual feedback to extract sentiment and key topics  
- Generates high-level analytical results for learning evaluation  

### 3.5 Data Management Layer
- Stores engagement scores, attendance records, and analytical metadata  
- Supports real-time data ingestion and historical data retrieval  
- Ensures data consistency and scalability for long-term analysis  

### 3.6 Web Application Layer
- Provides an interactive dashboard for teachers and administrators  
- Visualizes engagement statistics, attendance summaries, and temporal trends  
- Enables user interaction and system monitoring through a web interface  

## 4. Key Features

### 4.1 Real-Time Emotion Recognition
- Facial expression analysis using computer vision techniques  
- Emotion classification (e.g., focused, confused, bored, neutral)  
- Robust performance in real classroom environments  

### 4.2 Learning Behavior Detection
- Identification of observable behaviors related to attention and participation  
- Complements emotion recognition for more reliable engagement estimation  

### 4.3 Engagement Scoring Mechanism
- Fusion of emotional and behavioral indicators  
- Numerical engagement scores per student and per session  
- Supports longitudinal engagement analysis  

### 4.4 Automatic Attendance System
- Face-based identification for attendance recording  
- Reduces manual effort and improves attendance accuracy  

### 4.5 Interactive Dashboard
- Visual analytics and statistical summaries  
- Class-level and individual-level reports  
- Exportable results for further analysis  

## 5. Technologies and Tools

| Category | Technologies |
|--------|--------------|
| Programming Languages | Python, JavaScript |
| Computer Vision | OpenCV |
| Face Recognition | InsightFace |
| Emotion Recognition | DeepFace |
| Behavior Recognition | YOLOv8-Pose |
| Backend Framework | Flask / FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Database | SQLite |
| Real-Time Processing | Video Streaming, WebSocket |

## 6. Project Structure

## Overall Project Structure

The project is organized into three main components: **Backend**, **Frontend**, and
**AI Models**, forming a complete end-to-end AI-powered classroom engagement system.

```text
project/
├── BackEnd/                       # FastAPI-based database & API server
│   ├── database/                  # ORM, database utilities
│   ├── database_backups/          # SQLite database backups
│   ├── detected_faces_samples/    # Face detection sample outputs
│   ├── feedback_audio/            # Voice feedback audio files
│   ├── classroom_ai.db            # Main SQLite database
│   ├── main.py                    # Backend entry point
│   ├── requirements.txt           # Backend dependencies
│   └── README.md                  # Backend documentation
│
├── FrontEnd/
│   └── classroom-dashboard/       # Next.js classroom dashboard (App Router)
│       ├── public/                # Static assets
│       ├── src/app/               # Application pages and components
│       ├── package.json           # Frontend dependencies & scripts
│       ├── next.config.mjs        # Next.js configuration
│       └── README.md              # Frontend documentation
│
├── Models/
│   ├── face_recognition_model.pkl
│   ├── face_database.pkl
│   ├── main.py
|   ├── README.md
|   └── requirements.txt
│
└── README.md                      # Root project documentation
```

## 7. Installation and Setup
### 7.1 Clone the Repository
```bash
git clone https://github.com/namth1793/CapstoneProject_HoaiNam_ChiThanh_2025.git
cd CapstoneProject_HoaiNam_ChiThanh_2025
```

### 7.2 Install Dependencies

#### Frontend dependencies
```bash
cd webapp/FrontEnd/classroom-dashboard
npm install
```

### 7.3 Run the System
#### Start backend server
```bash
cd BackEnd
python main.py

cd Models
python main.py
```

#### Launch frontend dashboard
```bash
cd /FrontEnd/classroom-dashboard
npm run dev
```

## 8. Methodology
1. Capture or upload classroom video data
2. Detect and track student faces
3. Perform emotion recognition on detected faces
4. Analyze learning-related behaviors
5. Compute engagement metrics
6. Store results securely
7. Visualize insights through the web dashboard

## 9. Limitations and Future Work
### 9.1 Current Limitations

- Performance may degrade under poor lighting or occlusion
- Behavior recognition accuracy depends on dataset quality

### 9.2 Future Enhancements

- Multi-camera classroom support
- Improved multimodal engagement analysis
- Real-time alerts for low engagement
- Integration with Learning Management Systems (LMS)

## 10. Contributors
#### Hoài Nam
#### Chí Thành

## 11. License
#### This project is licensed under the MIT License.
#### © 2025 Capstone Project Team

