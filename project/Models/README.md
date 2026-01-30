## 🚀 Key Features

- 👤 **Face Recognition**
  - InsightFace for face detection & embedding
  - SVM classifier for identity recognition

- 😊 **Emotion Recognition**
  - DeepFace-based emotion analysis
  - Robust fallback when DeepFace is unavailable

- 🎯 **Behavior Recognition**
  - YOLOv8-Pose based behavior detection
  - Supported behaviors:
    `writing`, `look_straight`, `raising_one_hand`, `look_around`

- 📊 **Engagement Scoring**
  - Engagement score: **0 – 100**
  - Computed from **emotion + behavior**
  - 5 concentration levels:
    - Very Low
    - Low
    - Medium
    - High
    - Very High

- 📝 **Automatic Attendance System**
  - CSV-based local storage
  - Duplicate prevention within a time window
  - Each record includes:
    - Name, Emotion, Behavior
    - Engagement score
    - Concentration level

- 🌐 **Flask API Server**
  - Live camera streaming
  - REST APIs for engagement & attendance
  - Designed to connect with a web frontend (e.g. Next.js)

- 🔗 **Backend Integration**
  - Async data queueing
  - Sends:
    - Attendance data
    - Emotion data
    - Behavior data
    - Engagement data

---

## 📁 Project Structure

```text
.
├── Models/
│   ├── face_recognition_model.pkl
│   ├── face_database.pkl
│   ├── main.py
|   ├── README.md
|   └── requirements.txt
```

🧠 Model Files Explanation (.pkl)

1️⃣ face_recognition_model.pkl

This file stores the trained SVM classifier used for face recognition.

Contains:
- Trained sklearn.svm.SVC model
- Learned decision boundaries between identities
- Probability estimates for each known person

Used for:
- Predicting the identity of detected faces
- Returning (name, confidence) during real-time recognition

Generated when:
- Running menu option 2 – Train face recognition model

Loaded when:
- Starting real-time recognition
- Starting the Flask API server

2️⃣ face_database.pkl

This file stores the face feature database used during training.

Contains:
- Face embeddings extracted by InsightFace

Mapping between:
- Person name
- Feature vectors
- Training labels
- Typical structure:


Used for:
- Re-training or extending the recognition system
- Debugging or analyzing training data
- Ensuring reproducibility of experiments

⚙️ Installation (Using Conda)

1️⃣ Create Conda Environment

```bash
conda create -n engagement-ai python=3.10 -y
conda activate engagement-ai
```

2️⃣ Install Dependencies

The system automatically checks and installs required libraries at runtime.

Simply run:

```bash
pip install -r requirements.txt
```

▶️ How to Run
```bash
python main.py
```

You will see an interactive menu:

1. 📁 Create folder structure
2. 🎯 Train face recognition model
3. 🎥 Real-time recognition (Full system)
4. 📊 View attendance history
5. 🔗 Test backend connection
6. 🔧 GPU troubleshooting
7. 🌐 Start Flask API Server
8. 🚪 Exit

🧠 Engagement Scoring Logic
Emotion Weights
| Emotion | Weight |
| ------- | ------ |
| happy   | 0.85   |
| neutral | 0.70   |
| sad     | 0.40   |
| angry   | 0.30   |

Behavior Weights
| Behavior         | Weight |
| ---------------- | ------ |
| writing          | 0.90   |
| look_straight    | 0.80   |
| raising_one_hand | 0.75   |


Final engagement score is normalized to 0 – 100.

🌐 Flask API Endpoints
| Endpoint          | Description                 |
| ----------------- | --------------------------- |
| `/`               | Live camera stream UI       |
| `/video_feed`     | MJPEG camera stream         |
| `/api/engagement` | Classroom engagement report |
| `/api/attendance` | Attendance data             |


Default server:

```bash
http://localhost:5000
```

🎮 Keyboard Shortcuts (Real-time Mode)
| Key | Action                 |
| --- | ---------------------- |
| `q` | Quit                   |
| `s` | Save screenshot + info |
| `v` | View attendance        |
| `e` | Show engagement report |
| `d` | Debug information      |


🎓 Intended Use
- Smart classroom systems
- AI-based attendance tracking
- Student engagement analysis
- Academic research & thesis projects

⚠️ Limitations
- Face recognition accuracy depends on training data quality
- Designed for single-camera classroom setups
- Not optimized for large-scale multi-classroom deployment

📜 License
- This project is intended for academic and research purposes.
- Commercial usage requires proper licensing for models and datasets.