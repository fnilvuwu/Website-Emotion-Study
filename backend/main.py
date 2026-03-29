import base64
import cv2
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from backend.database import engine, get_db, Base
from backend import models
from backend.ai_pipeline import EmotionDetector

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Website Emotion Study Platform API")

# Initialize AI Pipeline
emotion_detector = EmotionDetector(model_path=None) # Load path if available

# Models for request body
class FrameRequest(BaseModel):
    image: str  # base64 encoded image
    session_id: int
    user_id: int

# Setup CORS to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Arousal-Valence Emotion Engine API"}

@app.post("/users/")
def create_user(username: str, role: str = "student", db: Session = Depends(get_db)):
    db_user = models.User(username=username, role=role)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/sessions/")
def create_session(title: str, teacher_id: int, db: Session = Depends(get_db)):
    db_session = models.CourseSession(title=title, teacher_id=teacher_id)
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

@app.post("/emotions/")
def upload_emotion(session_id: int, user_id: int, arousal: float, valence: float, status: str, db: Session = Depends(get_db)):
    """ Endpoint to receive lightweight emotion coordinates from frontend tracker """
    db_log = models.EmotionLog(
        session_id=session_id, 
        user_id=user_id, 
        arousal=arousal, 
        valence=valence,
        status=status
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log

@app.post("/predict")
def predict_emotion(request: FrameRequest, db: Session = Depends(get_db)):
    """ Main endpoint called natively from JS Tracker per frame """
    encoded_data = request.image.split(',')[1] # Removing base64 prefix if exists
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Pipeline Processing
    detection_result = emotion_detector.detect_face(frame)
    if detection_result is None:
        # If no face is found, could log it or ignore
        return {"arousal": 0.0, "valence": 0.0, "status": "No Face Detected", "bbox": None}
    
    face_img, bbox = detection_result

    predictions = emotion_detector.predict_emotion(face_img)
    arousal = predictions.get("arousal")
    valence = predictions.get("valence")
    
    # Classify State based on simple quadrants
    status = "Focused"
    if arousal < 0.4 and valence > -0.2:
        status = "Bored"
    elif valence < -0.3:
        status = "Frustrated"

    # Store persistently to SQLite
    db_log = models.EmotionLog(
        session_id=request.session_id, 
        user_id=request.user_id, 
        arousal=arousal, 
        valence=valence,
        status=status
    )
    db.add(db_log)
    db.commit()

    return {"arousal": arousal, "valence": valence, "status": status, "bbox": bbox}

