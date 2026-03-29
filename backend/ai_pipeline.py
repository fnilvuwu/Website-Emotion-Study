import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class EmotionDetector:
    def __init__(self, model_path=None):
        # Setup MediaPipe for Face Detection using Tasks API
        base_options = python.BaseOptions(model_asset_path='backend/detector.tflite')
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)
        
        # Determine device (CPU/GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained Arousal-Valence AffectNet model (PyTorch/ONNX)
        # Using a dummy model structure for compilation until user supplies the exact model .pth
        self.model = None
        if model_path:
            # self.model = torch.load(model_path, map_location=self.device)
            # self.model.eval()
            pass
        
        # Define image transforms typically used for PyTorch vision models
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def detect_face(self, frame):
        """ Extract the largest face from frame """
        # Convert BGR to RGB for MediaPipe Tasks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.detections:
            return None
            
        # Get highest confidence face
        detection = max(detection_result.detections, key=lambda d: d.categories[0].score)
        bbox = detection.bounding_box
        
        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
        ih, iw, _ = frame.shape
        
        # Boundary checks
        x, y = max(0, x), max(0, y)
        w, h = min(iw - x, w), min(ih - y, h)
        
        face_img = frame[y:y+h, x:x+w]
        return face_img, (x, y, w, h)

    def predict_emotion(self, face_img):
        """ Run Arousal-Valence inference """
        if self.model is None:
            # Fallback to random/mock values if no model is loaded yet
            # 0.0 to 1.0 Arousal, -1.0 to 1.0 Valence
            return {"arousal": np.random.uniform(0.1, 0.9), "valence": np.random.uniform(-0.8, 0.8)}
            
        # Convert OpenCV image to PIL
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            # Assuming output returns [valence, arousal]
            valence, arousal = output[0].tolist()
        
        return {"arousal": arousal, "valence": valence}
