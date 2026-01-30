#!/usr/bin/env python3
"""
FACE RECOGNITION SYSTEM - INSIGHTFACE + DEEPFACE + YOLO-POSE + ATTENDANCE + REAL-TIME BACKEND
GPU/CPU DUAL MODE - AUTO FALLBACK TO CPU
WITH FLASK API FOR WEB CONTROL
WITH ENGAGEMENT SCORE CALCULATION BASED ON EMOTION AND BEHAVIOR
"""

import os
# Fix numpy version issue đầu tiên
import random
import sys

# Fix cho numpy version mới
try:
    import numpy
    if hasattr(numpy, '_core'):
        numpy.core.multiarray = numpy._core.multiarray
except:
    pass

import json
import logging
import pickle
import subprocess
import threading
import time
import warnings
from collections import Counter, defaultdict, deque

from flask import Response  # 🔴 THÊM import này

warnings.filterwarnings('ignore', category=FutureWarning)
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
# ==================== FLASK API ====================
from flask import Flask, jsonify, request
from flask_cors import CORS

# ==================== MOVE SKLEARN IMPORTS LÊN SAU ====================
# Import sklearn SAU KHI đã fix numpy
try:
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import Normalizer
    from sklearn.svm import SVC
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False
    # Define placeholders
    class Normalizer:
        def __init__(self, norm='l2'):
            self.norm = norm
        def transform(self, X):
            return X / np.linalg.norm(X, axis=1, keepdims=True)
    
    class SVC:
        def __init__(self, **kwargs):
            print("⚠️ SVC is a placeholder - install scikit-learn")
        def fit(self, X, y):
            pass
        def predict(self, X):
            return ['Unknown'] * len(X)
        def predict_proba(self, X):
            return np.zeros((len(X), 1))
        @property
        def classes_(self):
            return np.array(['Unknown'])

# ==================== THÊM IMPORT CHO DEEPFACE ====================
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace đã được import thành công")
except ImportError as e:
    print(f"⚠️ DeepFace not available: {e}")
    DEEPFACE_AVAILABLE = False

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

# Biến toàn cục cho AI system
ai_running = False
ai_thread = None
system = None
last_detection_results = []
last_detection_time = None
ai_status_lock = threading.Lock()
detection_lock = threading.Lock()  # 🔴 THÊM: Lock cho detection results

# ==================== THÊM IMPORT CHO YAML ====================
try:
    import yaml
except ImportError:
    print("📥 Installing pyyaml for YOLO...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
    import yaml
    
# ==================== ENGAGEMENT CALCULATOR ====================
import numpy as np


class CameraManager:
    """Quản lý camera dùng chung cho AI và streaming"""
    
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.is_running = False
        self.last_read_time = 0
        self.read_errors = 0
        self.max_errors = 10
        
    def start(self):
        """Khởi động camera với retry mechanism"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            time.sleep(0.5)
        
        print(f"🔍 Đang kết nối camera index {self.camera_index}...")
        
        # Thử các camera index khác nhau
        camera_indices = [self.camera_index]
        if self.camera_index == 0:
            camera_indices = [0, 1, 2, 3, 4]
        elif self.camera_index == 1:
            camera_indices = [1, 0, 2, 3, 4]
        else:
            camera_indices = [self.camera_index, 0, 1, 2, 3]
        
        for idx in camera_indices:
            try:
                print(f"  Thử camera index {idx}...")
                
                # Dùng CAP_DSHOW cho Windows, CAP_V4L2 cho Linux
                if sys.platform == 'win32':
                    self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                
                if self.cap.isOpened():
                    # Test đọc frame
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        self.camera_index = idx
                        # Cấu hình camera
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        
                        # Kiểm tra thực tế
                        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                        
                        print(f"✅ Connected to camera index {idx}")
                        print(f"   Resolution: {actual_width}x{actual_height}")
                        print(f"   FPS: {actual_fps}")
                        
                        self.is_running = True
                        self.read_errors = 0
                        return True
                    else:
                        print(f"  ❌ Camera {idx}: Không đọc được test frame")
                        self.cap.release()
                        self.cap = None
                else:
                    print(f"  ❌ Camera {idx}: Không mở được")
                    
            except Exception as e:
                print(f"  ❌ Camera {idx} error: {str(e)}")
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
        
        print(f"❌ Không thể kết nối đến bất kỳ camera nào (đã thử {camera_indices})")
        return False
    
    def read_frame(self):
        """Đọc frame từ camera với error handling"""
        if self.cap is None or not self.cap.isOpened():
            print("⚠️ Camera chưa được khởi động, đang thử khởi động lại...")
            if self.start():
                time.sleep(0.5)
            else:
                return None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                self.read_errors += 1
                print(f"⚠️ Lỗi đọc frame #{self.read_errors}")
                
                if self.read_errors >= self.max_errors:
                    print("🔄 Quá nhiều lỗi, đang khởi động lại camera...")
                    self.stop()
                    time.sleep(1)
                    if self.start():
                        self.read_errors = 0
                        time.sleep(0.5)
                        # Thử đọc lại
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            with self.lock:
                                self.frame = frame.copy()
                            return frame
                
                return None
            
            # Reset error counter
            self.read_errors = 0
            
            # Lưu frame mới nhất
            with self.lock:
                self.frame = frame.copy()
                self.last_read_time = time.time()
            
            return frame
            
        except Exception as e:
            print(f"❌ Exception khi đọc frame: {str(e)}")
            self.read_errors += 1
            return None
    
    def get_latest_frame(self):
        """Lấy frame mới nhất cho streaming"""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        
        # Nếu không có frame, thử đọc ngay lập tức
        return self.read_frame()
    
    def stop(self):
        """Dừng camera"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False
        self.frame = None
        print("✅ Camera stopped")

# 🔴 THÊM: Global camera manager
camera_manager = CameraManager(camera_index=0)

class EngagementCalculator:
    """Lớp tính toán độ tập trung dựa trên cảm xúc và hành vi"""
    
    def __init__(self):
        # Trọng số cảm xúc (0-100)
        self.emotion_weights = {
            'neutral': 70,    # Bình thường
            'happy': 85,      # Vui vẻ - tích cực
            'surprised': 65,  # Ngạc nhiên
            'surprise': 65,   # Alias
            'sad': 40,        # Buồn
            'sadness': 40,    # Alias
            'angry': 30,      # Tức giận
            'anger': 30,      # Alias
            'fear': 35,       # Sợ hãi
            'disgust': 40,    # Khó chịu
            'fearful': 35,    # Alias
            'disgusted': 40   # Alias
        }
        
        # Trọng số hành vi (0-100)
        self.behavior_weights = {
            'writing': 90,              # Đang viết - rất tập trung
            'look_straight': 80,        # Nhìn thẳng - tập trung
            'raising_one_hand': 75,     # Giơ một tay - tham gia
            'raising_two_hands': 78,    # Giơ hai tay - tích cực
            'raising_hand': 75,         # Alias
            'normal': 60,               # Bình thường
            'look_around': 35,          # Nhìn quanh - phân tâm
            'distracted': 30,           # Mất tập trung
            'unknown': 50,              # Không xác định
            '': 50                      # Empty behavior
        }
        
        # Hệ số điều chỉnh confidence
        self.confidence_factors = {
            'high': 1.05,    # confidence > 0.8
            'medium': 1.0,   # confidence 0.5-0.8
            'low': 0.95      # confidence < 0.5
        }
        
        # History để smoothing
        self.engagement_history = {}
        self.history_length = 5
    
    def _normalize_emotion_behavior_scores(self, emotion_score, behavior_score):
        """Chuẩn hóa điểm cảm xúc và hành vi về khoảng 0-100"""
        # Giới hạn từ 0-100
        emotion_score = max(0, min(100, emotion_score))
        behavior_score = max(0, min(100, behavior_score))
        
        return emotion_score, behavior_score
    
    def get_confidence_factor(self, confidence):
        """Xác định hệ số dựa trên confidence"""
        if confidence >= 0.8:
            return self.confidence_factors['high']
        elif confidence >= 0.5:
            return self.confidence_factors['medium']
        else:
            return self.confidence_factors['low']
    
    def calculate_engagement(self, student_id, emotion, emotion_confidence, 
                            behavior, behavior_confidence=None, bbox=None):
        """
        Tính điểm tập trung (0-100)
        
        Args:
            student_id: ID học sinh
            emotion: Cảm xúc (string)
            emotion_confidence: Độ tin cậy cảm xúc (0-1)
            behavior: Hành vi (string)
            behavior_confidence: Độ tin cậy hành vi (0-1)
            bbox: Bounding box (optional, cho spatial analysis)
        """
        
        # 1. Chuẩn hóa đầu vào
        emotion = emotion.lower() if emotion else 'neutral'
        behavior = behavior.lower() if behavior else 'normal'
        
        # Mặc định confidence nếu không có
        if behavior_confidence is None:
            behavior_confidence = 0.7  # Giả định moderate confidence
        
        # 2. Lấy trọng số cơ bản
        emotion_weight = self.emotion_weights.get(emotion, 50)
        behavior_weight = self.behavior_weights.get(behavior, 50)
        
        # 3. Tính toán điểm cơ bản với confidence
        emotion_score = emotion_weight * emotion_confidence
        behavior_score = behavior_weight * behavior_confidence
        
        # 4. Chuẩn hóa về khoảng 0-100
        emotion_score, behavior_score = self._normalize_emotion_behavior_scores(
            emotion_score, behavior_score
        )
        
        # 5. Kết hợp điểm (40% cảm xúc, 60% hành vi)
        base_engagement = (emotion_score * 0.4 + behavior_score * 0.6)
        
        # 6. Áp dụng hệ số confidence
        emotion_conf_factor = self.get_confidence_factor(emotion_confidence)
        behavior_conf_factor = self.get_confidence_factor(behavior_confidence)
        confidence_factor = (emotion_conf_factor + behavior_conf_factor) / 2
        
        # 7. Hệ số đặc biệt
        special_factors = self._calculate_special_factors(behavior, bbox)
        
        # 8. Tính toán cuối cùng
        adjusted_engagement = base_engagement * confidence_factor * special_factors
        
        # 9. GIỚI HẠN NGHIÊM NGẶT trong khoảng 0-100
        final_engagement = max(0, min(100, adjusted_engagement))
        
        # 10. Làm mượt với history
        smoothed_engagement = self._apply_smoothing(student_id, final_engagement)
        
        # 11. Đảm bảo cuối cùng vẫn nằm trong 0-100
        smoothed_engagement = max(0, min(100, smoothed_engagement))
        
        # 12. Phân loại mức độ tập trung
        concentration_level = self._classify_concentration(smoothed_engagement)
        
        return {
            'engagement_score': round(smoothed_engagement, 2),
            'concentration_level': concentration_level,
            'base_components': {
                'emotion': {
                    'type': emotion,
                    'weight': emotion_weight,
                    'confidence': emotion_confidence,
                    'score': round(emotion_score, 2)
                },
                'behavior': {
                    'type': behavior,
                    'weight': behavior_weight,
                    'confidence': behavior_confidence,
                    'score': round(behavior_score, 2)
                }
            },
            'adjustments': {
                'confidence_factor': round(confidence_factor, 3),
                'special_factors': round(special_factors, 3),
                'base_engagement': round(base_engagement, 2),
                'final_engagement': round(final_engagement, 2)
            }
        }
    
    def _calculate_special_factors(self, behavior, bbox):
        """Tính hệ số đặc biệt dựa trên hành vi và vị trí"""
        factor = 1.0
        
        # Boost cho hành vi tích cực
        if 'writing' in behavior:
            factor *= 1.05
        elif 'raising' in behavior:
            factor *= 1.03
        elif 'look_straight' in behavior:
            factor *= 1.02
        
        # Penalty cho hành vi tiêu cực
        if 'look_around' in behavior or 'distracted' in behavior:
            factor *= 0.90
        
        # Nếu có bbox, thêm spatial analysis
        if bbox:
            try:
                x, y, w, h = bbox
                center_x = x + w/2
                center_y = y + h/2
                
                # Giả định frame width=640, height=480
                frame_center_x = 320
                frame_center_y = 240
                
                # Khoảng cách đến center
                distance = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
                
                # Hệ số dựa trên vị trí
                if distance < 100:
                    factor *= 1.02
                elif distance > 300:
                    factor *= 0.98
            except:
                pass
        
        return min(1.1, max(0.9, factor))  # Giới hạn hệ số đặc biệt
    
    def _apply_smoothing(self, student_id, current_score):
        """Áp dụng moving average để làm mượt kết quả"""
        if student_id not in self.engagement_history:
            self.engagement_history[student_id] = []
        
        history = self.engagement_history[student_id]
        history.append(current_score)
        
        # Giữ lịch sử tối đa
        if len(history) > self.history_length:
            history.pop(0)
        
        # Weighted moving average (mới hơn -> nặng hơn)
        if len(history) > 0:
            weights = np.linspace(0.5, 1.0, len(history))
            weights = weights / weights.sum()
            smoothed = np.average(history, weights=weights)
            return float(smoothed)
        
        return current_score
    
    def _classify_concentration(self, score):
        """Phân loại mức độ tập trung"""
        if score >= 80:
            return "very_high"
        elif score >= 70:
            return "high"
        elif score >= 60:
            return "medium"
        elif score >= 50:
            return "low"
        else:
            return "very_low"
    
    def get_engagement_report(self, student_data_list):
        """Tạo báo cáo tập trung cho tất cả học sinh"""
        report = {
            'total_students': len(student_data_list),
            'average_engagement': 0,
            'concentration_distribution': {
                'very_high': 0, 'high': 0, 'medium': 0, 
                'low': 0, 'very_low': 0
            },
            'students': []
        }
        
        total_score = 0
        
        for student in student_data_list:
            engagement_result = self.calculate_engagement(
                student_id=student.get('id'),
                emotion=student.get('emotion', 'neutral'),
                emotion_confidence=student.get('emotion_confidence', 0.5),
                behavior=student.get('behavior', 'normal'),
                behavior_confidence=student.get('behavior_confidence', 0.7),
                bbox=student.get('bbox')
            )
            
            # Thêm vào báo cáo
            report['students'].append({
                'id': student.get('id'),
                'name': student.get('name', 'Unknown'),
                'engagement': engagement_result['engagement_score'],
                'concentration_level': engagement_result['concentration_level'],
                'emotion': student.get('emotion'),
                'behavior': student.get('behavior')
            })
            
            # Cập nhật thống kê
            total_score += engagement_result['engagement_score']
            report['concentration_distribution'][engagement_result['concentration_level']] += 1
        
        if report['total_students'] > 0:
            report['average_engagement'] = round(total_score / report['total_students'], 2)
        
        return report
    
    def _get_engagement_color(self, score):
        """Lấy màu dựa trên engagement score"""
        if score >= 80:
            return (0, 255, 0)  # Xanh lá - rất tốt
        elif score >= 70:
            return (0, 200, 0)  # Xanh lá nhạt - tốt
        elif score >= 60:
            return (255, 255, 0)  # Vàng - trung bình
        elif score >= 50:
            return (255, 165, 0)  # Cam - thấp
        else:
            return (255, 0, 0)  # Đỏ - rất thấp

# ==================== BACKEND DATA SENDER - ENHANCED ====================
class EnhancedBackendDataSender:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.is_connected = False
        self.last_attendance_sent = {}
        self.last_behavior_sent = {}
        self.last_emotion_sent = {}
        self.session_id = f"session_{int(time.time())}"
        
        # 🔴 THÊM: Mapping tên sang ID cố định
        self.student_name_to_id = {
            # Học sinh từ dữ liệu mẫu
            "Dino": "SV001",
            "Thinh": "SV003",
            "Minh": "SV002",
            "Mini": "SV004",
            "Khoa": "SV005",
            "Nam": "SV006",
            "Thanh": "SV007",
        }
        
        # 🔴 THÊM: Reverse mapping (ID -> Tên)
        self.student_id_to_name = {
            "SV001": "Dino",
            "SV002": "Minh",
            "SV003": "Thinh",
            "SV004": "Mini",
            "SV005": "Khoa",
            "SV006": "Nam",
            "SV007": "Thanh",
        }
        
        # 🔴 THÊM: Lock cho pending requests
        self.request_lock = threading.Lock()
        self.pending_requests = {}
        
        # 🔴 THÊM: Danh sách từ khóa "unknown" để lọc
        self.unknown_keywords = ['unknown', 'unknow', 'không rõ', 'chưa biết', 'unknown student', 'unidentified']
        
        self.test_connection()
        self.setup_headers()
        
        # 🔴 THÊM: Batch endpoint
        self.batch_endpoint = f"{base_url}/api/ai/batch-process"
        print(f"✅ Backend sender initialized with FIXED ID mapping.")
        print(f"📦 Batch endpoint: {self.batch_endpoint}")
        print(f"📊 Student mapping: {len(self.student_name_to_id)} names mapped")

        self._test_connection_safe()
        
        # 🔴 THÊM: Queue cho async processing
        self.request_queue = []
        self.queue_lock = threading.Lock()
        self.max_queue_size = 50
        
        # 🔴 THÊM: Background thread xử lý queue
        self.processor_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processor_thread.start()
        
        print(f"✅ Backend sender initialized with async processing and fixed ID system.")
    
    def setup_headers(self):
        """Thiết lập headers cho requests"""
        self.headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'AI-Recognition-System/1.0'
        }

    def _test_connection_safe(self):
        """Kiểm tra kết nối đến backend an toàn"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=2)
            if response.status_code == 200:
                self.is_connected = True
                print("✅ Đã kết nối đến backend thành công!")
                return True
            else:
                print(f"⚠️ Backend trả về mã lỗi: {response.status_code}")
                self.is_connected = False
                return False
        except requests.exceptions.ConnectionError:
            print("⚠️ Không thể kết nối đến backend. Chạy ở chế độ offline.")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"⚠️ Lỗi kiểm tra kết nối: {str(e)}")
            self.is_connected = False
            return False
    
    def test_connection(self):
        """Alias cho _test_connection_safe"""
        return self._test_connection_safe()
    
    # ==================== CORE: HÀM CHUYỂN ĐỔI ID ====================
    def get_fixed_student_id(self, student_name, raw_student_id=None):
        """
        Chuyển đổi tên học sinh sang ID cố định
        
        Args:
            student_name: Tên học sinh (string)
            raw_student_id: ID thô từ AI (optional)
            
        Returns:
            ID cố định (string) hoặc None nếu là unknown
        """
        if not student_name or student_name.strip() == "":
            # Nếu không có tên, trả về None (không gửi)
            return None
        
        # Chuẩn hóa tên
        name_lower = student_name.lower().strip()
        
        # Kiểm tra xem có phải "unknown" không
        for keyword in self.unknown_keywords:
            if keyword in name_lower:
                return None  # Trả về None để không gửi
        
        # Tìm trong mapping (không phân biệt hoa thường)
        for mapped_name, mapped_id in self.student_name_to_id.items():
            if mapped_name.lower() == name_lower:
                return mapped_id
        
        # Tìm partial match
        for mapped_name, mapped_id in self.student_name_to_id.items():
            if mapped_name.lower() in name_lower or name_lower in mapped_name.lower():
                print(f"🔍 Partial match: '{student_name}' -> '{mapped_name}' ({mapped_id})")
                return mapped_id
        
        # Nếu không tìm thấy, tạo ID mới dựa trên hash của tên
        # Đảm bảo luôn trả về ID dạng SVxxx
        name_hash = abs(hash(student_name)) % 1000
        new_id = f"SV{name_hash + 100:03d}"  # SV100 đến SV999
        
        # Thêm vào mapping để dùng sau
        self.student_name_to_id[student_name] = new_id
        self.student_id_to_name[new_id] = student_name
        
        print(f"📝 Created new mapping: '{student_name}' -> {new_id}")
        return new_id
    
    def get_student_name_from_id(self, student_id):
        """Lấy tên học sinh từ ID"""
        return self.student_id_to_name.get(student_id, "Unknown Student")
    
    def add_student_mapping(self, student_name, student_id):
        """Thêm mapping mới"""
        if student_name and student_id:
            self.student_name_to_id[student_name] = student_id
            self.student_id_to_name[student_id] = student_name
            print(f"➕ Added mapping: '{student_name}' <-> {student_id}")
    
    def _process_queue(self):
        """Background thread xử lý queue"""
        while True:
            time.sleep(0.1)  # Kiểm tra queue mỗi 100ms
            
            with self.queue_lock:
                if not self.request_queue:
                    continue
                
                # Lấy batch để xử lý (tối đa 5 request cùng lúc)
                batch = self.request_queue[:10]
                self.request_queue = self.request_queue[10:]
            
            # Xử lý batch trong thread riêng
            if batch:
                thread = threading.Thread(target=self._process_batch, args=(batch,), daemon=True)
                thread.start()
    
    def _process_batch(self, batch):
        """Xử lý batch requests"""
        threads = []
        for request_data in batch:
            thread = threading.Thread(
                target=self._send_request_async,
                args=(request_data['endpoint'], request_data['data'], request_data['request_type']),
                daemon=True
            )
            thread.start()
            threads.append(thread)
    
    def _send_request_async(self, endpoint, data, request_type):
        """Gửi request async (không blocking)"""
        try:
            # 🔴 THÊM: Kiểm tra nếu tên học sinh là "Unknown"
            student_name = data.get('student_name', data.get('name', 'Unknown'))
            if self._is_unknown_name(student_name):
                return  # Không gửi nếu là unknown
            
            response = requests.post(
                endpoint,
                json=data,
                headers=self.headers,
                timeout=3
            )
            
            if response.status_code == 200:
                # Log ngắn gọn
                student_id = data.get('student_id', 'N/A')
                print(f"📤 {request_type[:3]}: {student_name[:10]} ({student_id})")
            else:
                # Không log error để tránh spam
                pass
                
        except Exception as e:
            # Bỏ qua lỗi network
            pass
    
    def _is_unknown_name(self, name):
        """Kiểm tra xem tên có phải là unknown không"""
        if not name:
            return True
        
        name_lower = str(name).lower().strip()
        
        # Kiểm tra các từ khóa unknown
        for keyword in self.unknown_keywords:
            if keyword in name_lower:
                return True
        
        # Kiểm tra nếu tên quá ngắn hoặc chỉ có ký tự đặc biệt
        if len(name_lower) < 2:
            return True
        
        return False
    
    def _get_concentration_level(self, engagement):
        """Map engagement score to concentration level"""
        if engagement >= 0.8:
            return "high"
        elif engagement >= 0.6:
            return "medium"
        else:
            return "low"
    
    # 🔴 THÊM: Simple batch sender với ID cố định
    def send_detection_batch(self, detections, fps, frame_count):
        """Gửi detection batch đơn giản (non-queue) với ID cố định - KHÔNG GỬI UNKNOWN"""
        if not self.is_connected or not detections:
            return False

        try:
            batch_data = {
                "type": "detection_update",
                "session_id": f"session_{int(time.time())}",
                "timestamp": datetime.now().isoformat(),
                "fps": fps,
                "frame_count": frame_count,
                "data": []
            }
    
            for face in detections:
                behavior_text = face.get("behavior")
                engagement = face.get("engagement")
                focus_score = float(engagement)
                student_name = face.get("name", "Unknown")
                
                # 🔴 SỬA: Kiểm tra nếu là unknown thì bỏ qua
                if self._is_unknown_name(student_name):
                    continue  # Bỏ qua không gửi
                
                # 🔴 SỬA: Dùng ID cố định
                raw_student_id = face.get("id", f"ID_{hash(str(face)) % 10000:04d}")
                fixed_student_id = self.get_fixed_student_id(student_name, raw_student_id)
                
                # 🔴 THÊM: Kiểm tra nếu fixed_student_id là None (unknown) thì bỏ qua
                if fixed_student_id is None:
                    continue
        
                now = datetime.now()
        
                item = {
                    # 🔴 DÙNG ID CỐ ĐỊNH
                    "student_id": fixed_student_id,
                    "student_name": student_name,
                
                    # 🔴 CÁC TRƯỜNG QUAN TRỌNG CHO FOCUS
                    "focus_score": focus_score,
                    "concentration_level": self._get_concentration_level(engagement),
                    "focus_duration": 45.0,
                
                    # Dữ liệu khác
                    "emotion": face.get("emotion"),
                    "emotion_confidence": face.get("emotion_confidence", 0.5),
                    "behavior_type": behavior_text,
                    "behavior_score": focus_score * 0.9,
                    "behavior_details": behavior_text,
                    "attendance_status": "present",
                    # 🔴 FIX: Datetime fields
                    "check_in_time": now.isoformat(),
                    "date": now.strftime("%Y-%m-%d"),
                    "class_name": "STEM 1",  # 🔴 THÊM class_name cố định
                    "session_id": batch_data["session_id"],
                    "recorded_by": "AI Recognition System"
                }
                batch_data["data"].append(item)
            
            # 🔴 THÊM: Kiểm tra nếu không có dữ liệu hợp lệ thì không gửi
            if not batch_data["data"]:
                return False
    
            # Gửi batch đến /api/ai/batch-process trong thread riêng
            print("send detection batch")
            thread = threading.Thread(
                target=self._send_direct_batch,
                args=(batch_data,),
                daemon=True
            )
            thread.start()
            return True
    
        except Exception as e:
            print(f"Error in send_detection_batch: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _send_direct_batch(self, batch_data):
        """Gửi batch trực tiếp đến /api/ai/batch-process"""
        try:
            # 🔴 THÊM: Kiểm tra nếu batch rỗng
            if not batch_data.get("data"):
                return
            
            response = requests.post(
                self.batch_endpoint,
                json=batch_data,
                headers=self.headers,
                timeout=2
            )
        
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    count = result.get("processed_count", 0)
                    # Log thông tin ID
                    if batch_data["data"]:
                        first_item = batch_data["data"][0]
                        name = first_item.get("student_name", "Unknown")
                        student_id = first_item.get("student_id", "N/A")
                        student_emotion = first_item.get("emotion")
                        student_behavior = first_item.get("behavior_type")
                        print(f"📦 Batch sent: {count} items | First: {name} ({student_id}) - {student_emotion} - {student_behavior}")
        except Exception as e:
            pass  # Bỏ qua lỗi network
    
    # 🔴 THÊM: Hàm debug để xem mapping
    def debug_mapping(self):
        """Hiển thị thông tin mapping hiện tại"""
        print("\n" + "="*80)
        print("🔍 STUDENT ID MAPPING DEBUG")
        print("="*80)
        print(f"Total mappings: {len(self.student_name_to_id)}")
        
        # Hiển thị 10 mapping đầu
        print("\nTop 10 mappings:")
        for i, (name, student_id) in enumerate(list(self.student_name_to_id.items())[:10]):
            print(f"{i+1:2d}. '{name}' -> {student_id}")
        
        if len(self.student_name_to_id) > 10:
            print(f"   ... and {len(self.student_name_to_id) - 10} more")
        
        print(f"\nReverse mappings: {len(self.student_id_to_name)}")
        print("="*80)

# ==================== GPU CONFIGURATION ====================
def setup_gpu():
    """Cấu hình và kiểm tra GPU chi tiết"""
    print("🔍 Kiểm tra hệ thống GPU...")
    
    # Kiểm tra PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
            
            print(f"✅ PyTorch GPU được hỗ trợ: {device_name}")
            print(f"🎯 Số GPU: {gpu_count}")
            print(f"💾 Bộ nhớ GPU: {gpu_memory:.1f} GB")
            
            # Thiết lập GPU mặc định
            torch.cuda.set_device(current_device)
            return True, 'cuda'
        else:
            print("❌ PyTorch không tìm thấy GPU")
    except Exception as e:
        print(f"❌ Lỗi kiểm tra PyTorch GPU: {e}")
    
    print("🔧 Sử dụng CPU mode - Hệ thống vẫn hoạt động bình thường")
    return False, 'cpu'

def install_dependencies():
    """Cài đặt dependencies với fix cho các lỗi"""
    print("🔧 Kiểm tra và cài đặt dependencies...")
    
    # Danh sách packages với versions ổn định
    packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "opencv-python>=4.8.0", 
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "pillow>=10.0.0",
        "numpy==1.24.3",  # Fixed version for compatibility
        "insightface>=0.7.3",
        "deepface>=0.0.79",  # Version cũ hơn, ổn định hơn
        "pandas>=2.0.0",
        "ultralytics==8.0.196",  # Version ổn định, tránh lỗi C3k2
        "requests>=2.31.0",
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
    ]
    
    # Kiểm tra GPU để quyết định onnxruntime version
    gpu_available, _ = setup_gpu()
    if gpu_available:
        packages.append("onnxruntime-gpu>=1.16.0")
        print("🎯 Sẽ cài đặt onnxruntime-gpu cho GPU")
    else:
        packages.append("onnxruntime>=1.16.0")
        print("🎯 Sẽ cài đặt onnxruntime thường cho CPU")
    
    # Thử cài đặt từng package
    for package in packages:
        try:
            # Extract package name (loại bỏ version specifier)
            pkg_name = package.split('>=')[0].split('==')[0]
            
            if pkg_name == "torch":
                import torch
                print(f"✅ torch {torch.__version__} đã được cài đặt")
            elif pkg_name == "torchvision":
                import torchvision
                print(f"✅ torchvision {torchvision.__version__} đã được cài đặt")
            elif pkg_name == "ultralytics":
                import ultralytics
                print(f"✅ ultralytics {ultralytics.__version__} đã được cài đặt")
            elif pkg_name == "onnxruntime-gpu":
                try:
                    import onnxruntime as ort
                    providers = ort.get_available_providers()
                    print(f"✅ onnxruntime {ort.__version__} đã được cài đặt")
                    print(f"   Providers: {providers}")
                    continue
                except ImportError:
                    pass
            elif pkg_name == "onnxruntime":
                try:
                    import onnxruntime
                    print(f"✅ onnxruntime {onnxruntime.__version__} đã được cài đặt")
                    continue
                except ImportError:
                    pass
            else:
                __import__(pkg_name.replace('-', '_'))
            print(f"✅ {pkg_name} đã được cài đặt")
        except ImportError:
            print(f"📥 Đang cài đặt {package}...")
            try:
                # Cài đặt với default pip
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Đã cài đặt {package}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Không thể cài đặt {package}: {e}")
                print("🔄 Thử cài đặt với --user option...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--user"])
                    print(f"✅ Đã cài đặt {package} với --user option")
                except subprocess.CalledProcessError as e2:
                    print(f"🚨 Không thể cài đặt {package}: {e2}")
                    print("⚠️ Tiếp tục với package khác...")

def check_system_capabilities():
    """Kiểm tra khả năng hệ thống chi tiết"""
    print("\n" + "="*50)
    print("🔍 KIỂM TRA HỆ THỐNG CHI TIẾT")
    print("="*50)
    
    # Kiểm tra Python
    print(f"🐍 Python Version: {sys.version}")
    
    # Kiểm tra OpenCV
    try:
        import cv2
        print(f"📷 OpenCV Version: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV chưa được cài đặt")
    
    # Kiểm tra PyTorch
    try:
        import torch
        print(f"🔥 PyTorch Version: {torch.__version__}")
        print(f"🔧 CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print("🎯 PyTorch CUDA: SẴN SÀNG")
            print(f"🔧 GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("🎯 PyTorch CUDA: KHÔNG SẴN SÀNG")
    except ImportError:
        print("❌ PyTorch chưa được cài đặt")
    
    # Kiểm tra Ultralytics (YOLO)
    try:
        import ultralytics
        print(f"🎯 Ultralytics Version: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics chưa được cài đặt")
    
    # Kiểm tra ONNX Runtime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"📊 ONNX Runtime Version: {ort.__version__}")
        print(f"🔧 Providers: {providers}")
    except ImportError:
        print("❌ ONNX Runtime chưa được cài đặt")
    
    print("="*50)

# ==================== BEHAVIOR DETECTION - OPTIMIZED VERSION ====================

class BehaviorDetector:
    """Behavior detector tối ưu với temporal smoothing tích hợp"""
    
    # COCO Keypoints indices
    KP = {
        'NOSE': 0,
        'LEFT_EYE': 1,
        'RIGHT_EYE': 2,
        'LEFT_EAR': 3,
        'RIGHT_EAR': 4,
        'LEFT_SHOULDER': 5,
        'RIGHT_SHOULDER': 6,
        'LEFT_ELBOW': 7,
        'RIGHT_ELBOW': 8,
        'LEFT_WRIST': 9,
        'RIGHT_WRIST': 10,
        'LEFT_HIP': 11,
        'RIGHT_HIP': 12
    }
    
    def __init__(self, device='cuda', history_length=15):
        """
        Args:
            device: 'cuda' or 'cpu'
            history_length: số frame lưu lại để temporal smoothing
        """
        self.device = device
        self.pose_model = None
        self.model_loaded = False
        
        # Temporal smoothing buffers với tracking
        self.history_length = history_length
        self.behavior_history = defaultdict(lambda: deque(maxlen=history_length))
        self.person_tracking = {}  # tracking_id -> (center_x, center_y, timestamp)
        self.last_seen = {}  # tracking_id -> last seen timestamp
        
        # Configuration thresholds (TẤT CẢ ĐỀU DÙNG TỈ LỆ SO VỚI THÂN NGƯỜI)
        self.thresholds = {
            'hand_raised_ratio': 0.4,          # cổ tay cao hơn vai 40% chiều cao thân trên
            'elbow_raised_ratio': 0.2,         # khuỷu tay cao hơn vai 20% chiều cao thân trên
            'writing_hand_below_shoulder': 0.1, # tay viết thấp hơn vai 10% chiều cao thân trên
            'head_down_ratio': 0.15,           # mũi thấp hơn mắt 15% chiều cao thân trên
            'look_straight_thresh': 0.25,      # ngưỡng nhìn thẳng (tỉ lệ so với khoảng cách mắt)
            'keypoint_confidence': 0.3,
            'temporal_min_frames': 8,          # số frame tối thiểu để smoothing
            'stabilization_confidence': 0.65,  # confidence để thay đổi behavior stabilized
            'tracking_distance': 100,          # khoảng cách tracking (pixels)
            'cleanup_timeout': 10.0           # thời gian xóa tracking cũ (giây)
        }
        
        self._initialize_pose_detector()
    
    def _initialize_pose_detector(self):
        """Khởi tạo YOLOv8 pose detector với GPU tối ưu"""
        try:
            import torch
            from ultralytics import YOLO
            
            print(f"🚀 Initializing YOLOv8 Pose Detector on {self.device.upper()}...")
            
            # Dùng model nano để nhanh nhất
            model_name = 'yolov8n-pose.pt'
            
            try:
                self.pose_model = YOLO(model_name)
                
                # Move to device với cấu hình tối ưu
                if self.device == 'cuda' and torch.cuda.is_available():
                    self.pose_model.to('cuda')
                    
                    # Warm-up với batch size nhỏ
                    print("🔥 Warming up GPU model...")
                    dummy_input = torch.randn(1, 3, 256, 256).to('cuda')
                    with torch.no_grad():
                        _ = self.pose_model(dummy_input)
                    torch.cuda.synchronize()
                    
                    print(f"✅ Loaded {model_name} on GPU")
                else:
                    print(f"✅ Loaded {model_name} on CPU")
                    self.device = 'cpu'
                
                self.model_loaded = True
                return True
                
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                return False
            
        except ImportError as e:
            print(f"❌ Ultralytics not installed: {e}")
            return False
        except Exception as e:
            print(f"❌ Error initializing pose detector: {e}")
            return False
    
    def _get_keypoint(self, keypoints, idx):
        """Lấy keypoint với confidence check - FIXED VERSION"""
        # Kiểm tra index hợp lệ
        if idx >= len(keypoints):
            return None
        
        # Kiểm tra confidence
        if keypoints[idx][2] > self.thresholds['keypoint_confidence']:
            # Return [x, y] - đây là mảng numpy
            return keypoints[idx][:2].copy()  # Dùng .copy() để tránh reference
        
        return None
    
    def _get_body_scale(self, keypoints):
        """Tính chiều cao thân trên (shoulder to hip) - FIXED VERSION"""
        kp = self.KP
        
        # Thử lấy shoulder và hip bên trái trước
        left_shoulder = self._get_keypoint(keypoints, kp['LEFT_SHOULDER'])
        left_hip = self._get_keypoint(keypoints, kp['LEFT_HIP'])
        
        # SỬA: Kiểm tra đúng cách
        if left_shoulder is not None and left_hip is not None:
            return abs(left_hip[1] - left_shoulder[1])
        
        # Thử bên phải nếu bên trái không có
        right_shoulder = self._get_keypoint(keypoints, kp['RIGHT_SHOULDER'])
        right_hip = self._get_keypoint(keypoints, kp['RIGHT_HIP'])
        
        if right_shoulder is not None and right_hip is not None:
            return abs(right_hip[1] - right_shoulder[1])
        
        # Tính xấp xỉ từ các keypoints có sẵn
        shoulders = []
        hips = []
        
        for side in ['LEFT', 'RIGHT']:
            shoulder = self._get_keypoint(keypoints, kp[f'{side}_SHOULDER'])
            hip = self._get_keypoint(keypoints, kp[f'{side}_HIP'])
            
            # SỬA: Kiểm tra đúng cách
            if shoulder is not None:
                shoulders.append(shoulder)
            if hip is not None:
                hips.append(hip)
        
        if shoulders and hips:
            avg_shoulder_y = sum(s[1] for s in shoulders) / len(shoulders)
            avg_hip_y = sum(h[1] for h in hips) / len(hips)
            return abs(avg_hip_y - avg_shoulder_y)
        
        return None
    
    def _arm_raised(self, wrist, elbow, shoulder_y, body_scale):
        """Kiểm tra tay có giơ lên không (theo chuẩn lớp học)"""
        if wrist is None or elbow is None or body_scale is None:
            return False
        
        # Điều kiện giơ tay:
        # 1. Cổ tay cao hơn vai ≥ 40% chiều cao thân trên
        # 2. Khuỷu tay cao hơn vai ≥ 20% chiều cao thân trên
        wrist_condition = wrist[1] < shoulder_y - (self.thresholds['hand_raised_ratio'] * body_scale)
        elbow_condition = elbow[1] < shoulder_y - (self.thresholds['elbow_raised_ratio'] * body_scale)
        
        return wrist_condition and elbow_condition
    
    def _detect_hand_raised(self, keypoints, body_scale):
        """Phát hiện giơ tay - phiên bản cải tiến cho lớp học"""
        kp = self.KP
        
        # Lấy keypoints
        lw = self._get_keypoint(keypoints, kp['LEFT_WRIST'])
        rw = self._get_keypoint(keypoints, kp['RIGHT_WRIST'])
        le = self._get_keypoint(keypoints, kp['LEFT_ELBOW'])
        re = self._get_keypoint(keypoints, kp['RIGHT_ELBOW'])
        ls = self._get_keypoint(keypoints, kp['LEFT_SHOULDER'])
        rs = self._get_keypoint(keypoints, kp['RIGHT_SHOULDER'])
        
        if body_scale is None:
            return None
        
        # Tính tọa độ vai trung bình
        shoulders = []
        if ls is not None:
            shoulders.append(ls[1])
        if rs is not None:
            shoulders.append(rs[1])
        
        if not shoulders:
            return None
        
        shoulder_y = sum(shoulders) / len(shoulders)
        
        # Kiểm tra từng tay
        left_raised = self._arm_raised(lw, le, shoulder_y, body_scale)
        right_raised = self._arm_raised(rw, re, shoulder_y, body_scale)
        
        # Chỉ phát hiện giơ 1 tay (giơ 2 tay ít xảy ra trong lớp học)
        if left_raised and not right_raised:
            return "raising_one_hand"
        elif right_raised and not left_raised:
            return "raising_one_hand"
        
        return None
    
    def _detect_writing(self, keypoints, body_scale):
        """Phát hiện viết bài với ngữ cảnh cúi đầu + tay thấp"""
        kp = self.KP
        
        if body_scale is None:
            return None
        
        # Kiểm tra điều kiện cúi đầu
        nose = self._get_keypoint(keypoints, kp['NOSE'])
        le = self._get_keypoint(keypoints, kp['LEFT_EYE'])
        re = self._get_keypoint(keypoints, kp['RIGHT_EYE'])
        
        head_down = False
        if nose is not None and le is not None and re is not None:
            eyes_y = (le[1] + re[1]) / 2
            # Mũi thấp hơn mắt 15% chiều cao thân trên
            if nose[1] > eyes_y + (self.thresholds['head_down_ratio'] * body_scale):
                head_down = True
        
        if not head_down:
            return None
        
        # Helper function để kiểm tra từng tay
        def check_writing_side(wrist, elbow, shoulder, hip):
            if wrist is None or elbow is None or shoulder is None:
                return False
            
            # 1. Tay viết phải thấp hơn vai (trên bàn)
            if wrist[1] < shoulder[1] - (self.thresholds['writing_hand_below_shoulder'] * body_scale):
                return False
            
            # 2. Góc khuỷu tay hợp lý
            def angle(A, B, C):
                BA = A - B
                BC = C - B
                cosang = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
                return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
            
            elbow_angle = angle(np.array(shoulder), np.array(elbow), np.array(wrist))
            if not (70 < elbow_angle < 140):  # góc hợp lý cho viết
                return False
            
            # 3. Tay không quá xa thân người
            if hip is not None:
                if abs(wrist[0] - hip[0]) > 0.8 * body_scale:
                    return False
            
            return True
        
        # Kiểm tra cả hai tay
        if check_writing_side(
            self._get_keypoint(keypoints, kp['LEFT_WRIST']),
            self._get_keypoint(keypoints, kp['LEFT_ELBOW']),
            self._get_keypoint(keypoints, kp['LEFT_SHOULDER']),
            self._get_keypoint(keypoints, kp['LEFT_HIP'])
        ):
            return "writing"
        
        if check_writing_side(
            self._get_keypoint(keypoints, kp['RIGHT_WRIST']),
            self._get_keypoint(keypoints, kp['RIGHT_ELBOW']),
            self._get_keypoint(keypoints, kp['RIGHT_SHOULDER']),
            self._get_keypoint(keypoints, kp['RIGHT_HIP'])
        ):
            return "writing"
        
        return None
    
    def _detect_look_direction(self, keypoints, body_scale):
        """Phát hiện hướng nhìn dùng tỉ lệ khoảng cách mắt"""
        kp = self.KP
        
        nose = self._get_keypoint(keypoints, kp['NOSE'])
        le = self._get_keypoint(keypoints, kp['LEFT_EYE'])
        re = self._get_keypoint(keypoints, kp['RIGHT_EYE'])
        
        if nose is None or le is None or re is None:
            return "unknown"
        
        # Tính tâm mặt
        face_center_x = (le[0] + re[0]) / 2
        
        # Tính khoảng cách giữa hai mắt
        eye_distance = abs(le[0] - re[0])
        if eye_distance < 1e-6:
            return "unknown"
        
        # Tính độ lệch mũi so với tâm mặt (chuẩn hóa theo khoảng cách mắt)
        yaw = (nose[0] - face_center_x) / eye_distance
        
        # Phân loại
        if abs(yaw) < self.thresholds['look_straight_thresh']:
            return "look_straight"
        else:
            return "look_around"
    
    def _determine_primary_behavior(self, hand_behavior, writing_behavior, look_behavior):
        """
        Priority cải tiến cho lớp học:
        1. Giơ tay
        2. Viết bài
        3. Nhìn quanh (chỉ khi KHÔNG viết)
        4. Nhìn thẳng
        """
        if hand_behavior == "raising_one_hand":
            return hand_behavior
        
        if writing_behavior == "writing":
            return "writing"
        
        if look_behavior == "look_around" and writing_behavior != "writing":
            return "look_around"
        
        if look_behavior == "look_straight":
            return look_behavior
        
        return "unknown"
    
    def _assign_tracking_id(self, center_x, center_y, current_time):
        """Gán hoặc tìm tracking ID dựa trên vị trí"""
        # Tìm tracking ID gần nhất
        closest_id = None
        min_distance = float('inf')
        
        for track_id, (last_x, last_y, last_time) in self.person_tracking.items():
            # Chỉ xét trong 5 giây gần nhất
            if current_time - last_time > 5.0:
                continue
            
            distance = np.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
            
            if distance < self.thresholds['tracking_distance'] and distance < min_distance:
                min_distance = distance
                closest_id = track_id
        
        # Nếu tìm thấy, cập nhật
        if closest_id is not None:
            self.person_tracking[closest_id] = (center_x, center_y, current_time)
            self.last_seen[closest_id] = current_time
            return closest_id
        
        # Tạo ID mới
        new_id = len(self.person_tracking)
        self.person_tracking[new_id] = (center_x, center_y, current_time)
        self.last_seen[new_id] = current_time
        return new_id
    
    def _cleanup_old_tracking(self, current_time):
        """Xóa tracking data cũ"""
        ids_to_remove = []
        
        for track_id, last_time in self.last_seen.items():
            if current_time - last_time > self.thresholds['cleanup_timeout']:
                ids_to_remove.append(track_id)
        
        for track_id in ids_to_remove:
            self.person_tracking.pop(track_id, None)
            self.last_seen.pop(track_id, None)
            self.behavior_history.pop(track_id, None)
        
        if ids_to_remove:
            print(f"🧹 Cleaned up {len(ids_to_remove)} old tracking IDs")
    
    def _apply_stabilization(self, tracking_id, current_behavior, current_time):
        """Áp dụng stabilization với tracking ID cố định"""
        # Thêm hành vi hiện tại vào lịch sử
        self.behavior_history[tracking_id].append((current_behavior, current_time))
        
        # Nếu chưa đủ frame, trả về hành vi hiện tại
        if len(self.behavior_history[tracking_id]) < self.thresholds['temporal_min_frames']:
            return current_behavior
        
        # Lấy danh sách behaviors gần đây (có timestamp trong 3 giây)
        recent_behaviors = []
        for behavior, timestamp in self.behavior_history[tracking_id]:
            if current_time - timestamp <= 3.0:
                recent_behaviors.append(behavior)
        
        if not recent_behaviors:
            return current_behavior
        
        # Tìm behavior phổ biến nhất
        behavior_counts = Counter(recent_behaviors)
        most_common_behavior, count = behavior_counts.most_common(1)[0]
        
        # Tính confidence
        confidence = count / len(recent_behaviors)
        
        # Ưu tiên các behaviors quan trọng
        important_behaviors = ['raising_one_hand', 'writing']
        
        # Nếu current là behavior quan trọng, ưu tiên giữ lại
        if current_behavior in important_behaviors:
            if most_common_behavior not in important_behaviors:
                return current_behavior
        
        # Chỉ thay đổi nếu confidence đủ cao
        if confidence >= self.thresholds['stabilization_confidence']:
            if most_common_behavior != current_behavior:
                print(f"🔄 Stabilized: {current_behavior} -> {most_common_behavior} "
                      f"(conf: {confidence:.2f}, track_id: {tracking_id})")
            return most_common_behavior
        
        return current_behavior
    
    def detect_behavior(self, image, use_stabilization=True):
        """Nhận diện hành vi với stabilization tích hợp"""
        if not self.model_loaded or self.pose_model is None:
            print("⚠️ Pose model not loaded")
            return []
        
        try:
            h, w = image.shape[:2]
            current_time = time.time()
            target_size = 320
            
            # Resize ảnh để tăng tốc xử lý
            if max(h, w) > 640:
                scale = target_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image_resized = cv2.resize(image, (new_w, new_h))
            else:
                image_resized = image
            
            # Run inference
            results = self.pose_model(
                image_resized,
                conf=0.3,
                iou=0.45,
                imgsz=target_size,
                device=self.device,
                half=False,
                verbose=False,
                max_det=10
            )
            
            behaviors = []
            
            for result_idx, result in enumerate(results):
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints_data = result.keypoints.data
                    
                    # Convert to numpy
                    if hasattr(keypoints_data, 'cpu'):
                        keypoints_np = keypoints_data.cpu().numpy()
                    else:
                        keypoints_np = keypoints_data
                    
                    # Xử lý shape của keypoints
                    # keypoints_np có shape (num_people, num_keypoints, 3)
                    if len(keypoints_np.shape) != 3:
                        continue
                    
                    # Scale keypoints về kích thước gốc
                    if image_resized is not image:
                        scale_h = h / image_resized.shape[0]
                        scale_w = w / image_resized.shape[1]
                        keypoints_np[:, :, 0] *= scale_w
                        keypoints_np[:, :, 1] *= scale_h
                    
                    for person_idx, keypoints in enumerate(keypoints_np):
                        # Đếm số keypoints visible
                        visible_kps = sum(1 for kp in keypoints 
                                        if kp[2] > self.thresholds['keypoint_confidence'])
                        
                        # Bỏ qua nếu quá ít keypoints
                        if visible_kps < 6:
                            continue
                        
                        # Tính body scale cho người này
                        body_scale = self._get_body_scale(keypoints)
                        
                        # Nếu không có body_scale, bỏ qua
                        if body_scale is None or body_scale < 10:
                            continue
                        
                        # Phát hiện các hành vi
                        hand_behavior = self._detect_hand_raised(keypoints, body_scale)
                        writing_behavior = self._detect_writing(keypoints, body_scale)
                        look_behavior = self._detect_look_direction(keypoints, body_scale)
                        
                        # Xác định hành vi chính
                        primary_behavior = self._determine_primary_behavior(
                            hand_behavior, writing_behavior, look_behavior
                        )
                        
                        # Lấy bounding box
                        bbox = None
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            boxes_data = result.boxes.data
                            if person_idx < len(boxes_data):
                                box = boxes_data[person_idx]
                                if hasattr(box, 'cpu'):
                                    bbox = box[:4].cpu().numpy()
                                else:
                                    bbox = box[:4]
                                
                                # Scale bbox về kích thước gốc
                                if image_resized is not image and bbox is not None:
                                    bbox[0] *= scale_w
                                    bbox[1] *= scale_h
                                    bbox[2] *= scale_w
                                    bbox[3] *= scale_h
                        
                        # Gán tracking ID nếu có bbox
                        tracking_id = None
                        if bbox is not None:
                            x1, y1, x2, y2 = bbox.astype(int)
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            tracking_id = self._assign_tracking_id(center_x, center_y, current_time)
                            
                            # Áp dụng stabilization nếu được yêu cầu
                            if use_stabilization and tracking_id is not None:
                                final_behavior = self._apply_stabilization(
                                    tracking_id, primary_behavior, current_time
                                )
                            else:
                                final_behavior = primary_behavior
                        else:
                            final_behavior = primary_behavior
                        
                        # Tính confidence
                        confidence = min(0.95, visible_kps / 13)
                        
                        behaviors.append({
                            'behavior': final_behavior,
                            'raw_behaviors': {
                                'hand': hand_behavior,
                                'writing': writing_behavior,
                                'look': look_behavior
                            },
                            'confidence': float(confidence),
                            'bbox': bbox,
                            'tracking_id': tracking_id,
                            'person_idx': person_idx,
                            'visible_keypoints': visible_kps,
                            'body_scale': float(body_scale) if body_scale else None,
                            'timestamp': current_time,
                            'history_size': len(self.behavior_history.get(tracking_id, []))
                        })
                else:
                    print(f"🎯 Result {result_idx}: No keypoints detected")
            
            # Cleanup old tracking data
            self._cleanup_old_tracking(current_time)
            
            return behaviors
            
        except Exception as e:
            print(f"❌ Error in behavior detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def clear_history(self, tracking_id=None):
        """Xóa lịch sử behavior và tracking"""
        if tracking_id:
            if tracking_id in self.behavior_history:
                self.behavior_history[tracking_id].clear()
            self.person_tracking.pop(track_id, None)
            self.last_seen.pop(track_id, None)
        else:
            self.behavior_history.clear()
            self.person_tracking.clear()
            self.last_seen.clear()
            print("🧹 Cleared all history and tracking data")
    
    def visualize(self, image, behaviors, show_raw=False, show_history=False, show_tracking=False):
        """Visualize behaviors on image"""
        visualized = image.copy()
        
        for behavior_data in behaviors:
            bbox = behavior_data.get('bbox')
            behavior = behavior_data.get('behavior')
            confidence = behavior_data.get('confidence', 0.5)
            tracking_id = behavior_data.get('tracking_id')
            
            if bbox is not None:
                x1, y1, x2, y2 = bbox.astype(int)
                
                # Màu sắc theo behavior
                color_map = {
                    'raising_one_hand': (0, 255, 0),      # Xanh lá
                    'writing': (255, 255, 0),            # Vàng
                    'look_around': (255, 165, 0),        # Cam
                    'look_straight': (0, 255, 255),      # Cyan
                    'unknown': (150, 150, 150)           # Xám
                }
                
                color = color_map.get(behavior, (200, 200, 200))
                
                # Vẽ bounding box
                cv2.rectangle(visualized, (x1, y1), (x2, y2), color, 2)
                
                # Chuẩn bị text
                text_parts = []
                
                if show_tracking and tracking_id is not None:
                    text_parts.append(f"ID:{tracking_id}")
                
                text_parts.append(f"{behavior}")
                text_parts.append(f"{confidence:.1f}")
                
                if show_raw:
                    raw = behavior_data.get('raw_behaviors', {})
                    text_parts.append(f"H:{raw.get('hand') or '-'}")
                    text_parts.append(f"W:{raw.get('writing') or '-'}")
                    text_parts.append(f"L:{raw.get('look') or '-'}")
                
                if show_history:
                    hist_size = behavior_data.get('history_size', 0)
                    text_parts.append(f"H:{hist_size}")
                
                text = " ".join(text_parts)
                
                # Background cho text
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Vẽ background rectangle cho text
                cv2.rectangle(visualized, 
                            (x1, y1 - text_height - 10),
                            (x1 + text_width + 10, y1),
                            (0, 0, 0), -1)
                
                # Vẽ text
                cv2.putText(visualized, text, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return visualized
        
# ==================== ATTENDANCE SYSTEM ====================
class AttendanceSystem:
    def __init__(self, csv_file="attendance.csv"):
        self.csv_file = csv_file
        self.backend_sender = EnhancedBackendDataSender()
        self.initialize_attendance_file()
    
    def initialize_attendance_file(self):
        """Khởi tạo file điểm danh"""
        try:
            if not os.path.exists(self.csv_file):
                df = pd.DataFrame(columns=[
                    'Name', 'Date', 'Time', 'Emotion', 'Behavior', 'Confidence', 'Engagement', 'Concentration_Level'
                ])
                df.to_csv(self.csv_file, index=False)
                print(f"✅ Đã tạo file điểm danh: {self.csv_file}")
            else:
                df = pd.read_csv(self.csv_file)
                print(f"✅ File điểm danh đã tồn tại: {len(df)} records")
        except Exception as e:
            print(f"❌ Lỗi khởi tạo file điểm danh: {str(e)}")
    
    def mark_attendance(self, name, emotion, emotion_confidence, behavior, engagement, concentration_level, confidence, bbox=None):
        """Queue dữ liệu điểm danh (async)"""
        try:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            
            student_id = f"SV{hash(name) % 10000:04d}"
            
            student_data = {
                'name': name,
                'student_id': student_id,
                'db_student_id': self.backend_sender.get_student_id_from_name(name),
                'emotion': emotion,
                'emotion_confidence': emotion_confidence,
                'behavior': behavior,
                'engagement': engagement,
                'concentration_level': concentration_level,
                'confidence': confidence,
                'timestamp': now.isoformat()
            }
            
            # 🔴 THAY ĐỔI: Queue tất cả dữ liệu async
            if self.backend_sender.is_connected:
                # Queue các loại dữ liệu
                self.backend_sender.queue_attendance_data(student_data)
                self.backend_sender.queue_behavior_data(student_data)
                self.backend_sender.queue_emotion_data(student_data)
                self.backend_sender.queue_engagement_data(student_data)
                
                # Log 1 lần duy nhất
                print(f"📥 Queued: {name} ({emotion[:3]}, {behavior}, engagement: {engagement})")
            
            # Lưu vào file local (sync nhưng nhanh)
            try:
                df = pd.read_csv(self.csv_file)
            except:
                df = pd.DataFrame(columns=[
                    'Name', 'Date', 'Time', 'Emotion', 'Behavior', 'Confidence', 'Engagement', 'Concentration_Level'
                ])
            
            # Kiểm tra duplicate trong 2 phút
            two_minutes_ago = (datetime.now() - pd.Timedelta(minutes=2)).strftime("%H:%M:%S")
            recent_entries = df[
                (df['Name'] == name) & 
                (df['Date'] == date_str) & 
                (df['Time'] > two_minutes_ago)
            ]
            
            if len(recent_entries) == 0:
                new_entry = {
                    'Name': name,
                    'Date': date_str,
                    'Time': time_str,
                    'Emotion': emotion,
                    'Behavior': behavior,
                    'Confidence': f"{confidence:.4f}",
                    'Engagement': f"{engagement:.4f}",
                    'Concentration_Level': concentration_level
                }
                
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                df.to_csv(self.csv_file, index=False)
                
                return True
            else:
                return False
                
        except Exception as e:
            # Không log để tránh spam
            return False
    
    def view_attendance(self):
        """Xem lịch sử điểm danh"""
        try:
            if not os.path.exists(self.csv_file):
                print("📭 Chưa có file điểm danh")
                return
                
            df = pd.read_csv(self.csv_file)
            if len(df) > 0:
                print("\n📊 LỊCH SỬ ĐIỂM DANH:")
                print("=" * 120)
                for _, row in df.iterrows():
                    print(f"👤 {row['Name']} | 📅 {row['Date']} | 🕒 {row['Time']} | 😊 {row['Emotion']} | 🎯 {row['Behavior']} | 📊 {row['Engagement']} | 🎯 {row['Concentration_Level']}")
                print("=" * 120)
                print(f"📈 Tổng số lượt điểm danh: {len(df)}")
            else:
                print("📭 Chưa có dữ liệu điểm danh")
        except Exception as e:
            print(f"❌ Lỗi đọc file điểm danh: {str(e)}")

# ==================== EMOTION DETECTION - FIXED VERSION ====================
class EmotionDetector:
    def __init__(self, min_face_size=64, confidence_threshold=0.3):
        """
        Args:
            min_face_size: Kích thước khuôn mặt tối thiểu (pixels)
            confidence_threshold: Ngưỡng confidence tối thiểu
        """
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Danh sách cảm xúc hỗ trợ bởi DeepFace
        self.supported_emotions = [
            'angry', 'disgust', 'fear', 'happy',
            'sad', 'surprise', 'neutral'
        ]
        
        # Kiểm tra DeepFace availability
        self.deepface_available = DEEPFACE_AVAILABLE
        
        if not self.deepface_available:
            print("⚠️ DeepFace không khả dụng. Emotion detection sẽ bị giới hạn.")
    
    def detect_emotion(self, face_image, return_all=False):
        """
        Nhận diện cảm xúc từ khuôn mặt
        
        Args:
            face_image: Ảnh khuôn mặt (BGR format)
            return_all: Trả về tất cả emotions hay chỉ dominant
        
        Returns:
            tuple: (dominant_emotion, confidence) hoặc dict của tất cả emotions
        """
        if not self.deepface_available:
            # Fallback đơn giản: luôn trả về neutral
            if return_all:
                return {"neutral": 0.5}
            else:
                return "neutral", 0.5
        
        # Kiểm tra ảnh đầu vào
        if face_image is None or face_image.size == 0:
            if return_all:
                return {"neutral": 0.3}
            else:
                return "neutral", 0.3
        
        try:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Sử dụng DeepFace với cấu hình đơn giản
            try:
                analysis = DeepFace.analyze(
                    img_path=face_rgb,
                    actions=['emotion'],
                    enforce_detection=False,  # Không bắt buộc phải detect được face
                    detector_backend='opencv',
                    silent=True
                )
            except Exception as deepface_error:
                print(f"⚠️ DeepFace analysis error: {deepface_error}")
                # Fallback
                if return_all:
                    return {"neutral": 0.4}
                else:
                    return "neutral", 0.4
            
            # Xử lý kết quả
            if isinstance(analysis, list) and len(analysis) > 0:
                analysis = analysis[0]
            
            if 'emotion' in analysis and 'dominant_emotion' in analysis:
                emotion_data = analysis['emotion']
                dominant_emotion = analysis['dominant_emotion']
                confidence = emotion_data.get(dominant_emotion, 50) / 100.0
                
                if return_all:
                    # Chuẩn hóa tất cả emotions về range 0-1
                    all_emotions = {}
                    for emotion, score in emotion_data.items():
                        all_emotions[emotion] = score / 100.0
                    return all_emotions
                else:
                    return dominant_emotion, confidence
            else:
                # No emotion data
                if return_all:
                    return {"neutral": 0.3}
                else:
                    return "neutral", 0.3
                
        except Exception as e:
            print(f"❌ Emotion detection error: {str(e)}")
            # Trả về neutral nhưng với confidence thấp để biết có lỗi
            if return_all:
                return {"neutral": 0.3}
            else:
                return "neutral", 0.3

# ==================== FACE RECOGNITION SYSTEM ====================
class CompleteRecognitionSystem:
    def __init__(self, model_name='buffalo_l', device='auto'):
        
        if device == 'auto':
            self.device = self._auto_detect_device()
        else:
            self.device = device
        
        print(f"🎯 System initialized on: {self.device.upper()}")
        self.model_name = model_name
        self.face_analyzer = None
        self.l2_normalizer = Normalizer('l2')
        
        # ==================== THÊM: Khởi tạo Emotion Detector ====================
        self.emotion_detector = EmotionDetector()
        print(f"😊 Emotion detector initialized")
                
        # Sử dụng StabilizedBehaviorDetectorGPU
        self.behavior_detector = BehaviorDetector(device)
        self.attendance_system = AttendanceSystem()
        self.backend_sender = self.attendance_system.backend_sender  # Use enhanced sender
        self.engagement_calculator = EngagementCalculator()  # 🔴 THÊM: Engagement calculator
        
        # 🔴 THÊM: Khởi tạo tracking
        self.face_tracking_ids = {}
        self.last_tracking_cleanup = time.time()
        # Model
        self.svm_model = None

    def _auto_detect_device(self):
        """Tự động phát hiện và chọn device tốt nhất"""
        print("🔍 Auto-detecting best device...")
        
        # Ưu tiên 1: CUDA GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                print(f"✅ Found GPU: {gpu_name}")
                print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
                
                # Kiểm tra memory đủ không (ít nhất 2GB)
                if gpu_memory >= 2.0:
                    print("🎯 Using CUDA (GPU)")
                    return 'cuda'
                else:
                    print("⚠️ GPU memory too low (< 2GB), using CPU")
                    return 'cpu'
        except:
            pass
        
        # Ưu tiên 2: ONNX Runtime GPU
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in providers or 'TensorrtExecutionProvider' in providers:
                print("✅ Found ONNX Runtime GPU provider")
                print("🎯 Using CUDA (ONNX)")
                return 'cuda'
        except:
            pass
        
        # Mặc định: CPU
        print("🔧 Using CPU (no suitable GPU found)")
        return 'cpu'
    
    def _cleanup_old_tracking(self):
        """Xóa tracking IDs cũ"""
        current_time = time.time()
        if current_time - self.last_tracking_cleanup > 30:  # Mỗi 30 giây cleanup 1 lần
            ids_to_remove = []
            for face_id, data in self.face_tracking_ids.items():
                if current_time - data.get('last_seen', 0) > 60:  # Xóa sau 60 giây không thấy
                    ids_to_remove.append(face_id)
            
            for face_id in ids_to_remove:
                del self.face_tracking_ids[face_id]
            
            self.last_tracking_cleanup = current_time
        
    def initialize_system(self):
        """Khởi tạo toàn bộ hệ thống với device phù hợp"""
        print(f"🚀 Initializing system on {self.device.upper()}...")
        
        # 1. Khởi tạo InsightFace với device
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            print("📥 Loading InsightFace model...")
            self.face_analyzer = FaceAnalysis(name=self.model_name)
            
            # Cấu hình device cho InsightFace
            # -1 = CPU, 0 = GPU 0, 1 = GPU 1, ...
            ctx_id = -1 if self.device == 'cpu' else 0
            
            self.face_analyzer.prepare(
                ctx_id=ctx_id,
                det_size=(480, 480)  # Có thể điều chỉnh cho performance
            )
            
            print(f"✅ InsightFace ready on {'GPU' if ctx_id >= 0 else 'CPU'}")
            
        except Exception as e:
            print(f"❌ Failed to initialize InsightFace: {str(e)}")
            return False
        
        # 2. Emotion detector đã được khởi tạo trong __init__
        print(f"😊 Emotion detector ready")
        
        # 3. Test backend connection
        if self.backend_sender.is_connected:
            print("🌐 Backend connection: CONNECTED")
        else:
            print("⚠️ Backend connection: DISCONNECTED")
        
        print("✅ System initialization complete!")
        return True
    
    def get_device_info(self):
        """Lấy thông tin device chi tiết"""
        device_info = {
            'system_device': self.device,
            'components': {}
        }
        
        # InsightFace device info
        if self.face_analyzer:
            device_info['components']['insightface'] = {
                'status': 'loaded',
                'device': 'GPU' if hasattr(self.face_analyzer, 'ctx_id') and self.face_analyzer.ctx_id >= 0 else 'CPU'
            }
        
        # Behavior detector device info
        if hasattr(self.behavior_detector, 'device'):
            device_info['components']['behavior_detector'] = {
                'status': 'loaded',
                'device': self.behavior_detector.device.upper(),
                'model': 'yolov8n-pose'
            }
        
        # Emotion detector
        device_info['components']['emotion_detector'] = {
            'status': 'loaded',
            'device': 'CPU',  # DeepFace chỉ chạy trên CPU
            'backend': 'DeepFace' if DEEPFACE_AVAILABLE else 'Simple'
        }
        
        # GPU memory info nếu có
        try:
            import torch
            if torch.cuda.is_available():
                device_info['gpu'] = {
                    'name': torch.cuda.get_device_name(0),
                    'memory_total': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
                    'memory_allocated': f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                    'memory_cached': f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
                }
        except:
            pass
        
        return device_info

    def detect_faces(self, image):
        """Phát hiện khuôn mặt với InsightFace"""
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = self.face_analyzer.get(image_rgb)
            
            face_results = []
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                
                face_roi = image[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue
                
                embedding = face.normed_embedding
                
                # Nhận diện cảm xúc - SỬA LỖI: DÙNG emotion_detector
                emotion, emotion_conf = self.emotion_detector.detect_emotion(face_roi)
                
                face_results.append({
                    'face_image': face_roi,
                    'bbox': (x1, y1, w, h),
                    'embedding': embedding,
                    'det_score': face.det_score,
                    'landmarks': face.kps if hasattr(face, 'kps') else None,
                    'emotion': emotion,
                    'emotion_confidence': emotion_conf
                })
            
            return face_results
            
        except Exception as e:
            print(f"❌ Lỗi detect faces: {str(e)}")
            return []

    def extract_features(self, face_data):
        """Trích xuất features từ khuôn mặt"""
        try:
            embedding = face_data['embedding']
            embedding = embedding.reshape(1, -1)
            features_normalized = self.l2_normalizer.transform(embedding)
            return features_normalized[0]
        except Exception as e:
            print(f"❌ Lỗi extract features: {str(e)}")
            return None

    def train_face_recognition(self, database_path="database"):
        """Train hệ thống nhận diện khuôn mặt"""
        if not os.path.exists(database_path):
            print(f"❌ Thư mục database không tồn tại: {database_path}")
            return False
        
        database = {}
        features_list = []
        labels_list = []
        
        print("📁 Đang xử lý database...")
        
        persons = [p for p in os.listdir(database_path) if os.path.isdir(os.path.join(database_path, p))]
        if len(persons) < 1:
            print("❌ Không có người nào trong database!")
            return False
        
        for person_name in persons:
            person_path = os.path.join(database_path, person_name)
            print(f"👤 Đang xử lý: {person_name}")
            person_features = []
            
            image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for image_file in image_files:
                image_path = os.path.join(person_path, image_file)
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                face_results = self.detect_faces(image)
                for face_data in face_results:
                    features = self.extract_features(face_data)
                    if features is not None:
                        person_features.append(features)
                        features_list.append(features)
                        labels_list.append(person_name)
            
            if person_features:
                database[person_name] = person_features
                print(f"  ➕ {person_name}: {len(person_features)} khuôn mặt")
        
        if len(features_list) == 0:
            print("❌ Không có dữ liệu để train!")
            return False
        
        print(f"\n📊 Thống kê database:")
        print(f"👥 Số người: {len(database)}")
        print(f"🖼️ Tổng khuôn mặt: {len(features_list)}")
        
        # Train SVM model
        print("\n🎯 Đang train SVM model...")
        self.svm_model = SVC(kernel='linear', probability=True, random_state=42)
        self.svm_model.fit(features_list, labels_list)
        
        accuracy = accuracy_score(labels_list, self.svm_model.predict(features_list))
        print(f"✅ Training hoàn tất! Accuracy: {accuracy:.4f}")
        
        # Lưu model
        with open("face_recognition_model.pkl", 'wb') as f:
            pickle.dump(self.svm_model, f)
        
        with open("face_database.pkl", 'wb') as f:
            pickle.dump({
                'database': database,
                'features': features_list,
                'labels': labels_list
            }, f)
        
        print("💾 Đã lưu model và database")
        return True

    def load_trained_model(self):
        """Load model đã train với fix numpy version"""
        try:
            print("📂 Đang load trained model...")
            
            if not os.path.exists("face_recognition_model.pkl"):
                print("❌ Không tìm thấy file model. Vui lòng train model trước.")
                return False
            
            # Thử load với pickle
            import pickle
            
            try:
                with open("face_recognition_model.pkl", 'rb') as f:
                    self.svm_model = pickle.load(f)
                print("✅ Đã load model thành công")
            except Exception as e:
                print(f"❌ Lỗi load model: {e}")
                
                # Thử với encoding khác
                try:
                    with open("face_recognition_model.pkl", 'rb') as f:
                        self.svm_model = pickle.load(f, encoding='latin1')
                    print("✅ Đã load model với encoding latin1")
                except:
                    print("❌ Không thể load model với bất kỳ encoding nào")
                    return False
            
            if hasattr(self.svm_model, 'classes_'):
                print(f"✅ Đã load trained model - {len(self.svm_model.classes_)} classes")
                return True
            else:
                print("⚠️ Model được load nhưng không có classes")
                return False
            
        except Exception as e:
            print(f"❌ Lỗi load model: {str(e)}")
            return False

    def recognize_face(self, face_data, threshold=0.4):
        """Nhận diện khuôn mặt"""
        if not hasattr(self, 'svm_model') or self.svm_model is None:
            return "Unknown", 0.0
        
        features = self.extract_features(face_data)
        if features is None:
            return "Unknown", 0.0
        
        try:
            probabilities = self.svm_model.predict_proba([features])[0]
            max_prob = np.max(probabilities)
            predicted_class = self.svm_model.classes_[np.argmax(probabilities)]
            
            if max_prob < threshold:
                return "Unknown", max_prob
            else:
                return predicted_class, max_prob
        except:
            return "Unknown", 0.0

    def _match_face_to_behavior(self, face_data, behavior_results):
        """Matching với tracking ID để ổn định hơn - DEBUG VERSION"""
        
        face_bbox = face_data['bbox']
        x, y, w, h = face_bbox
        face_center_x = x + w/2
        face_center_y = y + h/2
        
        
        # THÊM: Khởi tạo face_tracking_ids nếu chưa có
        if not hasattr(self, 'face_tracking_ids'):
            self.face_tracking_ids = {}
        
        # Tìm tracking ID cho face này (nếu có)
        face_id = self._assign_face_id(face_bbox, self.face_tracking_ids)
        
        # Cập nhật tracking
        self.face_tracking_ids[face_id] = {
            'bbox': face_bbox,
            'last_seen': time.time()
        }
        
        best_match = {'type': 'normal', 'confidence': 0.8, 'distance': float('inf')}
        
        # Nếu face có tracking ID, ưu tiên matching với behavior có cùng ID
        for behavior_idx, behavior in enumerate(behavior_results):
            if behavior['bbox'] is not None:
                try:
                    bx1, by1, bx2, by2 = behavior['bbox'].astype(int)
                    behavior_center_x = (bx1 + bx2) / 2
                    behavior_center_y = (by1 + by2) / 2
                    
                    # Tính khoảng cách Euclid
                    distance = np.sqrt((face_center_x - behavior_center_x)**2 + (face_center_y - behavior_center_y)**2)
                    
                    # Tính IoU (Intersection over Union)
                    intersection_x1 = max(x, bx1)
                    intersection_y1 = max(y, by1)
                    intersection_x2 = min(x + w, bx2)
                    intersection_y2 = min(y + h, by2)
                    
                    if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                        face_area = w * h
                        behavior_area = (bx2 - bx1) * (by2 - by1)
                        union_area = face_area + behavior_area - intersection_area
                        
                        iou = intersection_area / union_area if union_area > 0 else 0
                        
                    else:
                        iou = 0
                    
                    if distance < best_match['distance']:
                        best_match = {
                            'type': behavior['behavior'],
                            'confidence': min(0.9, max(0.7, 1 - distance/300)),
                            'distance': distance,
                            'iou': iou
                        }
                        
                except Exception as e:
                    continue
            else:
                print(f"  Behavior {behavior_idx}: No bbox")
        
        return best_match

    def _assign_face_id(self, face_bbox, face_tracking_ids):
        """Gán tracking ID cho face dựa trên bbox và tracking system"""
        x, y, w, h = face_bbox
        face_center_x = x + w/2
        face_center_y = y + h/2
        
        # Tìm ID gần nhất
        best_id = None
        min_distance = float('inf')
        
        for face_id, bbox_data in face_tracking_ids.items():
            if isinstance(bbox_data, dict) and 'bbox' in bbox_data:
                tracked_bbox = bbox_data['bbox']
                tracked_x, tracked_y, tracked_w, tracked_h = tracked_bbox
                tracked_center_x = tracked_x + tracked_w/2
                tracked_center_y = tracked_y + tracked_h/2
                
                distance = np.sqrt((face_center_x - tracked_center_x)**2 + 
                                  (face_center_y - tracked_center_y)**2)
                
                # Nếu khoảng cách < 100 pixels, coi như cùng một người
                if distance < 100 and distance < min_distance:
                    min_distance = distance
                    best_id = face_id
        
        # Nếu không tìm thấy, tạo ID mới
        if best_id is None:
            best_id = len(face_tracking_ids)
        
        return best_id

    def process_frame_with_engagement(self, frame, face_results, behavior_results):
        """Xử lý frame với tính toán engagement nâng cao"""
        student_data_list = []
        
        for i, face_data in enumerate(face_results):
            bbox = face_data['bbox']
            x, y, w, h = bbox
            emotion = face_data['emotion']
            emotion_conf = face_data['emotion_confidence']
            
            if hasattr(self, 'svm_model') and self.svm_model:
                name, confidence = self.recognize_face(face_data)
            else:
                name, confidence = "Unknown", 0.0
            
            # Ghép với hành vi
            matched_behavior = self._match_face_to_behavior(face_data, behavior_results)
            behavior = matched_behavior['type']
            behavior_confidence = matched_behavior['confidence']
            
            # Tính engagement score
            engagement_result = self.engagement_calculator.calculate_engagement(
                student_id=f"{name}_{i}",
                emotion=emotion,
                emotion_confidence=emotion_conf,
                behavior=behavior,
                behavior_confidence=behavior_confidence,
                bbox=(x, y, w, h)
            )
            
            student_data = {
                'id': i + 1,
                'name': name,
                'emotion': emotion,
                'emotion_confidence': emotion_conf,
                'behavior': behavior,
                'engagement': engagement_result['engagement_score'],  # NEW
                'concentration_level': engagement_result['concentration_level'],  # NEW
                'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                'face_confidence': confidence,
                'engagement_details': engagement_result  # Chi tiết tính toán
            }
            
            student_data_list.append(student_data)
        
        return student_data_list

    def get_class_engagement_report(self):
        """Lấy báo cáo engagement cho cả lớp"""
        if hasattr(self, 'last_detection_results'):
            return self.engagement_calculator.get_engagement_report(self.last_detection_results)
        return None

    def _get_engagement_color(self, score):
        """Lấy màu dựa trên engagement score"""
        return self.engagement_calculator._get_engagement_color(score)

# ==================== FLASK API ENDPOINTS ====================
# ==================== VIDEO STREAM ENDPOINT ====================
@app.route('/video_feed')
def video_feed():
    """Endpoint stream video MJPEG - CHỈ CAMERA THƯỜNG"""
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0'
                    })

@app.route('/')
def index():
    """Trang chính với video stream camera thường"""
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Camera Live Stream</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #1a1a2e;
                color: white;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }
            
            .header {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .header h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
                background: linear-gradient(45deg, #fff, #e0e0e0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }
            
            .container {
                flex: 1;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 20px;
                max-width: 1200px;
                margin: 0 auto;
                width: 100%;
            }
            
            .video-container {
                width: 100%;
                max-width: 800px;
                background: #16213e;
                border-radius: 20px;
                overflow: hidden;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
                margin-bottom: 30px;
                position: relative;
                border: 3px solid #4cc9f0;
            }
            
            .video-container::before {
                content: '';
                position: absolute;
                top: -2px;
                left: -2px;
                right: -2px;
                bottom: -2px;
                background: linear-gradient(45deg, #4cc9f0, #4361ee, #3a0ca3, #7209b7);
                border-radius: 22px;
                z-index: -1;
                animation: border-glow 3s ease-in-out infinite alternate;
            }
            
            @keyframes border-glow {
                0% {
                    opacity: 0.5;
                }
                100% {
                    opacity: 1;
                }
            }
            
            .video-header {
                background: rgba(0, 0, 0, 0.7);
                padding: 15px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 2px solid #4cc9f0;
            }
            
            .video-header h3 {
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 1.2rem;
            }
            
            .status-indicator {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .status-dot {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #4CAF50;
                animation: pulse 2s infinite;
            }
            
            .status-dot.offline {
                background: #f44336;
                animation: none;
            }
            
            @keyframes pulse {
                0%, 100% {
                    opacity: 1;
                    transform: scale(1);
                }
                50% {
                    opacity: 0.7;
                    transform: scale(1.1);
                }
            }
            
            #videoStream {
                width: 100%;
                display: block;
                background: #000;
                min-height: 480px;
                object-fit: cover;
            }
            
            .controls {
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
                justify-content: center;
                margin: 20px 0;
            }
            
            .btn {
                padding: 12px 24px;
                border: none;
                border-radius: 50px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 8px;
                text-decoration: none;
            }
            
            .btn-primary {
                background: linear-gradient(135deg, #4cc9f0, #4361ee);
                color: white;
                box-shadow: 0 4px 15px rgba(76, 201, 240, 0.4);
            }
            
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(76, 201, 240, 0.6);
            }
            
            .btn-secondary {
                background: rgba(255, 255, 255, 0.1);
                color: white;
                border: 2px solid #4cc9f0;
            }
            
            .btn-secondary:hover {
                background: rgba(76, 201, 240, 0.2);
                transform: translateY(-2px);
            }
            
            .stats {
                display: flex;
                gap: 30px;
                margin-top: 20px;
                flex-wrap: wrap;
                justify-content: center;
            }
            
            .stat-box {
                background: rgba(255, 255, 255, 0.05);
                padding: 20px;
                border-radius: 15px;
                min-width: 200px;
                text-align: center;
                border: 1px solid rgba(76, 201, 240, 0.2);
                transition: transform 0.3s ease;
            }
            
            .stat-box:hover {
                transform: translateY(-5px);
                border-color: #4cc9f0;
            }
            
            .stat-value {
                font-size: 2rem;
                font-weight: bold;
                background: linear-gradient(45deg, #4cc9f0, #4361ee);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 10px 0;
            }
            
            .stat-label {
                color: #a0a0a0;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .info-panel {
                background: rgba(22, 33, 62, 0.8);
                padding: 20px;
                border-radius: 15px;
                margin-top: 30px;
                border: 1px solid rgba(76, 201, 240, 0.3);
                width: 100%;
                max-width: 800px;
            }
            
            .info-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 15px;
            }
            
            .info-item {
                padding: 10px;
                background: rgba(0, 0, 0, 0.2);
                border-radius: 8px;
                border-left: 4px solid #4cc9f0;
            }
            
            .footer {
                text-align: center;
                padding: 20px;
                margin-top: 40px;
                color: #a0a0a0;
                font-size: 0.9rem;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            @media (max-width: 768px) {
                .container {
                    padding: 10px;
                }
                
                .video-container {
                    border-radius: 10px;
                }
                
                .header h1 {
                    font-size: 1.8rem;
                }
                
                .btn {
                    padding: 10px 20px;
                    font-size: 0.9rem;
                }
                
                .stats {
                    gap: 15px;
                }
                
                .stat-box {
                    min-width: 150px;
                    padding: 15px;
                }
            }
            
            .loading {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.8);
                z-index: 1000;
                justify-content: center;
                align-items: center;
                flex-direction: column;
            }
            
            .loading.show {
                display: flex;
            }
            
            .spinner {
                width: 50px;
                height: 50px;
                border: 5px solid rgba(255, 255, 255, 0.1);
                border-top: 5px solid #4cc9f0;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-bottom: 20px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <!-- Header -->
        <div class="header">
            <h1>📹 Camera Live Stream</h1>
            <p>Real-time camera feed without AI processing</p>
        </div>
        
        <!-- Loading Overlay -->
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Connecting to camera...</p>
        </div>
        
        <!-- Main Container -->
        <div class="container">
            <!-- Video Container -->
            <div class="video-container">
                <div class="video-header">
                    <h3>
                        <span>🔴 LIVE</span>
                        <span>|</span>
                        <span>Camera Feed</span>
                    </h3>
                    <div class="status-indicator">
                        <div class="status-dot" id="statusDot"></div>
                        <span id="statusText">Connecting...</span>
                    </div>
                </div>
                <img id="videoStream" src="/video_feed" alt="Live Camera Stream">
            </div>
            
            <!-- Controls -->
            <div class="controls">
                <button class="btn btn-primary" onclick="refreshStream()">
                    <span>🔄</span>
                    Refresh Stream
                </button>
                <button class="btn btn-secondary" onclick="toggleFullscreen()">
                    <span>📺</span>
                    Fullscreen
                </button>
                <button class="btn btn-secondary" onclick="captureFrame()">
                    <span>📸</span>
                    Capture Photo
                </button>
                <a href="/video_feed" target="_blank" class="btn btn-secondary">
                    <span>🔗</span>
                    Direct Stream Link
                </a>
            </div>
            
            <!-- Statistics -->
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-label">Stream Status</div>
                    <div class="stat-value" id="streamStatus">Live</div>
                    <div>Camera Feed Active</div>
                </div>
                
                <div class="stat-box">
                    <div class="stat-label">Resolution</div>
                    <div class="stat-value">640x480</div>
                    <div>Video Quality</div>
                </div>
                
                <div class="stat-box">
                    <div class="stat-label">Frame Rate</div>
                    <div class="stat-value" id="fpsCounter">30</div>
                    <div>Frames Per Second</div>
                </div>
            </div>
            
            <!-- Information Panel -->
            <div class="info-panel">
                <h3>ℹ️ Stream Information</h3>
                <div class="info-grid">
                    <div class="info-item">
                        <strong>Server URL:</strong>
                        <div style="word-break: break-all; color: #4cc9f0; margin-top: 5px;">
                            http://localhost:5000
                        </div>
                    </div>
                    
                    <div class="info-item">
                        <strong>Stream Endpoint:</strong>
                        <div style="word-break: break-all; color: #4cc9f0; margin-top: 5px;">
                            /video_feed
                        </div>
                    </div>
                    
                    <div class="info-item">
                        <strong>Connection Type:</strong>
                        <div style="color: #4cc9f0; margin-top: 5px;">
                            MJPEG Stream
                        </div>
                    </div>
                    
                    <div class="info-item">
                        <strong>Last Updated:</strong>
                        <div style="color: #4cc9f0; margin-top: 5px;" id="lastUpdate">
                            Just now
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>Camera Stream System | Real-time Video Feed</p>
            <p>Server running on http://localhost:5000</p>
        </div>
        
        <script>
            // DOM Elements
            const videoStream = document.getElementById('videoStream');
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            const streamStatus = document.getElementById('streamStatus');
            const fpsCounter = document.getElementById('fpsCounter');
            const lastUpdate = document.getElementById('lastUpdate');
            const loading = document.getElementById('loading');
            
            // Variables
            let frameCount = 0;
            let lastTime = Date.now();
            let isConnected = false;
            let refreshInterval;
            
            // Auto-refresh stream to avoid cache
            function refreshStream() {
                console.log('🔄 Refreshing stream...');
                const timestamp = Date.now();
                videoStream.src = `/video_feed?t=${timestamp}`;
                updateLastUpdate();
                showLoading();
                
                // Hide loading after 2 seconds
                setTimeout(() => {
                    hideLoading();
                }, 2000);
            }
            
            // Toggle fullscreen
            function toggleFullscreen() {
                if (!document.fullscreenElement) {
                    videoStream.requestFullscreen().catch(err => {
                        console.log('Fullscreen error:', err);
                    });
                } else {
                    document.exitFullscreen();
                }
            }
            
            // Capture frame
            function captureFrame() {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                
                canvas.width = videoStream.videoWidth || 640;
                canvas.height = videoStream.videoHeight || 480;
                
                ctx.drawImage(videoStream, 0, 0, canvas.width, canvas.height);
                
                const link = document.createElement('a');
                link.download = `capture_${Date.now()}.jpg`;
                link.href = canvas.toDataURL('image/jpeg', 0.9);
                link.click();
                
                alert('📸 Photo captured!');
            }
            
            // Update connection status
            function updateStatus(connected) {
                isConnected = connected;
                
                if (connected) {
                    statusDot.classList.remove('offline');
                    statusText.textContent = 'Connected';
                    streamStatus.textContent = 'Live';
                    streamStatus.style.color = '#4CAF50';
                } else {
                    statusDot.classList.add('offline');
                    statusText.textContent = 'Disconnected';
                    streamStatus.textContent = 'Offline';
                    streamStatus.style.color = '#f44336';
                }
            }
            
            // Calculate FPS
            function calculateFPS() {
                frameCount++;
                const now = Date.now();
                const delta = now - lastTime;
                
                if (delta >= 1000) { // Update every second
                    const fps = Math.round((frameCount * 1000) / delta);
                    fpsCounter.textContent = fps;
                    frameCount = 0;
                    lastTime = now;
                }
                
                requestAnimationFrame(calculateFPS);
            }
            
            // Update last update time
            function updateLastUpdate() {
                const now = new Date();
                const timeString = now.toLocaleTimeString('en-US', {
                    hour12: true,
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit'
                });
                lastUpdate.textContent = `${timeString}`;
            }
            
            // Show loading
            function showLoading() {
                loading.classList.add('show');
            }
            
            // Hide loading
            function hideLoading() {
                loading.classList.remove('show');
            }
            
            // Check stream health
            async function checkStreamHealth() {
                try {
                    const response = await fetch('/api/health');
                    const data = await response.json();
                    
                    updateStatus(data.status === 'healthy');
                    
                    // If disconnected, try to reconnect
                    if (!data.status === 'healthy' && isConnected) {
                        console.log('Stream disconnected, attempting to reconnect...');
                        refreshStream();
                    }
                } catch (error) {
                    console.log('Health check failed:', error);
                    updateStatus(false);
                }
            }
            
            // Handle stream errors
            videoStream.onerror = function() {
                console.log('Stream error occurred');
                updateStatus(false);
                refreshStream();
            };
            
            videoStream.onload = function() {
                console.log('Stream loaded successfully');
                updateStatus(true);
                hideLoading();
            };
            
            // Initialize
            function init() {
                // Start FPS calculation
                calculateFPS();
                
                // Initial refresh
                refreshStream();
                
                // Start health checks every 5 seconds
                setInterval(checkStreamHealth, 5000);
                
                // Auto-refresh every 30 seconds to prevent timeout
                refreshInterval = setInterval(refreshStream, 30000);
                
                // Initial status check
                checkStreamHealth();
                
                // Keyboard shortcuts
                document.addEventListener('keydown', (e) => {
                    // Space to refresh
                    if (e.code === 'Space') {
                        e.preventDefault();
                        refreshStream();
                    }
                    
                    // F for fullscreen
                    if (e.code === 'KeyF') {
                        e.preventDefault();
                        toggleFullscreen();
                    }
                    
                    // C for capture
                    if (e.code === 'KeyC') {
                        e.preventDefault();
                        captureFrame();
                    }
                });
                
                console.log('🎥 Camera stream system initialized');
            }
            
            // Start when page loads
            window.addEventListener('load', init);
            
            // Cleanup on page unload
            window.addEventListener('beforeunload', () => {
                if (refreshInterval) {
                    clearInterval(refreshInterval);
                }
            });
        </script>
    </body>
    </html>
    '''
    return html

@app.route('/api/capture', methods=['POST'])
def capture_frame():
    """Chụp frame hiện tại"""
    global camera_manager
    
    try:
        frame = camera_manager.get_latest_frame()
        if frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            return jsonify({
                'status': 'success',
                'message': f'Frame captured: {filename}',
                'filename': filename
            })
        
        return jsonify({'status': 'error', 'message': 'No frame available'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    
@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Khởi động camera"""
    global camera_manager
    
    try:
        camera_index = request.json.get('camera_index', 0)
        camera_manager = CameraManager(camera_index=camera_index)
        
        if camera_manager.start():
            return jsonify({
                'status': 'success',
                'message': f'Camera {camera_index} started',
                'camera_index': camera_index
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Cannot start camera'
            }), 500
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Dừng camera"""
    global camera_manager
    
    try:
        camera_manager.stop()
        return jsonify({
            'status': 'success',
            'message': 'Camera stopped'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/camera/list', methods=['GET'])
def list_cameras():
    """Liệt kê các camera có sẵn"""
    cameras = []
    
    for i in range(10):  # Thử 10 camera index
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # DSHOW cho Windows
        if cap.isOpened():
            cameras.append({
                'index': i,
                'name': f'Camera {i}',
                'resolution': f'{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}',
                'fps': int(cap.get(cv2.CAP_PROP_FPS))
            })
            cap.release()
    
    return jsonify({
        'status': 'success',
        'cameras': cameras
    })
    
@app.route('/api/status', methods=['GET'])
def get_status():
    """API trả về trạng thái AI model"""
    global ai_running, system
    
    try:
        backend_connected = False
        if system and hasattr(system, 'backend_sender'):
            backend_connected = system.backend_sender.is_connected
        
        return jsonify({
            'status': 'running' if ai_running else 'stopped',
            'ai_system_initialized': system is not None,
            'backend_connected': backend_connected,
            'camera_source': 'webcam',
            'has_trained_model': hasattr(system, 'svm_model') and system.svm_model is not None,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/control', methods=['POST'])
def control_model():
    """API điều khiển model từ web"""
    global ai_running, ai_thread, system
    
    data = request.get_json()
    if not data:
        print("⚠️ No JSON data received in request")
        return jsonify({'error': 'No data provided'}), 400
    
    action = data.get('action')
    
    if not action:
        print("⚠️ No 'action' field in request data")
        return jsonify({'error': 'No action specified'}), 400
    
    print(f"📡 Received action: {action}")
    
    if action == 'start':
        with ai_status_lock:
            if ai_running:
                return jsonify({
                    'status': 'already_running',
                    'message': 'AI is already running'
                })
            
            # Khởi tạo hệ thống nếu chưa có
            if system is None:
                gpu_available, device = setup_gpu()
                system = CompleteRecognitionSystem(device=device)
                if not system.initialize_system():
                    return jsonify({
                        'error': 'Failed to initialize AI system'
                    }), 500
                
                # Thử load trained model
                system.load_trained_model()
            
            # Start AI thread
            ai_running = True
            ai_thread = threading.Thread(target=ai_processing_loop, daemon=True)
            ai_thread.start()
            
            return jsonify({
                'status': 'success',
                'message': 'AI model started successfully',
                'device': system.device,
                'timestamp': datetime.now().isoformat()
            })
        
    elif action == 'stop':
        with ai_status_lock:
            if not ai_running:
                return jsonify({
                    'status': 'already_stopped',
                    'message': 'AI is already stopped'
                })
            
            ai_running = False
            
            # Chờ thread dừng
            if ai_thread:
                ai_thread.join(timeout=3)
            
            return jsonify({
                'status': 'success',
                'message': 'AI model stopped successfully',
                'timestamp': datetime.now().isoformat()
            })
    
    else:
        return jsonify({'error': 'Invalid action'}), 400

@app.route('/api/start_ai', methods=['POST'])
def start_ai():
    """API khởi động AI"""
    global ai_running, ai_thread, system
    
    print("📡 /api/start_ai endpoint called")
    
    with ai_status_lock:
        if ai_running:
            return jsonify({
                'status': 'already_running',
                'message': 'AI is already running',
                'timestamp': datetime.now().isoformat()
            })
        
        # Khởi tạo hệ thống nếu chưa có
        if system is None:
            gpu_available, device = setup_gpu()
            system = CompleteRecognitionSystem(device=device)
            if not system.initialize_system():
                return jsonify({
                    'error': 'Failed to initialize AI system',
                    'timestamp': datetime.now().isoformat()
                }), 500
            
            # Thử load trained model
            system.load_trained_model()
        
        # Start AI thread
        ai_running = True
        ai_thread = threading.Thread(target=ai_processing_loop, daemon=True)
        ai_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'AI model started successfully',
            'device': system.device,
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/stop_ai', methods=['POST'])
def stop_ai():
    """API dừng AI"""
    global ai_running, ai_thread
    
    print("📡 /api/stop_ai endpoint called")
    
    with ai_status_lock:
        if not ai_running:
            return jsonify({
                'status': 'already_stopped',
                'message': 'AI is already stopped',
                'timestamp': datetime.now().isoformat()
            })
        
        ai_running = False
        
        # Chờ thread dừng
        if ai_thread:
            ai_thread.join(timeout=3)
        
        return jsonify({
            'status': 'success',
            'message': 'AI model stopped successfully',
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/latest_results', methods=['GET'])
def get_latest_results():
    """API lấy kết quả detection mới nhất"""
    global last_detection_results, last_detection_time, detection_lock
    
    # 🔴 FIX: Sử dụng lock khi đọc dữ liệu
    with detection_lock:
        current_results = last_detection_results.copy() if last_detection_results else []
        current_time = last_detection_time
    
    if not current_results:
        return jsonify({
            'status': 'no_data',
            'message': 'No detection results available',
            'timestamp': datetime.now().isoformat()
        })
    
    # 🔴 THÊM: Debug log để biết API đang trả về gì
    print(f"📡 API /latest_results: returning {len(current_results)} detections")
    
    return jsonify({
        'status': 'success',
        'count': len(current_results),
        'results': current_results,
        'last_update': current_time.isoformat() if current_time else None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/engagement', methods=['GET'])
def get_engagement_data():
    """API lấy dữ liệu engagement của lớp học"""
    global system
    
    if not system or not hasattr(system, 'engagement_calculator'):
        return jsonify({'error': 'System not initialized'}), 400
    
    report = system.get_class_engagement_report()
    
    if report:
        return jsonify({
            'status': 'success',
            'report': report,
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({
            'status': 'no_data',
            'message': 'No engagement data available',
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/debug/ai_status', methods=['GET'])
def debug_ai_status():
    """Debug endpoint để kiểm tra trạng thái AI system"""
    global ai_running, system, last_detection_results, last_detection_time, ai_thread
    
    with detection_lock:
        results_count = len(last_detection_results) if last_detection_results else 0
        last_time = last_detection_time.isoformat() if last_detection_time else "None"
    
    return jsonify({
        'ai_running': ai_running,
        'system_initialized': system is not None,
        'detection_results_count': results_count,
        'last_detection_time': last_time,
        'thread_active': ai_thread is not None and ai_thread.is_alive() if ai_thread else False,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ai_running': ai_running,
        'ai_system_initialized': system is not None,
        'camera_source': 'webcam',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    global system
    
    config = {
        'camera_source': 'webcam',
        'ai_system': 'CompleteRecognitionSystem',
        'has_face_detector': system and hasattr(system, 'face_analyzer') and system.face_analyzer is not None,
        'has_behavior_detector': system and hasattr(system, 'behavior_detector') and system.behavior_detector is not None,
        'has_svm_model': system and hasattr(system, 'svm_model') and system.svm_model is not None,
        'has_backend_connection': system and hasattr(system, 'backend_sender') and system.backend_sender.is_connected,
        'has_engagement_calculator': system and hasattr(system, 'engagement_calculator')  # NEW
    }
    
    if system:
        config.update({
            'device': system.device,
            'model_name': system.model_name
        })
    
    return jsonify(config)

# ==================== AI PROCESSING LOOP ====================
def process_and_send_engagement_data(system, student_data_list):
    """Xử lý và gửi engagement data đến backend"""
    if not system or not student_data_list:
        return
    
    try:
        # Chuẩn bị batch data
        engagement_batch = []
        
        for student in student_data_list:
            engagement_item = {
                "student_id": f"AI_{student.get('id', hash(str(student)) % 10000):04d}",
                "student_name": student.get('name', 'Unknown Student'),
                "name": student.get('name', 'Unknown Student'),
                "engagement_score": student.get('engagement', 75.0),
                "concentration_level": student.get('concentration_level', 'medium'),
                "emotion": student.get('emotion', 'neutral'),
                "behavior": student.get('behavior', 'normal'),
                "emotion_confidence": student.get('emotion_confidence', 0.5),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "session_id": f"session_{int(time.time())}",
                "recorded_by": "AI Recognition System",
                "class_name": "AI Classroom"
            }
            engagement_batch.append(engagement_item)
        
        # Gửi qua backend sender
        if hasattr(system, 'backend_sender') and system.backend_sender.is_connected:
            # Option 1: Gửi batch qua endpoint mới
            system.backend_sender.send_engagement_batch(engagement_batch)
        
        print(f"📤 Sent engagement data for {len(engagement_batch)} students")
        
    except Exception as e:
        print(f"❌ Error processing engagement data: {e}")
        
def ai_processing_loop():
    """Thread xử lý AI - DÙNG CAMERA MANAGER"""
    global ai_running, system, last_detection_results, last_detection_time
    global camera_manager, detection_lock
    
    print("🤖 Starting AI processing loop with shared camera...")
    
    # Khởi động camera với retry
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries and ai_running:
        if camera_manager.start():
            print("✅ Camera started successfully")
            break
        else:
            retry_count += 1
            print(f"⚠️ Không thể khởi động camera, thử lại lần {retry_count}/{max_retries}")
            time.sleep(2)
    
    if not camera_manager.is_running:
        print("❌ Không thể khởi động camera sau nhiều lần thử")
        with ai_status_lock:
            ai_running = False
        return
    
    frame_count = 0
    fps_time = time.time()
    fps_counter = 0
    last_batch_sent = 0
    consecutive_errors = 0
    max_consecutive_errors = 10
    
    while ai_running:
        try:
            # 🔴 ĐỌC FRAME TRỰC TIẾP TỪ CAMERA MANAGER
            frame = camera_manager.read_frame()
            
            if frame is None:
                consecutive_errors += 1
                if consecutive_errors > max_consecutive_errors:
                    print("🔄 Quá nhiều lỗi liên tiếp, đang khởi động lại camera...")
                    camera_manager.stop()
                    time.sleep(1)
                    if camera_manager.start():
                        consecutive_errors = 0
                        time.sleep(0.5)
                    else:
                        print("❌ Không thể khởi động lại camera, dừng AI loop")
                        break
                else:
                    time.sleep(0.1)
                continue
            
            # Reset error counter
            consecutive_errors = 0
            
            frame_count += 1
            fps_counter += 1
            
            # Tính FPS
            current_time = time.time()
            if current_time - fps_time >= 2.0:
                fps = fps_counter / (current_time - fps_time)
                fps_counter = 0
                fps_time = current_time
                # Có thể bật log FPS khi debug
                if frame_count % 60 == 0:
                    print(f"📊 AI Loop FPS: {fps:.1f}, Frame: {frame_count}")
            
            # Process AI
            student_data_list = []
            if system and frame_count % 2 == 0:
                face_results = system.detect_faces(frame)
                behavior_results = []
                
                if hasattr(system.behavior_detector, 'pose_model'):
                    behavior_results = system.behavior_detector.detect_behavior(frame)
                
                if face_results:
                    student_data_list = system.process_frame_with_engagement(
                        frame, face_results, behavior_results
                    )
                
                # Lưu kết quả
                with detection_lock:
                    if student_data_list:
                        last_detection_results = student_data_list.copy()
                        last_detection_time = datetime.now()
                    else:
                        last_detection_results = []
                        last_detection_time = datetime.now()
                
                # ==================== 🔴 THÊM: GỬI BATCH DATA ====================
                if student_data_list and system.backend_sender.is_connected:
                    # Gửi batch mỗi 30 frames (~1 giây nếu 30fps)
                    if frame_count - last_batch_sent >= 30:
                        try:
                            # Chuẩn bị detections cho batch
                            detections_for_batch = []
                            for student in student_data_list:
                                # Map student_data sang format của send_detection_batch
                                detection_item = {
                                    "name": student.get('name', 'Unknown'),
                                    "id": student.get('id', 0),
                                    "emotion": student.get('emotion', 'neutral'),
                                    "emotion_confidence": student.get('emotion_confidence', 0.5),
                                    "behavior": student.get('behavior', 'normal'),
                                    "engagement": student.get('engagement', 50.0),
                                    "bbox": [
                                        student['bbox']['x'],
                                        student['bbox']['y'], 
                                        student['bbox']['width'],
                                        student['bbox']['height']
                                    ] if 'bbox' in student else None,
                                    "face_confidence": student.get('face_confidence', 0.5)
                                }
                                detections_for_batch.append(detection_item)
                            
                            # Gọi send_detection_batch
                            if detections_for_batch:
                                success = system.backend_sender.send_detection_batch(
                                    detections=detections_for_batch,
                                    fps=fps,
                                    frame_count=frame_count
                                )
                                
                                if success:
                                    last_batch_sent = frame_count
                                    # Log nhẹ để không spam console
                                    if frame_count % 90 == 0:  # Mỗi 3 batch (3 giây)
                                        print(f"📦 AI Loop: Sent batch with {len(detections_for_batch)} detections")
                                        
                        except Exception as batch_error:
                            print(f"⚠️ Error in batch sending: {batch_error}")
                            # Không crash thread vì lỗi batch
            
            # ==================== 🔴 THÊM: GỬI ATTENDANCE DATA ====================
            # Gửi attendance cho các student detected
            if student_data_list and hasattr(system, 'attendance_system'):
                for student_data in student_data_list:
                    name = student_data.get('name', 'Unknown')
                    if name != "Unknown" and student_data.get('face_confidence', 0) > 0.6:
                        # Gửi attendance với tần suất thấp hơn
                        if frame_count % 60 == 0:  # Mỗi 2 giây
                            system.attendance_system.mark_attendance(
                                name=name,
                                emotion=student_data.get('emotion', 'neutral'),
                                emotion_confidence=student_data.get('emotion_confidence', 0.5),
                                behavior=student_data.get('behavior', 'normal'),
                                engagement=student_data.get('engagement', 50.0),
                                concentration_level=student_data.get('concentration_level', 'medium'),
                                confidence=student_data.get('face_confidence', 0.5)
                            )
            
            # Giữ FPS ổn định
            time.sleep(0.001)  # Rất nhỏ, vì camera đã có FPS cố định
            
        except Exception as e:
            print(f"⚠️ Error in AI loop: {e}")
            import traceback
            traceback.print_exc()
            consecutive_errors += 1
            time.sleep(0.1)
    
    print("✅ AI processing stopped")
    camera_manager.stop()


def generate_mjpeg():
    """Generate MJPEG stream từ camera manager - CHỈ HIỂN THỊ CAMERA THƯỜNG"""
    global camera_manager
    
    while True:
        # 🔴 LẤY FRAME TRỰC TIẾP TỪ CAMERA MANAGER
        frame = camera_manager.get_latest_frame()
        
        if frame is not None:
            try:
                # 🔴 CHỈ HIỂN THỊ CAMERA THƯỜNG, KHÔNG CÓ AI OVERLAY
                # Không xử lý AI, không vẽ bounding box, không overlay
                
                # 🔴 THÊM: Resize để giảm bandwidth nếu cần
                display_frame = cv2.resize(frame, (640, 480))
                
                # Encode frame thành JPEG
                ret, jpeg = cv2.imencode('.jpg', display_frame, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 80])  # Chất lượng vừa
                if ret:
                    frame_bytes = jpeg.tobytes()
                    
                    # Tạo MJPEG frame
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + 
                           frame_bytes + b'\r\n')
            
            except Exception as e:
                # Log lỗi nhẹ
                print(f"Stream encode error: {e}")
                # Vẫn yield frame rỗng để không break stream
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       b'\r\n')
        else:
            # Nếu không có frame, đợi một chút
            time.sleep(0.1)
        
        time.sleep(0.033)  # ~30 FPS

# ==================== CÁC HÀM PHỤ TRỢ ====================
def create_folder_structure():
    """Tạo cấu trúc thư mục"""
    folders = [
        "database",
        "database/person1",
        "database/person2", 
        "database/person3",
        "test_images"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Đã tạo: {folder}/")
    
    print("\n📁 Cấu trúc thư mục đã được tạo!")

def train_model():
    """Train model từ database"""
    gpu_available, device = setup_gpu()
    system = CompleteRecognitionSystem(device=device)
    
    if not system.initialize_system():
        return
    
    if not os.path.exists("database"):
        os.makedirs("database")
        print("📁 Đã tạo thư mục 'database'")
        print("💡 Hãy thêm ảnh của bạn vào thư mục database/person1, database/person2, etc.")
        return
    
    success = system.train_face_recognition()
    if success:
        print("🎉 Train model thành công!")
    else:
        print("❌ Train model thất bại!")

def view_attendance():
    """Xem lịch sử điểm danh"""
    attendance_system = AttendanceSystem()
    attendance_system.view_attendance()

def test_backend_connection():
    """Kiểm tra kết nối backend"""
    sender = EnhancedBackendDataSender()
    if sender.is_connected:
        print("✅ Kết nối backend: THÀNH CÔNG")
    else:
        print("❌ Kết nối backend: THẤT BẠI")

def troubleshoot_gpu():
    """Khắc phục sự cố GPU"""
    print("\n" + "="*60)
    print("🔧 KHẮC PHỤC SỰ CỐ GPU")
    print("="*60)
    
    print("1. 📋 Kiểm tra card đồ họa:")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU được phát hiện")
            print(result.stdout.split('\n')[0])  # Hiển thị dòng đầu tiên
        else:
            print("❌ Không tìm thấy NVIDIA GPU hoặc driver")
    except:
        print("❌ Không thể chạy nvidia-smi")
    
    print("\n2. 🔄 Cài đặt PyTorch với CUDA support:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n3. 🎯 Cài đặt Ultralytics ổn định:")
    print("   pip install ultralytics==8.0.196")
    
    print("\n4. 🔧 Fix numpy version issue:")
    print("   pip install numpy==1.24.3")
    print("   Hoặc thêm code fix:")
    print("   import numpy")
    print("   if hasattr(numpy, '_core'):")
    print("       numpy.core.multiarray = numpy._core.multiarray")
    
    print("="*60)

def start_flask_server():
    """Khởi động Flask server"""
    print("\n" + "="*80)
    print("🌐 FLASK API SERVER")
    print("="*80)
    print("📡 Endpoints:")
    print("   • GET  /api/status         - Check AI status")
    print("   • POST /api/control        - Control AI (action: start/stop)")
    print("   • POST /api/start_ai       - Start AI model")
    print("   • POST /api/stop_ai        - Stop AI model")
    print("   • GET  /api/latest_results - Get latest detection results")
    print("   • GET  /api/engagement     - Get engagement report (NEW)")
    print("   • GET  /api/health         - Health check")
    print("   • GET  /api/config         - Get configuration")
    print("   • GET  /api/debug/ai_status - Debug AI status")
    print("="*80)
    print("🎯 AI System: Ready to be controlled via API")
    print("📊 Engagement System: Calculates focus score based on emotion and behavior")
    print("📷 Camera source: Webcam Direct (Camera 0)")  # 🔴 SỬA DÒNG NÀY
    print("📊 Backend connection: Will send attendance, emotion, behavior, engagement data")
    print("="*80)
    print("🚀 Starting Flask server on http://localhost:5000")
    
    # Chạy Flask server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

# ==================== REAL-TIME RECOGNITION ====================
def real_time_recognition():
    """Chạy real-time recognition với camera dùng chung cho AI và streaming"""
    global camera_manager, system
    
    # Kiểm tra và thiết lập GPU
    gpu_available, device = setup_gpu()
    
    # Khởi tạo hệ thống
    system = CompleteRecognitionSystem(device=device)
    
    if not system.initialize_system():
        print("❌ Không thể khởi tạo hệ thống")
        return
    
    model_loaded = system.load_trained_model()
    if not model_loaded:
        print("⚠️ Chạy ở chế độ chỉ detect cảm xúc và hành vi")
    
    # ==================== THÊM: Camera Manager (DÙNG CHUNG) ====================
    camera_manager = CameraManager(camera_index=0)
    
    # Thử khởi động camera với retry
    max_retries = 3
    for attempt in range(max_retries):
        print(f"🔍 Đang khởi động camera (lần thử {attempt + 1}/{max_retries})...")
        if camera_manager.start():
            break
        else:
            if attempt < max_retries - 1:
                print(f"⚠️ Không thành công, thử lại sau 2 giây...")
                time.sleep(2)
            else:
                print("❌ Không thể khởi động camera sau nhiều lần thử!")
                
                # Hỏi người dùng có muốn tiếp tục không (chỉ dùng streaming)
                choice = input("🚫 Không thể kết nối camera. Bạn có muốn tiếp tục chỉ với streaming? (y/n): ")
                if choice.lower() != 'y':
                    return
                else:
                    print("⚠️ Chạy ở chế độ không có camera - chỉ streaming")
    
    if camera_manager.is_running:
        print(f"\n🎥 Camera {camera_manager.camera_index} đã khởi động")
    else:
        print(f"\n⚠️ Cảnh báo: Không có camera, chỉ hiển thị streaming")
    
    print("📊 Chế độ: AI Recognition + Live Streaming")
    print("🌐 Stream URL: http://localhost:5000/video_feed")
    print("🎮 Nhấn 'q' để thoát, 's' để chụp ảnh, 'v' để xem điểm danh, 'e' để xem engagement report")
    
    # ==================== THÊM: Improved Face Tracker ====================
    class SimpleFaceTracker:
        def __init__(self, max_disappeared=15):
            self.next_id = 0
            self.objects = {}  # id -> {'bbox': ..., 'last_seen': frame_count}
            self.max_disappeared = max_disappeared
            self.frame_count = 0
        
        def update(self, detected_bboxes):
            """Update tracking với detected bboxes"""
            self.frame_count += 1
            
            # Giảm last_seen cho tất cả objects hiện có
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id]['last_seen'] += 1
                
                # Xóa object nếu không thấy quá lâu
                if self.objects[obj_id]['last_seen'] > self.max_disappeared:
                    del self.objects[obj_id]
            
            # Nếu không có detection, return empty list
            if not detected_bboxes:
                return []
            
            # Gán ID cho các bbox mới
            assigned_ids = []
            
            for bbox in detected_bboxes:
                x, y, w, h = bbox
                center_x = x + w/2
                center_y = y + h/2
                
                # Tìm object gần nhất
                min_distance = float('inf')
                matched_id = None
                
                for obj_id, obj_data in self.objects.items():
                    obj_bbox = obj_data['bbox']
                    obj_center_x = obj_bbox[0] + obj_bbox[2]/2
                    obj_center_y = obj_bbox[1] + obj_bbox[3]/2
                    
                    distance = np.sqrt((center_x - obj_center_x)**2 + (center_y - obj_center_y)**2)
                    
                    # Nếu khoảng cách < 100 pixels và object chưa được gán
                    if distance < 100 and distance < min_distance and obj_id not in assigned_ids:
                        min_distance = distance
                        matched_id = obj_id
                
                if matched_id is not None:
                    # Cập nhật existing object
                    self.objects[matched_id]['bbox'] = bbox
                    self.objects[matched_id]['last_seen'] = 0
                    assigned_ids.append(matched_id)
                else:
                    # Tạo object mới
                    new_id = self.next_id
                    self.objects[new_id] = {
                        'bbox': bbox,
                        'last_seen': 0,
                        'created_at': self.frame_count
                    }
                    assigned_ids.append(new_id)
                    self.next_id += 1
            
            return assigned_ids  # Trả về list ID theo thứ tự detected_bboxes
    
    # ==================== THÊM: Display Stabilizer ====================
    class DisplayStabilizer:
        def __init__(self):
            self.face_display_data = {}
            self.min_display_time = 0.5  # ⚡ GIẢM từ 2.0s xuống 0.5s
            self.last_update_time = {}
            self.smoothing_factor = 0.3  # Thêm smoothing
            
        def get_stable_display(self, face_key, new_data, current_time):
            """Lấy dữ liệu hiển thị SMOOTH"""
            if face_key not in self.face_display_data:
                # Khởi tạo mới
                self.face_display_data[face_key] = {
                    'behavior': new_data.get('behavior', 'normal'),
                    'name': new_data.get('name', 'Unknown'),
                    'emotion': new_data.get('emotion', 'neutral'),
                    'engagement': new_data.get('engagement', 50.0),
                    'bbox_smooth': new_data.get('bbox', {}),  # Thêm bbox smoothing
                    'last_update': current_time
                }
                return self.face_display_data[face_key]
            
            # Áp dụng smoothing cho bbox
            old_bbox = self.face_display_data[face_key].get('bbox_smooth', {})
            new_bbox = new_data.get('bbox', {})
            
            if old_bbox and new_bbox:
                smoothed_bbox = {
                    'x': int(old_bbox.get('x', 0) * 0.7 + new_bbox.get('x', 0) * 0.3),
                    'y': int(old_bbox.get('y', 0) * 0.7 + new_bbox.get('y', 0) * 0.3),
                    'width': int(old_bbox.get('width', 0) * 0.7 + new_bbox.get('width', 0) * 0.3),
                    'height': int(old_bbox.get('height', 0) * 0.7 + new_bbox.get('height', 0) * 0.3)
                }
                self.face_display_data[face_key]['bbox_smooth'] = smoothed_bbox
            
            # Update các thông tin khác với smoothing
            update_threshold = 0.8  # Chỉ update nếu confidence cao
            
            # Behavior: chỉ update nếu behavior mới có confidence cao
            new_behavior = new_data.get('behavior', 'normal')
            old_behavior = self.face_display_data[face_key]['behavior']
            
            if new_behavior != old_behavior:
                behavior_conf = new_data.get('behavior_confidence', 0.5)
                if behavior_conf > update_threshold:
                    self.face_display_data[face_key]['behavior'] = new_behavior
                # Hoặc áp dụng gradual change
                elif random.random() < 0.3:  # 30% chance để update
                    self.face_display_data[face_key]['behavior'] = new_behavior
            
            # Cập nhật timestamp
            self.face_display_data[face_key]['last_update'] = current_time
            
            return self.face_display_data[face_key]
    
    # Khởi tạo trackers và stabilizers
    face_tracker = SimpleFaceTracker(max_disappeared=20)
    display_stabilizer = DisplayStabilizer()
    
    attendance_status = {}
    frame_count = 0
    
    # Biến để đo FPS
    fps_counter = 0
    fps_time = time.time()
    
    # Biến để tracking face IDs qua các frames
    tracked_face_ids = {}
    
    # ==================== FIX: Thêm hàm _match_face_to_behavior_improved ====================
    def _match_face_to_behavior_improved(face_data, behavior_results, face_bboxes_list, face_ids_list):
        """Improved version của _match_face_to_behavior với tracking"""
        face_bbox = face_data['bbox']
        x, y, w, h = face_bbox
        
        # Tìm tracking ID cho face này (nếu có)
        face_id = None
        for idx, bbox in enumerate(face_bboxes_list):
            bx, by, bw, bh = bbox
            # Kiểm tra overlap
            intersection_x1 = max(x, bx)
            intersection_y1 = max(y, by)
            intersection_x2 = min(x + w, bx + bw)
            intersection_y2 = min(y + h, by + bh)
            
            if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                if idx < len(face_ids_list):
                    face_id = face_ids_list[idx]
                    break
        
        if not behavior_results:
            return {'type': 'normal', 'confidence': 0.7}
        
        best_match = {'type': 'normal', 'confidence': 0.7, 'distance': float('inf')}
        
        for behavior in behavior_results:
            if behavior['bbox'] is not None:
                try:
                    bx1, by1, bx2, by2 = behavior['bbox'].astype(int)
                    # Tính trung điểm của bbox
                    face_center_x = x + w/2
                    face_center_y = y + h/2
                    behavior_center_x = (bx1 + bx2) / 2
                    behavior_center_y = (by1 + by2) / 2
                    
                    # Tính khoảng cách Euclid
                    distance = np.sqrt((face_center_x - behavior_center_x)**2 + (face_center_y - behavior_center_y)**2)
                    
                    # Tính IoU (Intersection over Union)
                    intersection_x1 = max(x, bx1)
                    intersection_y1 = max(y, by1)
                    intersection_x2 = min(x + w, bx2)
                    intersection_y2 = min(y + h, by2)
                    
                    if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                        face_area = w * h
                        behavior_area = (bx2 - bx1) * (by2 - by1)
                        union_area = face_area + behavior_area - intersection_area
                        
                        iou = intersection_area / union_area if union_area > 0 else 0
                        
                        # Giảm distance nếu có overlap tốt
                        if iou > 0.3:
                            distance *= 0.3
                        elif iou > 0.1:
                            distance *= 0.7
                    
                    if distance < best_match['distance']:
                        best_match = {
                            'type': behavior['behavior'],
                            'confidence': min(0.9, max(0.7, 1 - distance/300)),
                            'distance': distance,
                            'iou': iou if 'iou' in locals() else 0
                        }
                except Exception as e:
                    continue
        
        return best_match
    
    # Mở cửa sổ preview
    cv2.namedWindow('AI Face Recognition + Streaming Preview', cv2.WINDOW_NORMAL)
    
    while True:
        try:
            # 🔴 ĐỌC FRAME TRỰC TIẾP TỪ CAMERA MANAGER (DÙNG CHUNG)
            frame = camera_manager.read_frame()
            
            if frame is None:
                print("⚠️ Không đọc được frame từ camera")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            fps_counter += 1
            
            # Tính FPS
            current_time = time.time()
            if current_time - fps_time >= 1.0:
                fps = fps_counter / (current_time - fps_time)
                fps_counter = 0
                fps_time = current_time
                fps_text = f"FPS: {fps:.1f}"
            else:
                fps_text = "FPS: calculating..."
            
            # Giảm tần suất detection để tăng performance
            detection_interval = 3
            student_data_list = []
            face_results = []
            behavior_results = []
            
            if frame_count % detection_interval == 0:
                # Phát hiện khuôn mặt
                face_results = system.detect_faces(frame)
                
                # Phát hiện hành vi mỗi 6 frames
                if frame_count % 6 == 0 and hasattr(system.behavior_detector, 'pose_model'):
                    behavior_results = system.behavior_detector.detect_behavior(frame)
                
                # ==================== THÊM: Face Tracking ====================
                # Tạo bboxes từ face_results để tracking
                face_bboxes = []
                for face in face_results:
                    x, y, w, h = face['bbox']
                    face_bboxes.append([x, y, w, h])
                
                # Update face tracking
                face_ids = []
                if face_bboxes:
                    face_ids = face_tracker.update(face_bboxes)
                    # Gán ID cho faces
                    for idx, face in enumerate(face_results):
                        if idx < len(face_ids):
                            face['tracking_id'] = face_ids[idx]
                            tracked_face_ids[face_ids[idx]] = current_time
                
                # ==================== XỬ LÝ AI VÀ ENGAGEMENT ====================
                for i, face_data in enumerate(face_results):
                    bbox = face_data['bbox']
                    x, y, w, h = bbox
                    emotion = face_data['emotion']
                    emotion_conf = face_data['emotion_confidence']
                    
                    if hasattr(system, 'svm_model') and system.svm_model:
                        name, confidence = system.recognize_face(face_data)
                    else:
                        name, confidence = "Unknown", 0.0
                    
                    # Sử dụng hàm matching improved với tracking
                    matched_behavior = _match_face_to_behavior_improved(
                        face_data, 
                        behavior_results,
                        face_bboxes,
                        face_ids
                    )
                    
                    behavior = matched_behavior['type']
                    behavior_confidence = matched_behavior['confidence']
                    
                    # Tính engagement score
                    engagement_result = system.engagement_calculator.calculate_engagement(
                        student_id=f"{name}_{i}",
                        emotion=emotion,
                        emotion_confidence=emotion_conf,
                        behavior=behavior,
                        behavior_confidence=behavior_confidence,
                        bbox=(x, y, w, h)
                    )
                    
                    student_data = {
                        'id': i + 1,
                        'name': name,
                        'emotion': emotion,
                        'emotion_confidence': emotion_conf,
                        'behavior': behavior,
                        'engagement': engagement_result['engagement_score'],
                        'concentration_level': engagement_result['concentration_level'],
                        'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                        'face_confidence': confidence,
                        'tracking_id': face_ids[i] if i < len(face_ids) else i,
                        'engagement_details': engagement_result
                    }
                    
                    student_data_list.append(student_data)
                
                # ==================== GỬI DỮ LIỆU ĐẾN BACKEND ====================
                for student_data in student_data_list:
                    name = student_data['name']
                    emotion = student_data['emotion']
                    emotion_conf = student_data['emotion_confidence']
                    behavior = student_data['behavior']
                    engagement = student_data['engagement']
                    concentration_level = student_data['concentration_level']
                    confidence = student_data['face_confidence']
                    
                    # Gửi dữ liệu điểm danh, cảm xúc, hành vi, độ tập trung
                    if name != "Unknown" and confidence > 0.6:
                        # Tạo unique key với tracking_id
                        tracking_id = student_data.get('tracking_id', hash(str(student_data['bbox'])) % 10000)
                        attendance_key = f"{name}_{tracking_id}"
                        
                        if attendance_key not in attendance_status or frame_count % 30 == 0:
                            system.attendance_system.mark_attendance(
                                name=name,
                                emotion=emotion,
                                emotion_confidence=emotion_conf,
                                behavior=behavior,
                                engagement=engagement,
                                concentration_level=concentration_level,
                                confidence=confidence
                            )
                            attendance_status[attendance_key] = True
                
                # 🔴 CẬP NHẬT KẾT QUẢ CHO STREAMING
                with detection_lock:
                    if student_data_list:
                        last_detection_results = student_data_list.copy()
                        last_detection_time = datetime.now()
            
            # ==================== VẼ OVERLAY AI LÊN FRAME ====================
            overlay_frame = frame.copy()
            
            # Vẽ overlay cho mỗi face
            for i, student_data in enumerate(student_data_list):
                bbox = student_data['bbox']
                x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                
                # Tạo face_key unique
                tracking_id = student_data.get('tracking_id', i)
                face_key = f"face_{tracking_id}"
                
                # Lấy dữ liệu hiển thị đã được stabilized
                display_data = display_stabilizer.get_stable_display(
                    face_key, 
                    student_data, 
                    current_time
                )
                
                # Lấy thông tin từ display_data đã stabilized
                name = display_data['name']
                emotion = display_data['emotion']
                behavior = display_data['behavior']
                engagement = display_data['engagement']
                
                # Lấy confidence và concentration_level từ student_data gốc
                confidence = student_data.get('face_confidence', 0.5)
                concentration_level = student_data.get('concentration_level', 'medium')
                emotion_conf = student_data.get('emotion_confidence', 0.5)
                
                # Màu sắc
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                
                # Vẽ bounding box
                cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), color, 2)
                
                # ==================== HIỂN THỊ THÔNG TIN ====================
                # Dòng 1: Tên và confidence
                info_text = f"{name} ({confidence:.2f})"
                cv2.putText(overlay_frame, info_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Dòng 2: Hành vi (ĐÃ ĐƯỢC STABILIZED)
                behavior_display = f"{behavior}"
                # Màu cho behavior
                behavior_color = (255, 255, 0)  # Vàng mặc định
                
                if 'raising' in behavior:
                    behavior_color = (0, 255, 255)  # Vàng đậm cho giơ tay
                elif 'writing' in behavior:
                    behavior_color = (255, 255, 0)  # Vàng sáng cho viết
                elif 'look_around' in behavior:
                    behavior_color = (0, 165, 255)  # Cam cho nhìn quanh
                
                cv2.putText(overlay_frame, behavior_display, (x, y - 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, behavior_color, 1)
                
                # Dòng 3: Cảm xúc
                emotion_text = f"{emotion} ({emotion_conf:.1f})"
                emotion_color = (0, 255, 255)  # Vàng cho cảm xúc
                cv2.putText(overlay_frame, emotion_text, (x, y + h + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_color, 1)
                
                # Dòng 4: Engagement score
                engagement_text = f"Engagement: {engagement:.0f} ({concentration_level})"
                # Màu theo engagement level
                engagement_color = system._get_engagement_color(engagement)
                cv2.putText(overlay_frame, engagement_text, (x, y + h + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, engagement_color, 1)
                
                # ==================== VẼ ENGAGEMENT BAR ====================
                bar_width = 100
                bar_height = 8
                bar_x = x
                bar_y = y + h + 60
                
                # Tính filled width dựa trên engagement (0-100)
                filled_width = int(bar_width * engagement / 100)
                
                # Vẽ thanh nền
                cv2.rectangle(overlay_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                             (100, 100, 100), -1)
                
                # Vẽ thanh giá trị với gradient color
                engagement_color = system._get_engagement_color(engagement)
                cv2.rectangle(overlay_frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                             engagement_color, -1)
                
                # Vẽ viền cho thanh
                cv2.rectangle(overlay_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                             (200, 200, 200), 1)
                
                # Hiển thị tracking ID (nhỏ, để debug)
                tracking_text = f"ID: {tracking_id}"
                cv2.putText(overlay_frame, tracking_text, (x + w - 40, y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            # ==================== HIỂN THỊ STATUS BAR ====================
            backend_status = "🟢 REAL-TIME" if system.backend_sender.is_connected else "🔴 OFFLINE"
            device_status = "⚡ GPU" if gpu_available else "💻 CPU"
            
            # Status bar chính
            info_text = f"Camera {camera_manager.camera_index} | Faces: {len(face_results)} | Backend: {backend_status} | Device: {device_status} | {fps_text}"
            cv2.putText(overlay_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Dòng thông tin phụ
            try:
                # Tính engagement trung bình
                if student_data_list and len(student_data_list) > 0:
                    avg_engagement = np.mean([s.get('engagement', 50) for s in student_data_list])
                    total_students = len(student_data_list)
                    
                    # Đếm số học sinh tập trung
                    engaged_count = sum(1 for s in student_data_list if s.get('engagement', 0) >= 70)
                    distracted_count = sum(1 for s in student_data_list if s.get('engagement', 0) < 50)
                    
                    engagement_summary = f"Students: {total_students} | Avg Eng: {avg_engagement:.1f} | Focused: {engaged_count} | Distracted: {distracted_count}"
                    cv2.putText(overlay_frame, engagement_summary, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Hiển thị tracking info
                    tracking_info = f"Tracked Faces: {len(tracked_face_ids)} | Frame: {frame_count}"
                    cv2.putText(overlay_frame, tracking_info, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            except Exception as e:
                # Bỏ qua lỗi nếu có vấn đề với dữ liệu
                pass
            
            # ==================== HIỂN THỊ PREVIEW ====================
            cv2.imshow('AI Face Recognition + Streaming Preview', overlay_frame)
            
            # ==================== XỬ LÝ PHÍM ====================
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Stopping recognition and streaming...")
                break
            elif key == ord('s'):
                # Lưu ảnh chụp màn hình
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, overlay_frame)
                print(f"✅ Đã lưu ảnh: {filename}")
                
                # Lưu thêm file txt với thông tin detection
                info_filename = f"capture_{timestamp}_info.txt"
                with open(info_filename, 'w') as f:
                    f.write(f"Capture time: {datetime.now().isoformat()}\n")
                    f.write(f"FPS: {fps if 'fps' in locals() else 0}\n")
                    f.write(f"Faces detected: {len(student_data_list)}\n")
                    f.write(f"Frame count: {frame_count}\n")
                    f.write("\nDetected faces:\n")
                    for i, student in enumerate(student_data_list):
                        f.write(f"\nFace {i+1}:\n")
                        f.write(f"  Name: {student.get('name', 'Unknown')}\n")
                        f.write(f"  Emotion: {student.get('emotion', 'neutral')}\n")
                        f.write(f"  Behavior: {student.get('behavior', 'normal')}\n")
                        f.write(f"  Engagement: {student.get('engagement', 0):.1f}\n")
                        f.write(f"  Confidence: {student.get('face_confidence', 0):.2f}\n")
                print(f"✅ Đã lưu thông tin: {info_filename}")
                
            elif key == ord('v'):
                # Xem attendance
                print("\n" + "="*80)
                print("📋 ATTENDANCE RECORDS")
                print("="*80)
                system.attendance_system.view_attendance()
                print("="*80)
                
            elif key == ord('e'):
                # Xem engagement report
                report = system.get_class_engagement_report()
                if report:
                    print("\n" + "="*80)
                    print("📊 ENGAGEMENT REPORT")
                    print("="*80)
                    print(f"Total Students: {report['total_students']}")
                    print(f"Average Engagement: {report['average_engagement']}")
                    print(f"Concentration Distribution:")
                    for level, count in report['concentration_distribution'].items():
                        percentage = (count / report['total_students'] * 100) if report['total_students'] > 0 else 0
                        print(f"  {level}: {count} students ({percentage:.1f}%)")
                    
                    print("\nTop 5 Students:")
                    sorted_students = sorted(report['students'], 
                                            key=lambda x: x['engagement'], 
                                            reverse=True)[:5]
                    for i, student in enumerate(sorted_students):
                        print(f"{i+1}. {student['name']}: {student['engagement']} ({student['concentration_level']})")
                    print("="*80)
                else:
                    print("📭 No engagement data available")
                    
            elif key == ord('d'):
                # Debug info
                print("\n" + "="*80)
                print("🐛 DEBUG INFORMATION")
                print("="*80)
                print(f"Frame count: {frame_count}")
                print(f"Current FPS: {fps if 'fps' in locals() else 'calculating...'}")
                print(f"Face results: {len(face_results)}")
                print(f"Behavior results: {len(behavior_results)}")
                print(f"Student data: {len(student_data_list)}")
                print(f"Tracked faces: {len(tracked_face_ids)}")
                print(f"Display stabilizer: {len(display_stabilizer.face_display_data)} entries")
                
                # Hiển thị tracking info
                if tracked_face_ids:
                    print(f"\nTracked IDs (last {min(10, len(tracked_face_ids))}):")
                    sorted_ids = sorted(tracked_face_ids.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)[:10]
                    for face_id, last_seen in sorted_ids:
                        age = current_time - last_seen
                        print(f"  ID {face_id}: last seen {age:.1f}s ago")
                print("="*80)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"⚠️ Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)
    
    # Cleanup
    camera_manager.stop()
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "="*80)
    print("📊 RECOGNITION SESSION SUMMARY")
    print("="*80)
    print(f"Total frames processed: {frame_count}")
    print(f"Device used: {'GPU' if gpu_available else 'CPU'}")
    print(f"Backend connection: {'Connected' if system.backend_sender.is_connected else 'Disconnected'}")
    
    if hasattr(system, 'attendance_system'):
        try:
            df = pd.read_csv(system.attendance_system.csv_file)
            print(f"Attendance records: {len(df)} entries")
        except:
            print(f"Attendance records: Unknown")
    
    print("👋 Session ended!")
    print("="*80)
# ==================== MAIN MENU ====================
def main_menu():
    """Hiển thị menu chính"""
    # Kiểm tra hệ thống chi tiết
    check_system_capabilities()
    
    while True:
        print("\n" + "="*80)
        print("🎭 COMPLETE RECOGNITION SYSTEM - FACE + EMOTION + BEHAVIOR + ENGAGEMENT + ATTENDANCE")
        print("="*80)
        print("1. 📁 Tạo cấu trúc thư mục")
        print("2. 🎯 Train face recognition model")
        print("3. 🎥 Real-time (Face + Emotion + Behavior + Engagement + Attendance + Backend)")
        print("4. 📊 Xem lịch sử điểm danh")
        print("5. 🔗 Kiểm tra kết nối backend")
        print("6. 🔧 Khắc phục sự cố GPU")
        print("7. 🌐 Start Flask API Server (for web control)")
        print("8. 🚪 Thoát")
        print("="*80)
        print("📊 Hệ thống tính engagement dựa trên cảm xúc và hành vi:")
        print("   - Cảm xúc: happy(0.85), neutral(0.7), sad(0.4), angry(0.3)")
        print("   - Hành vi: writing(0.9), look_straight(0.8), raising_hand(0.75)")
        print("   - Kết quả: 0-100 điểm, 5 mức độ tập trung")
        print("="*80)
        
        choice = input("👉 Chọn chức năng (1-8): ").strip()
        
        if choice == "1":
            create_folder_structure()
        elif choice == "2":
            train_model()
        elif choice == "3":
            real_time_recognition()
        elif choice == "4":
            view_attendance()
        elif choice == "5":
            test_backend_connection()
        elif choice == "6":
            troubleshoot_gpu()
        elif choice == "7":
            start_flask_server()
        elif choice == "8":
            print("👋 Tạm biệt!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ!")
        
        if choice != "7":  # Không cần nhấn Enter nếu đang chạy Flask
            input("\n👉 Nhấn Enter để tiếp tục...")

# ==================== MAIN ====================
if __name__ == "__main__":
    print("🔧 Đang kiểm tra hệ thống...")
    install_dependencies()
    
    print("\n" + "="*80)
    print("🎯 COMPLETE RECOGNITION SYSTEM WITH ENGAGEMENT SCORING")
    print("="*80)
    print("📊 Tính năng:")
    print("   • Nhận diện khuôn mặt (InsightFace)")
    print("   • Nhận diện cảm xúc (DeepFace)")
    print("   • Nhận diện hành vi (YOLOv8-Pose)")
    print("   • Tính điểm tập trung (Engagement Score):")
    print("     📈 Dựa trên cảm xúc + hành vi")
    print("     🎯 0-100 điểm, 5 mức độ tập trung")
    print("     ⚖️ Trọng số khoa học cho từng yếu tố")
    print("   • Điểm danh tự động")
    print("   • Backend integration - Gửi toàn bộ dữ liệu:")
    print("     📋 Điểm danh (attendance)")
    print("     😊 Cảm xúc (emotion)")
    print("     🎯 Hành vi (behavior)")
    print("     📊 Độ tập trung (engagement)")
    print("   • 🌐 Flask API Server (port 5000)")
    print("="*80)
    print("🚀 Chạy option 7 để khởi động Flask API Server")
    print("🌐 Web frontend có thể gọi API tại: http://localhost:5000")
    print("📊 Engagement API: /api/engagement - Lấy báo cáo tập trung lớp học")
    print("📊 Backend: http://localhost:8000")
    print("="*80)
    
    main_menu()