'use client'
import { useRouter } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import { buildApiUrl } from '../config/api';

// Utility function để safe access localStorage - ĐÃ SỬA
const safeLocalStorage = {
    setItem: (key, value) => {
        // Kiểm tra kỹ hơn
        if (typeof window === 'undefined') return;
        try {
            localStorage.setItem(key, value);
        } catch (e) {
            console.warn('localStorage setItem error:', e);
        }
    },

    getItem: (key) => {
        if (typeof window === 'undefined') return null;
        try {
            return localStorage.getItem(key);
        } catch (e) {
            console.warn('localStorage getItem error:', e);
            return null;
        }
    },

    removeItem: (key) => {
        if (typeof window === 'undefined') return;
        try {
            localStorage.removeItem(key);
        } catch (e) {
            console.warn('localStorage removeItem error:', e);
        }
    }
};

export default function WelcomePage() {
    const router = useRouter();
    const [student, setStudent] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [connectionStatus, setConnectionStatus] = useState('checking');
    const [debugInfo, setDebugInfo] = useState('Đang kiểm tra kết nối AI...');
    const [allStudents, setAllStudents] = useState([]); // Thêm state để lưu tất cả học sinh
    const pollIntervalRef = useRef(null);
    const lastDetectionRef = useRef('');
    const detectionCountRef = useRef(0);
    const [mounted, setMounted] = useState(false); // THÊM STATE NÀY
    const [lastKnownStudent, setLastKnownStudent] = useState(null); // Lưu học sinh đã biết cuối cùng

    // THÊM: State cho feedback
    const [showFeedbackModal, setShowFeedbackModal] = useState(false);
    const [feedbackMode, setFeedbackMode] = useState(''); // 'text' hoặc 'voice'
    const [feedbackText, setFeedbackText] = useState('');
    const [isRecording, setIsRecording] = useState(false);
    const [recordedAudio, setRecordedAudio] = useState(null);
    const [audioRecorder, setAudioRecorder] = useState(null);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);
    const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);

    // THÊM: State cho popup success
    const [showSuccessPopup, setShowSuccessPopup] = useState(false);
    const [successMessage, setSuccessMessage] = useState('');
    const [voiceRetryCount, setVoiceRetryCount] = useState(0); // Đếm số lần thử lại khi gửi voice feedback lỗi

    // THÊM: Effect để set mounted state
    useEffect(() => {
        setMounted(true);
        return () => setMounted(false);
    }, []);

    // Hàm kiểm tra tên có phải "Unknown" không
    const isUnknownStudent = (studentName) => {
        if (!studentName) return true;

        const nameLower = studentName.toLowerCase().trim();
        const unknownKeywords = [
            'unknown',
            'unknow', // Trường hợp lỗi chính tả
            'không rõ',
            'chưa biết',
            'unknown student',
            'student'
        ];

        return unknownKeywords.some(keyword => nameLower.includes(keyword));
    };

    // Hàm lấy dữ liệu từ AI server (port 5000) - ĐÃ SỬA
    const fetchAIDetection = async () => {
        try {
            // Gọi API mới từ AI server (port 5000)
            const response = await fetch('http://localhost:5000/api/latest_results', {
                cache: 'no-store',
                headers: {
                    'Accept': 'application/json',
                    'Cache-Control': 'no-cache'
                },
                signal: AbortSignal.timeout(3000) // Timeout sau 3 giây
            });

            if (!response.ok) throw new Error(`AI API error: ${response.status}`);

            const data = await response.json();
            detectionCountRef.current++;

            console.log('📡 AI Detection data:', data);

            if (data.status === 'success' && data.results && data.results.length > 0) {
                // Lọc bỏ các học sinh "Unknown"
                const knownStudents = data.results.filter(s => !isUnknownStudent(s.name));
                setAllStudents(knownStudents);

                if (knownStudents.length > 0) {
                    // Chọn học sinh đầu tiên KHÔNG PHẢI "Unknown"
                    const latestKnownStudent = knownStudents[0];
                    const detectedStudent = {
                        name: latestKnownStudent.name,
                        id: latestKnownStudent.id || `face_${latestKnownStudent.bbox?.x}_${latestKnownStudent.bbox?.y}`,
                        class: 'AI Nhận Diện',
                        emotion: latestKnownStudent.emotion || 'neutral',
                        confidence: latestKnownStudent.face_confidence || latestKnownStudent.confidence || 0.5,
                        emotion_confidence: latestKnownStudent.emotion_confidence || 0.5,
                        behavior: latestKnownStudent.behavior || 'normal',
                        engagement: latestKnownStudent.engagement || 75.0,
                        concentration_level: latestKnownStudent.concentration_level || 'medium',
                        timestamp: data.last_update || new Date().toISOString(),
                        source: 'ai_detection',
                        bbox: latestKnownStudent.bbox,
                        face_confidence: latestKnownStudent.face_confidence
                    };

                    // Kiểm tra xem có phải detection mới không
                    const currentKey = `${detectedStudent.name}_${detectedStudent.emotion}_${detectedStudent.behavior}`;
                    if (currentKey !== lastDetectionRef.current) {
                        setStudent(detectedStudent);
                        setLastKnownStudent(detectedStudent); // Lưu học sinh đã biết
                        setConnectionStatus('connected');
                        setDebugInfo(`Nhận diện: ${detectedStudent.name} - ${detectedStudent.emotion} - ${detectedStudent.behavior}`);

                        // Lưu vào localStorage
                        safeLocalStorage.setItem('detectedStudent', JSON.stringify(detectedStudent));

                        // Hiển thị notification
                        showNotification(`Xin chào ${detectedStudent.name}!`);

                        lastDetectionRef.current = currentKey;
                    }
                } else {
                    // Có detection nhưng tất cả đều là "Unknown"
                    setConnectionStatus('no_known_students');
                    setDebugInfo(`Phát hiện ${data.results.length} khuôn mặt nhưng chưa nhận diện được`);

                    // Giữ lại học sinh đã biết cuối cùng nếu có
                    if (!student && lastKnownStudent) {
                        setStudent(lastKnownStudent);
                        setDebugInfo(`Hiển thị học sinh đã biết: ${lastKnownStudent.name}`);
                    } else if (student && isUnknownStudent(student.name)) {
                        // Nếu student hiện tại là "Unknown", xóa nó
                        setStudent(null);
                        safeLocalStorage.removeItem('detectedStudent');
                    }
                }
            } else if (data.status === 'no_data') {
                // Không có dữ liệu detection
                setConnectionStatus('no_detection');
                setDebugInfo('AI đang chạy nhưng chưa phát hiện học sinh');

                // Giữ lại học sinh đã biết cuối cùng nếu có
                if (!student && lastKnownStudent) {
                    setStudent(lastKnownStudent);
                    setDebugInfo(`Hiển thị học sinh đã biết: ${lastKnownStudent.name}`);
                }
            } else {
                setConnectionStatus('no_data');
                setDebugInfo('Không có dữ liệu từ AI');
            }
        } catch (error) {
            console.error('❌ Error fetching AI detection:', error);
            setConnectionStatus('error');
            setDebugInfo(`Lỗi: ${error.message}`);

            // Khi có lỗi, vẫn hiển thị học sinh đã biết nếu có
            if (!student && lastKnownStudent) {
                setStudent(lastKnownStudent);
                setDebugInfo(`Hiển thị học sinh đã biết (lỗi kết nối): ${lastKnownStudent.name}`);
            }
        }
    };

    // Kiểm tra kết nối đến AI server (port 5000) - ĐÃ SỬA
    const checkAIConnection = async () => {
        try {
            console.log('🔗 Checking AI server connection...');
            const response = await fetch('http://localhost:5000/api/health', {
                signal: AbortSignal.timeout(3000)
            });

            if (response.ok) {
                const healthData = await response.json();
                console.log('✅ AI Server health:', healthData);
                setConnectionStatus('ready');
                return true;
            }
            return false;
        } catch (error) {
            console.warn('⚠️ AI Server not responding:', error.message);
            setConnectionStatus('offline');
            return false;
        }
    };

    // Hàm hiển thị notification
    const showNotification = (message) => {
        // KIỂM TRA mounted và window
        if (!mounted || typeof window === 'undefined' || !("Notification" in window)) return;

        if (Notification.permission === "granted") {
            new Notification("AI Recognition", {
                body: message,
                icon: "/favicon.ico",
                silent: true
            });
        }
    };

    // Hàm request notification permission
    const requestNotificationPermission = () => {
        if (!mounted || typeof window === 'undefined' || !("Notification" in window)) return;

        if (Notification.permission === "default") {
            Notification.requestPermission().then(permission => {
                console.log('Notification permission:', permission);
            });
        }
    };

    // Hàm kiểm tra và khởi động AI nếu cần
    const ensureAIRunning = async () => {
        try {
            // Kiểm tra trạng thái AI
            const statusResponse = await fetch('http://localhost:5000/api/status');
            if (statusResponse.ok) {
                const statusData = await statusResponse.json();

                // Nếu AI chưa chạy, khởi động nó
                if (statusData.status === 'stopped' || !statusData.ai_system_initialized) {
                    console.log('🚀 Starting AI system...');
                    const startResponse = await fetch('http://localhost:5000/api/start_ai', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                    });

                    if (startResponse.ok) {
                        console.log('✅ AI system started successfully');
                        return true;
                    }
                } else {
                    console.log('✅ AI system is already running');
                    return true;
                }
            }
        } catch (error) {
            console.warn('Cannot check/start AI:', error);
        }
        return false;
    };

    // THÊM: Khởi tạo audio recorder
    const initializeAudioRecorder = async () => {
        try {
            if (typeof window === 'undefined') return;

            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const recorder = new MediaRecorder(stream);

            recorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            recorder.onstop = () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                setRecordedAudio({
                    blob: audioBlob,
                    url: audioUrl,
                    timestamp: new Date().toISOString()
                });
                audioChunksRef.current = [];
            };

            mediaRecorderRef.current = recorder;
            console.log('🎤 Audio recorder initialized');
        } catch (error) {
            console.error('Error initializing audio recorder:', error);
            alert('Không thể truy cập microphone. Vui lòng kiểm tra quyền truy cập.');
        }
    };

    // THÊM: Bắt đầu ghi âm
    const startRecording = () => {
        if (!mediaRecorderRef.current) {
            alert('Audio recorder chưa được khởi tạo');
            return;
        }

        audioChunksRef.current = [];
        mediaRecorderRef.current.start();
        setIsRecording(true);
        console.log('🎤 Bắt đầu ghi âm...');
    };

    // THÊM: Dừng ghi âm
    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            console.log('⏹️ Dừng ghi âm');
        }
    };

    // THÊM: Mở modal feedback
    const openFeedbackModal = (mode) => {
        setFeedbackMode(mode);
        setShowFeedbackModal(true);
        setFeedbackText('');
        setRecordedAudio(null);
        setVoiceRetryCount(0); // Reset retry count khi mở modal mới

        if (mode === 'voice') {
            initializeAudioRecorder();
        }
    };

    // THÊM: Đóng modal feedback và reset
    const closeFeedbackModal = () => {
        if (isRecording) {
            stopRecording();
        }
        setShowFeedbackModal(false);
        setFeedbackMode('');
        setFeedbackText('');
        setRecordedAudio(null);
        setVoiceRetryCount(0);
    };

    // THÊM: Hiển thị popup success
    const showSuccessPopupMessage = (message) => {
        setSuccessMessage(message);
        setShowSuccessPopup(true);

        // Tự động ẩn popup sau 3 giây
        setTimeout(() => {
            setShowSuccessPopup(false);
            setSuccessMessage('');

            // Tự động load lại phần nhập feedback
            if (showFeedbackModal) {
                setFeedbackText('');
                setRecordedAudio(null);
                if (feedbackMode === 'voice') {
                    setVoiceRetryCount(0);
                    initializeAudioRecorder();
                }
            }
        }, 3000);
    };

    // THÊM: Xử lý yêu cầu nói lại khi gửi voice feedback lỗi
    const requestRetryVoice = () => {
        setVoiceRetryCount(prev => prev + 1);
        setRecordedAudio(null);

        // Hiển thị thông báo yêu cầu nói lại
        alert(`Gửi feedback thất bại. Vui lòng nói lại lần ${voiceRetryCount + 1}.`);

        // Tự động bắt đầu ghi âm lại nếu đang ở voice mode
        if (feedbackMode === 'voice' && mediaRecorderRef.current) {
            setTimeout(() => {
                startRecording();
            }, 500);
        }
    };

    // THÊM: Gửi feedback
    const submitFeedback = async () => {
        if (!student) {
            alert('Không tìm thấy thông tin học sinh');
            return;
        }

        setIsSubmittingFeedback(true);

        try {
            let apiEndpoint = '';
            let payload = {};

            if (feedbackMode === 'text' && feedbackText.trim()) {
                // Text feedback
                apiEndpoint = buildApiUrl('/api/feedback/text');
                payload = {
                    student_id: student.id,
                    student_name: student.name,
                    feedback_text: feedbackText.trim(),
                    feedback_type: 'text',
                    emotion: student.emotion,
                    class_name: student.class || 'AI Class',
                    session_id: `FB_${Date.now()}`
                };
            } else if (feedbackMode === 'voice' && recordedAudio) {
                // Voice feedback - convert audio to base64
                const audioBase64 = await blobToBase64(recordedAudio.blob);
                // Đảm bảo blob có đúng type
                if (!recordedAudio.blob.type) {
                    // Set default type
                    recordedAudio.blob = new Blob([recordedAudio.blob], { type: 'audio/wav' });
                }

                apiEndpoint = buildApiUrl('/api/feedback/voice');
                payload = {
                    student_id: student.id,
                    student_name: student.name,
                    audio_base64: audioBase64,
                    audio_format: 'wav',
                    feedback_type: 'voice',
                    class_name: student.class || 'AI Class',
                    session_id: `FB_VOICE_${Date.now()}`
                };
                console.log('Sending voice feedback:', {
                    student_name: student.name,
                    audio_size: recordedAudio.blob.size,
                    has_audio: !!audioBase64
                });
            } else {
                alert('Vui lòng nhập feedback hoặc ghi âm trước khi gửi');
                setIsSubmittingFeedback(false);
                return;
            }

            // Gửi đến database server
            const response = await fetch(apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            if (response.ok) {
                const result = await response.json();

                // Lưu vào localStorage (backup)
                const existingFeedbacks = JSON.parse(localStorage.getItem('studentFeedbacks') || '[]');
                existingFeedbacks.push({
                    ...payload,
                    timestamp: new Date().toISOString(),
                    server_response: result
                });
                localStorage.setItem('studentFeedbacks', JSON.stringify(existingFeedbacks));

                // Hiển thị popup success
                const successMsg = feedbackMode === 'voice'
                    ? `✅ Feedback đã được gửi thành công!\nĐã chuyển đổi thành text: ${result.transcribed_text?.substring(0, 100)}...`
                    : '✅ Feedback đã được gửi thành công!';

                showSuccessPopupMessage(successMsg);

                // Đóng modal sau khi hiển thị success
                setTimeout(() => {
                    closeFeedbackModal();
                }, 500);
            } else {
                const errorText = await response.text();

                // Xử lý riêng cho voice feedback lỗi
                if (feedbackMode === 'voice') {
                    // Kiểm tra số lần đã thử lại
                    if (voiceRetryCount < 2) { // Cho phép thử lại tối đa 2 lần
                        setIsSubmittingFeedback(false);
                        requestRetryVoice();
                        return;
                    } else {
                        // Sau 3 lần thử vẫn lỗi, thông báo và lưu cục bộ
                        throw new Error(`Không thể gửi feedback sau ${voiceRetryCount + 1} lần thử. Đã lưu cục bộ.`);
                    }
                } else {
                    // Text feedback lỗi
                    throw new Error(`Server error: ${response.status} - ${errorText}`);
                }
            }
        } catch (error) {
            console.error('Error submitting feedback:', error);

            // Fallback: lưu cục bộ
            const localBackup = {
                student_id: student.id,
                student_name: student.name,
                feedback_text: feedbackMode === 'text' ? feedbackText : '[Voice feedback]',
                type: feedbackMode,
                timestamp: new Date().toISOString(),
                audio_data: feedbackMode === 'voice' ? 'base64_audio_data' : null,
                error: error.message
            };

            const existing = JSON.parse(localStorage.getItem('feedback_backup') || '[]');
            existing.push(localBackup);
            localStorage.setItem('feedback_backup', JSON.stringify(existing));

            // Hiển thị popup success cho fallback
            showSuccessPopupMessage('✅ Feedback đã được lưu cục bộ!');

            // Đóng modal sau khi hiển thị success
            setTimeout(() => {
                closeFeedbackModal();
            }, 500);
        } finally {
            setIsSubmittingFeedback(false);
        }
    };

    // Utility: Convert Blob to Base64
    const blobToBase64 = (blob) => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                resolve(reader.result);
            };
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    };

    useEffect(() => {
        console.log('🚀 WelcomePage component mounted');

        // Khởi tạo
        const initialize = async () => {
            setIsLoading(true);

            // Request notification permission - CHỈ KHI MOUNTED
            if (mounted) {
                requestNotificationPermission();
            }

            // Load từ localStorage nếu có - CHỈ KHI MOUNTED
            if (mounted) {
                const storedStudent = safeLocalStorage.getItem('detectedStudent');
                if (storedStudent) {
                    try {
                        const parsedStudent = JSON.parse(storedStudent);

                        // Kiểm tra xem có phải "Unknown" không
                        if (!isUnknownStudent(parsedStudent.name)) {
                            console.log('📁 Loaded from storage:', parsedStudent.name);
                            setStudent(parsedStudent);
                            setLastKnownStudent(parsedStudent); // Lưu học sinh đã biết
                            setDebugInfo(`Chào lại ${parsedStudent.name}!`);
                        } else {
                            console.log('🚫 Ignored Unknown student from storage');
                            safeLocalStorage.removeItem('detectedStudent');
                        }
                    } catch (e) {
                        console.error('Error parsing stored student:', e);
                    }
                }
            }

            // Kiểm tra kết nối AI server
            const isConnected = await checkAIConnection();

            if (isConnected) {
                // Đảm bảo AI đang chạy
                await ensureAIRunning();

                // Lấy dữ liệu ngay lần đầu
                await fetchAIDetection();

                // Bắt đầu polling (mỗi 2 giây)
                pollIntervalRef.current = setInterval(fetchAIDetection, 2000);
            } else {
                // Thử lại sau 5 giây
                setTimeout(async () => {
                    const retryConnected = await checkAIConnection();
                    if (retryConnected) {
                        await ensureAIRunning();
                        await fetchAIDetection();
                        pollIntervalRef.current = setInterval(fetchAIDetection, 2000);
                    }
                }, 5000);
            }

            setIsLoading(false);
        };

        // Chỉ initialize khi đã mount
        if (mounted) {
            initialize();
        }

        // Cleanup function
        return () => {
            console.log('🧹 WelcomePage cleanup');

            // Clear interval
            if (pollIntervalRef.current) {
                clearInterval(pollIntervalRef.current);
                pollIntervalRef.current = null;
            }

            // Dừng ghi âm nếu đang ghi
            if (isRecording && mediaRecorderRef.current) {
                mediaRecorderRef.current.stop();
            }
        };
    }, [mounted]); // THÊM mounted vào dependencies

    const handleReturn = () => {
        router.push('/');
    };

    const getEmoji = (emotion) => {
        const emojis = {
            'happy': '😊',
            'neutral': '😐',
            'sad': '😢',
            'surprised': '😲',
            'angry': '😠',
            'fearful': '😨',
            'disgusted': '🤢',
            'vui vẻ': '😊',
            'bình thường': '😐',
            'buồn bã': '😢',
            'ngạc nhiên': '😲',
            'tức giận': '😠',
            'sợ hãi': '😨',
            'kinh tởm': '🤢'
        };
        return emojis[emotion?.toLowerCase()] || '👤';
    };

    const getBehaviorEmoji = (behavior) => {
        const behaviorEmojis = {
            'writing': '✍️',
            'raising_one_hand': '✋',
            'raising_two_hands': '🙌',
            'look_straight': '👀',
            'look_around': '👁️',
            'normal': '💭',
            'unknown': '❓'
        };
        return behaviorEmojis[behavior] || '💭';
    };

    const getEngagementColor = (score) => {
        if (score >= 80) return 'text-green-400';
        if (score >= 70) return 'text-green-300';
        if (score >= 60) return 'text-yellow-300';
        if (score >= 50) return 'text-orange-400';
        return 'text-red-400';
    };

    const getConnectionStatusColor = () => {
        switch (connectionStatus) {
            case 'connected': return 'bg-green-500';
            case 'ready': return 'bg-blue-500';
            case 'checking': return 'bg-yellow-500';
            case 'no_known_students': return 'bg-yellow-400'; // Màu mới cho trường hợp chỉ có Unknown
            case 'no_detection': return 'bg-yellow-300';
            case 'no_data': return 'bg-gray-400';
            case 'server_error': return 'bg-orange-500';
            case 'error': return 'bg-red-500';
            case 'offline': return 'bg-red-700';
            default: return 'bg-gray-500';
        }
    };

    const getConnectionStatusText = () => {
        switch (connectionStatus) {
            case 'connected': return 'AI Đã nhận diện';
            case 'ready': return 'AI Sẵn sàng';
            case 'checking': return 'Đang kiểm tra...';
            case 'no_known_students': return 'Chỉ phát hiện Unknown';
            case 'no_detection': return 'Chưa phát hiện';
            case 'no_data': return 'Chưa có dữ liệu';
            case 'server_error': return 'Lỗi server';
            case 'error': return 'Lỗi kết nối';
            case 'offline': return 'AI offline';
            default: return 'Không xác định';
        }
    };

    const forceRefresh = () => {
        setDebugInfo('Đang refresh thủ công...');
        fetchAIDetection();
    };

    // THÊM: Popup Success Component
    const renderSuccessPopup = () => {
        if (!showSuccessPopup) return null;

        return (
            <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 backdrop-blur-sm">
                <div className="bg-gradient-to-br from-green-600 to-emerald-700 rounded-2xl p-8 max-w-md w-full mx-4 border border-white/30 shadow-2xl animate-fade-in">
                    <div className="flex flex-col items-center justify-center space-y-4">
                        <div className="text-6xl animate-bounce">✅</div>
                        <h3 className="text-2xl font-bold text-white text-center">
                            Thành công!
                        </h3>
                        <div className="text-white/90 text-center whitespace-pre-line">
                            {successMessage}
                        </div>
                        <div className="text-sm text-white/70 text-center mt-2">
                            Popup sẽ tự đóng sau 3 giây...
                        </div>
                    </div>
                </div>
            </div>
        );
    };

    // Hiển thị tất cả học sinh được phát hiện (KHÔNG BAO GỒM Unknown)
    const renderAllStudents = () => {
        if (!allStudents || allStudents.length === 0) return null;

        return (
            <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                <h3 className="col-span-full text-xl font-semibold text-white mb-2">
                    📊 Học sinh đã nhận diện ({allStudents.length})
                </h3>
                {allStudents.map((s, index) => (
                    <div
                        key={`${s.name}_${index}`}
                        className="bg-black/30 backdrop-blur-sm rounded-xl p-4 border border-white/10"
                    >
                        <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-3">
                                <div className="text-2xl">{getEmoji(s.emotion)}</div>
                                <div>
                                    <div className="font-medium text-white">{s.name}</div>
                                    <div className="text-sm text-gray-300">
                                        {s.behavior} {getBehaviorEmoji(s.behavior)}
                                    </div>
                                </div>
                            </div>
                            <div className={`text-lg font-bold ${getEngagementColor(s.engagement)}`}>
                                {s.engagement?.toFixed(1) || '?'}
                            </div>
                        </div>
                        <div className="mt-2 text-xs text-gray-400">
                            {s.concentration_level} • {s.face_confidence?.toFixed(2) || '0.00'} confidence
                        </div>
                    </div>
                ))}
            </div>
        );
    };

    // THÊM: Modal Feedback
    const renderFeedbackModal = () => {
        if (!showFeedbackModal) return null;

        return (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
                <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-6 md:p-8 max-w-md w-full mx-4 border border-white/20 shadow-2xl">
                    <div className="flex justify-between items-center mb-6">
                        <h3 className="text-2xl font-bold text-white">
                            {feedbackMode === 'text' ? '📝 Gửi Phản Hồi' : '🎤 Ghi Âm Phản Hồi'}
                        </h3>
                        <button
                            onClick={closeFeedbackModal}
                            className="text-gray-400 hover:text-white text-2xl"
                            disabled={isSubmittingFeedback}
                        >
                            ×
                        </button>
                    </div>

                    {feedbackMode === 'text' ? (
                        <div className="space-y-4">
                            <div className="text-gray-300 mb-4">
                                Xin chào <span className="text-yellow-300 font-semibold">{student?.name}</span>!
                                Hãy chia sẻ phản hồi của bạn về buổi học:
                            </div>
                            <textarea
                                value={feedbackText}
                                onChange={(e) => setFeedbackText(e.target.value)}
                                placeholder="Nhập phản hồi của bạn tại đây..."
                                className="w-full h-40 bg-black/40 text-white rounded-xl p-4 border border-white/20 focus:border-blue-400 focus:outline-none resize-none"
                                disabled={isSubmittingFeedback}
                            />
                            <div className="text-sm text-gray-400 mt-2">
                                Gợi ý: Bạn có thể phản hồi về nội dung bài học, cách giảng dạy, hoặc bất kỳ điều gì bạn muốn cải thiện.
                            </div>
                        </div>
                    ) : (
                        <div className="space-y-6">
                            <div className="text-gray-300 mb-4">
                                Xin chào <span className="text-yellow-300 font-semibold">{student?.name}</span>!
                                Hãy nói phản hồi của bạn về buổi học:
                            </div>

                            {/* Hiển thị số lần thử lại nếu có */}
                            {voiceRetryCount > 0 && (
                                <div className="bg-yellow-500/20 border border-yellow-500/30 rounded-xl p-3">
                                    <div className="text-yellow-300 text-sm font-medium">
                                        ⚠️ Lần thử {voiceRetryCount + 1}: Vui lòng nói lại feedback
                                    </div>
                                </div>
                            )}

                            <div className="flex flex-col items-center justify-center space-y-4">
                                <div className={`w-24 h-24 rounded-full flex items-center justify-center ${isRecording ? 'bg-red-500 animate-pulse' : 'bg-blue-500/30'}`}>
                                    <div className="text-4xl">
                                        {isRecording ? '🎙️' : '🎤'}
                                    </div>
                                </div>

                                <div className="text-center">
                                    <div className={`text-lg font-semibold ${isRecording ? 'text-red-400 animate-pulse' : 'text-blue-300'}`}>
                                        {isRecording ? 'Đang ghi âm...' : 'Sẵn sàng ghi âm'}
                                    </div>
                                    <div className="text-sm text-gray-400 mt-2">
                                        {isRecording ? 'Nhấn "Dừng" để kết thúc ghi âm' : 'Nhấn "Bắt đầu" để ghi âm phản hồi'}
                                    </div>
                                </div>

                                {recordedAudio && (
                                    <div className="w-full mt-4">
                                        <div className="text-green-400 text-sm mb-2">✅ Đã ghi âm thành công</div>
                                        <audio controls className="w-full">
                                            <source src={recordedAudio.url} type="audio/wav" />
                                            Trình duyệt của bạn không hỗ trợ phát audio.
                                        </audio>
                                    </div>
                                )}

                                <div className="flex space-x-4 mt-4">
                                    {!isRecording && !recordedAudio ? (
                                        <button
                                            onClick={startRecording}
                                            className="px-6 py-3 bg-red-500 hover:bg-red-600 text-white rounded-xl font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                                            disabled={isSubmittingFeedback}
                                        >
                                            🎤 Bắt đầu ghi âm
                                        </button>
                                    ) : isRecording ? (
                                        <button
                                            onClick={stopRecording}
                                            className="px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-xl font-medium transition-all"
                                        >
                                            ⏹️ Dừng ghi âm
                                        </button>
                                    ) : recordedAudio && (
                                        <button
                                            onClick={() => {
                                                setRecordedAudio(null);
                                                startRecording();
                                            }}
                                            className="px-6 py-3 bg-yellow-500 hover:bg-yellow-600 text-white rounded-xl font-medium transition-all"
                                            disabled={isSubmittingFeedback}
                                        >
                                            🔄 Ghi âm lại
                                        </button>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    <div className="flex justify-end space-x-4 mt-8">
                        <button
                            onClick={closeFeedbackModal}
                            className="px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-xl font-medium transition-all"
                            disabled={isSubmittingFeedback}
                        >
                            Hủy
                        </button>
                        <button
                            onClick={submitFeedback}
                            disabled={isSubmittingFeedback || (feedbackMode === 'text' && !feedbackText.trim()) || (feedbackMode === 'voice' && !recordedAudio)}
                            className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white rounded-xl font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
                        >
                            {isSubmittingFeedback ? (
                                <>
                                    <div className="animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-white"></div>
                                    <span>Đang gửi...</span>
                                </>
                            ) : (
                                <>
                                    <span>📤</span>
                                    <span>Gửi phản hồi</span>
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </div>
        );
    };

    // THÊM: Loading screen với mount check
    if (!mounted || isLoading) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-900 to-purple-900">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-white mx-auto mb-4"></div>
                    <p className="text-xl text-white/80">
                        {!mounted ? 'Đang khởi động...' : 'Đang kết nối đến AI Recognition...'}
                    </p>
                    <p className="text-sm text-white/50 mt-2">
                        Kết nối đến AI server (localhost:5000)
                        <span className="dots">.</span>
                        <span className="dots">.</span>
                        <span className="dots">.</span>
                    </p>
                    <style jsx>{`
                        .dots {
                            animation: blink 1.4s infinite;
                            animation-fill-mode: both;
                        }
                        .dots:nth-child(2) { animation-delay: 0.2s; }
                        .dots:nth-child(3) { animation-delay: 0.4s; }
                        @keyframes blink {
                            0%, 100% { opacity: 0; }
                            50% { opacity: 1; }
                        }
                    `}</style>
                </div>
            </div>
        );
    }

    // Hiển thị chính - chỉ hiện khi có học sinh đã biết
    const shouldShowWelcome = student && !isUnknownStudent(student.name);

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-pink-900 flex flex-col items-center justify-center p-4 relative overflow-hidden">
            {/* Success Popup */}
            {renderSuccessPopup()}

            {/* Feedback Modal */}
            {renderFeedbackModal()}

            {/* Background Animation - Floating faces */}
            <div className="absolute inset-0 overflow-hidden opacity-10">
                {[...Array(20)].map((_, i) => (
                    <div
                        key={i}
                        className="absolute text-4xl animate-float"
                        style={{
                            left: `${Math.random() * 100}%`,
                            top: `${Math.random() * 100}%`,
                            animationDelay: `${Math.random() * 5}s`,
                            animationDuration: `${20 + Math.random() * 30}s`
                        }}
                    >
                        {['😊', '😐', '😲', '👤', '🎓', '🤖'][i % 6]}
                    </div>
                ))}
            </div>

            {/* Close Button */}
            <button
                onClick={handleReturn}
                className="absolute top-6 right-6 z-10 bg-white/20 hover:bg-white/30 text-white px-6 py-3 rounded-full text-sm font-medium backdrop-blur-sm border border-white/30 transition-all hover:scale-105 flex items-center space-x-2"
            >
                <span>←</span>
                <span>Quay lại Dashboard</span>
            </button>

            {/* Connection Status */}
            <div className="absolute top-6 left-6 z-10">
                <div className="flex items-center space-x-2 bg-black/40 backdrop-blur-sm px-3 py-2 rounded-full">
                    <div className={`w-3 h-3 rounded-full ${getConnectionStatusColor()} animate-pulse`}></div>
                    <span className="text-xs text-white">
                        {getConnectionStatusText()}
                    </span>
                </div>
            </div>

            {/* Refresh Button */}
            <button
                onClick={forceRefresh}
                className="absolute top-20 left-6 z-10 bg-white/20 hover:bg-white/30 text-white px-4 py-2 rounded-full text-sm font-medium backdrop-blur-sm border border-white/30 transition-all hover:scale-105 flex items-center space-x-2"
            >
                <span>🔄</span>
                <span>Refresh AI</span>
            </button>

            {/* AI Server Info */}
            <div className="absolute top-20 right-6 z-10">
                <div className="flex items-center space-x-2 bg-black/40 backdrop-blur-sm px-3 py-2 rounded-full">
                    <div className={`w-2 h-2 rounded-full animate-ping ${shouldShowWelcome ? 'bg-green-400' : 'bg-yellow-400'}`}></div>
                    <span className="text-xs text-green-300">AI:5000</span>
                    <span className="text-xs text-white/70">•</span>
                    <span className="text-xs text-cyan-300">Filter: No Unknown</span>
                </div>
            </div>

            {/* Main Content */}
            <div className="relative z-10 text-center max-w-4xl px-4 w-full">
                {/* Animated Face - chỉ hiện khi có học sinh đã biết */}
                {shouldShowWelcome ? (
                    <>
                        <div className="mb-8 relative">
                            <div className="text-[200px] leading-none animate-bounce relative">
                                {'😊'}
                            </div>
                            {/* Pulsing ring effect */}
                            <div className="absolute inset-0 flex items-center justify-center">
                                <div className="w-64 h-64 rounded-full border-2 border-white border-opacity-20 animate-ping"></div>
                            </div>
                        </div>

                        {/* Welcome Text */}
                        <h1 className="text-6xl md:text-7xl font-bold text-white mb-4">
                            <span className="typing-animation">Xin chào</span>
                        </h1>

                        {/* Student Name */}
                        <div className="min-h-[120px] flex items-center justify-center">
                            <h2 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-yellow-300 via-pink-300 to-purple-300 bg-clip-text text-transparent mb-6 transition-all duration-500 ease-in-out transform">
                                {student.name}!
                            </h2>
                        </div>
                        {/* Detection Info Card */}
                        <div className="bg-black/40 backdrop-blur-sm rounded-3xl p-6 md:p-8 mb-6 border border-white/10 shadow-2xl transition-all duration-300 hover:border-white/20">
                            <div className="space-y-4">
                                <div className="">
                                    <div className="text-3xl md:text-4xl text-blue-100 mb-2">
                                        Chào mừng {student.name.split(' ')[0]} đến lớp học! 🎓
                                    </div>
                                    {/* THÊM: Thông báo về feedback */}
                                    <div className="text-lg text-green-300 mt-4 flex items-center justify-center space-x-2">
                                        <span>💬</span>
                                        <span>Hãy chia sẻ phản hồi của bạn về buổi học!</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* THÊM: Feedback Buttons - NẰM NGANG DƯỚI CÂU CHÀO */}
                        <div className="flex justify-center space-x-6 mb-8">
                            <button
                                onClick={() => openFeedbackModal('text')}
                                className="bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white px-8 py-4 rounded-xl text-lg font-medium backdrop-blur-sm border border-white/30 transition-all hover:scale-105 flex items-center space-x-3 shadow-lg transform hover:-translate-y-1"
                            >
                                <span className="text-2xl">📝</span>
                                <div className="text-left">
                                    <div className="font-bold">Nhập feedback</div>
                                    <div className="text-sm opacity-80">Gõ phản hồi của bạn</div>
                                </div>
                            </button>
                            <button
                                onClick={() => openFeedbackModal('voice')}
                                className="bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 text-white px-8 py-4 rounded-xl text-lg font-medium backdrop-blur-sm border border-white/30 transition-all hover:scale-105 flex items-center space-x-3 shadow-lg transform hover:-translate-y-1"
                            >
                                <span className="text-2xl">🎤</span>
                                <div className="text-left">
                                    <div className="font-bold">Nói feedback</div>
                                    <div className="text-sm opacity-80">Ghi âm phản hồi</div>
                                </div>
                            </button>
                        </div>

                    </>
                ) : (
                    /* Hiển thị khi chưa có học sinh đã biết */
                    <>
                        <div className="mb-8 relative">
                            <div className="text-[200px] leading-none animate-pulse relative">
                                🔍
                            </div>
                        </div>

                        <h1 className="text-6xl md:text-7xl font-bold text-white mb-4">
                            Đang tìm học sinh...
                        </h1>

                        <div className="min-h-[120px] flex items-center justify-center">
                            <h2 className="text-4xl md:text-5xl font-bold text-gray-300 mb-6">
                                {connectionStatus === 'no_known_students' ?
                                    'Chỉ phát hiện học sinh chưa biết' :
                                    'Chưa nhận diện được học sinh'}
                            </h2>
                        </div>
                    </>
                )}
            </div>

            {/* Add CSS animations */}
            <style jsx global>{`
                @keyframes float {
                    0%, 100% { 
                        transform: translateY(0) rotate(0deg); 
                        opacity: 0.3;
                    }
                    50% { 
                        transform: translateY(-20px) rotate(180deg); 
                        opacity: 0.6;
                    }
                }
                .animate-float {
                    animation: float linear infinite;
                }
                
                @keyframes typing {
                    from { width: 0; }
                    to { width: 100%; }
                }
                
                .typing-animation {
                    overflow: hidden;
                    border-right: 3px solid white;
                    white-space: nowrap;
                    animation: typing 3s steps(20, end), blink-caret 0.75s step-end infinite;
                }
                
                @keyframes blink-caret {
                    from, to { border-color: transparent; }
                    50% { border-color: white; }
                }
                
                @keyframes fade-in {
                    from {
                        opacity: 0;
                        transform: scale(0.9);
                    }
                    to {
                        opacity: 1;
                        transform: scale(1);
                    }
                }
                
                .animate-fade-in {
                    animation: fade-in 0.3s ease-out;
                }
            `}</style>
        </div>
    );
}