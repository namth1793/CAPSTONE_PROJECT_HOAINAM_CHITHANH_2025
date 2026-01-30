// components/WelcomeDisplay/WelcomeDisplay.js
'use client'
import { useEffect, useState } from 'react'
import WelcomeOverlay from './WelcomeOverlay'

export default function WelcomeDisplay() {
    const [detectedStudent, setDetectedStudent] = useState(null)
    const [showOverlay, setShowOverlay] = useState(false)
    const [isDetecting, setIsDetecting] = useState(false)

    // Simulate face detection (replace with actual AI integration)
    const simulateFaceDetection = () => {
        setIsDetecting(true)
        
        // Simulate API call to AI face recognition
        setTimeout(() => {
            const sampleStudents = [
                { id: 'SV001', name: 'Nguyễn Văn A', emotion: 'happy', confidence: 0.92 },
                { id: 'SV002', name: 'Trần Thị B', emotion: 'neutral', confidence: 0.87 },
                { id: 'SV003', name: 'Lê Văn C', emotion: 'surprised', confidence: 0.78 },
                { id: 'SV004', name: 'Phạm Thị D', emotion: 'happy', confidence: 0.95 },
            ]
            
            const randomStudent = sampleStudents[Math.floor(Math.random() * sampleStudents.length)]
            setDetectedStudent(randomStudent)
            setIsDetecting(false)
            
            // Auto-show welcome message
            setShowOverlay(true)
        }, 1500)
    }

    // Listen for WebSocket messages from AI system
    useEffect(() => {
        const handleAIDetection = (event) => {
            // This would come from WebSocket or API real-time updates
            if (event.detail && event.detail.type === 'face_detected') {
                setDetectedStudent(event.detail.student)
                setShowOverlay(true)
            }
        }

        // Add event listener for AI detection messages
        window.addEventListener('ai_detection', handleAIDetection)
        
        // Simulate initial detection for demo
        const timer = setTimeout(simulateFaceDetection, 2000)

        return () => {
            window.removeEventListener('ai_detection', handleAIDetection)
            clearTimeout(timer)
        }
    }, [])

    // Handle manual trigger from sidebar
    const handleShowWelcome = () => {
        if (detectedStudent) {
            setShowOverlay(true)
        } else {
            simulateFaceDetection()
        }
    }

    // Trigger AI detection manually
    const triggerDetection = () => {
        simulateFaceDetection()
    }

    // Get emotion emoji
    const getEmotionEmoji = (emotion) => {
        const emojis = {
            happy: '😊',
            neutral: '😐',
            sad: '😢',
            surprised: '😲',
            angry: '😠',
            fearful: '😨',
            disgusted: '🤢'
        }
        return emojis[emotion] || '😐'
    }

    return (
        <>
            {/* Welcome Button in Main UI */}
            <div className="fixed bottom-6 right-6 z-50">
                <button
                    onClick={triggerDetection}
                    className="flex items-center space-x-2 bg-gradient-to-r from-blue-500 to-purple-500 text-white px-6 py-3 rounded-full shadow-lg hover:shadow-xl transition-all hover:scale-105"
                >
                    <span className="text-xl">👋</span>
                    <span className="font-semibold">Detect Face</span>
                    {isDetecting && (
                        <span className="ml-2 animate-spin">⟳</span>
                    )}
                </button>
            </div>

            {/* Status Indicator */}
            {detectedStudent && !showOverlay && (
                <div className="fixed top-6 right-6 z-50">
                    <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-4 border border-gray-700 shadow-xl">
                        <div className="flex items-center space-x-3">
                            <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full flex items-center justify-center text-white text-2xl">
                                {getEmotionEmoji(detectedStudent.emotion)}
                            </div>
                            <div>
                                <p className="text-sm text-gray-400">Last detected</p>
                                <p className="text-lg font-bold text-white">{detectedStudent.name}</p>
                                <p className="text-xs text-gray-500">
                                    Confidence: {(detectedStudent.confidence * 100).toFixed(1)}%
                                </p>
                            </div>
                        </div>
                        <button
                            onClick={() => setShowOverlay(true)}
                            className="mt-3 w-full bg-blue-600 hover:bg-blue-700 text-white py-2 rounded-lg text-sm font-medium transition-colors"
                        >
                            Show Welcome Message
                        </button>
                    </div>
                </div>
            )}

            {/* Welcome Overlay */}
            {showOverlay && detectedStudent && (
                <WelcomeOverlay
                    student={detectedStudent}
                    onClose={() => setShowOverlay(false)}
                />
            )}
        </>
    )
}