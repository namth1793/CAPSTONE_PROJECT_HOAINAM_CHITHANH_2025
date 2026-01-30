// components/WelcomeDisplay/WelcomeOverlay.js
'use client'
import { useEffect, useState } from 'react'

export default function WelcomeOverlay({ student, onClose }) {
    const [visible, setVisible] = useState(false)
    const [pulse, setPulse] = useState(true)

    useEffect(() => {
        // Trigger entrance animation
        setTimeout(() => setVisible(true), 100)
        
        // Auto-hide after 10 seconds
        const timer = setTimeout(() => {
            setVisible(false)
            setTimeout(onClose, 500)
        }, 10000)

        // Pulse effect for smiley face
        const pulseInterval = setInterval(() => {
            setPulse(prev => !prev)
        }, 2000)

        return () => {
            clearTimeout(timer)
            clearInterval(pulseInterval)
        }
    }, [onClose])

    // Get emotion emoji with animation
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

    // Get welcome message based on emotion
    const getWelcomeMessage = (emotion) => {
        const messages = {
            happy: "Chào bạn! Hôm nay trông bạn thật vui vẻ! 😊",
            neutral: "Xin chào! Chúc bạn một ngày học tập hiệu quả! 📚",
            sad: "Chào bạn! Hy vọng ngày hôm nay của bạn sẽ tốt hơn! 💪",
            surprised: "Ồ! Xin chào! Hôm nay có điều gì thú vị sao? 🤩",
            angry: "Xin chào! Hãy hít thở sâu và bắt đầu ngày mới nhé! 🌿",
            fearful: "Chào bạn! Mọi thứ sẽ ổn thôi, hãy tự tin lên! ✨",
            disgusted: "Xin chào! Hy vọng bạn có một ngày tốt lành! 🌈"
        }
        return messages[emotion] || "Xin chào! Chào mừng bạn đến với lớp học! 🎓"
    }

    return (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center">
            {/* Animated Background */}
            <div className={`absolute inset-0 bg-gradient-to-br from-blue-900/95 via-purple-900/95 to-pink-900/95 transition-opacity duration-1000 ${visible ? 'opacity-100' : 'opacity-0'}`}>
                {/* Animated particles */}
                <div className="absolute inset-0 overflow-hidden">
                    {[...Array(20)].map((_, i) => (
                        <div
                            key={i}
                            className="absolute w-1 h-1 bg-white/30 rounded-full animate-float"
                            style={{
                                left: `${Math.random() * 100}%`,
                                top: `${Math.random() * 100}%`,
                                animationDelay: `${Math.random() * 5}s`,
                                animationDuration: `${10 + Math.random() * 10}s`
                            }}
                        />
                    ))}
                </div>
            </div>

            {/* Main Content */}
            <div className={`relative z-10 text-center px-8 transition-all duration-1000 transform ${visible ? 'opacity-100 scale-100' : 'opacity-0 scale-90'}`}>
                {/* Smiley Face */}
                <div className={`mb-8 transition-transform duration-1000 ${pulse ? 'scale-110' : 'scale-100'}`}>
                    <div className="text-[200px] leading-none">
                        {getEmotionEmoji(student.emotion)}
                    </div>
                </div>

                {/* Welcome Text */}
                <h1 className="text-7xl font-bold text-white mb-4 animate-glow">
                    Xin chào
                </h1>
                
                {/* Student Name */}
                <h2 className="text-6xl font-bold bg-gradient-to-r from-yellow-300 via-pink-300 to-purple-300 bg-clip-text text-transparent mb-6">
                    {student.name}!
                </h2>

                {/* Emotion Message */}
                <p className="text-2xl text-blue-100 mb-12 max-w-2xl mx-auto">
                    {getWelcomeMessage(student.emotion)}
                </p>

                {/* Confidence Indicator */}
                <div className="inline-flex items-center space-x-4 bg-black/30 backdrop-blur-sm rounded-full px-6 py-3 mb-8 border border-white/20">
                    <div className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                        <span className="text-green-300 font-medium">Face Recognized</span>
                    </div>
                    <div className="h-6 w-px bg-white/30"></div>
                    <div className="text-white/80">
                        Confidence: <span className="font-bold text-yellow-300">{(student.confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>

                {/* Close Button */}
                <button
                    onClick={() => {
                        setVisible(false)
                        setTimeout(onClose, 500)
                    }}
                    className="group bg-white/20 hover:bg-white/30 text-white px-8 py-4 rounded-full text-lg font-semibold backdrop-blur-sm border border-white/30 transition-all duration-300 hover:scale-105 hover:shadow-2xl"
                >
                    <span className="flex items-center space-x-2">
                        <span>Return to Dashboard</span>
                        <span className="group-hover:translate-x-1 transition-transform">→</span>
                    </span>
                </button>

                {/* Instruction */}
                <p className="mt-8 text-white/50 text-sm">
                    This message will auto-close in 10 seconds, or click above to close manually
                </p>
            </div>

            {/* Add CSS animations */}
            <style jsx global>{`
                @keyframes float {
                    0%, 100% { transform: translateY(0) rotate(0deg); }
                    50% { transform: translateY(-20px) rotate(180deg); }
                }
                @keyframes glow {
                    0%, 100% { text-shadow: 0 0 20px rgba(255,255,255,0.5); }
                    50% { text-shadow: 0 0 40px rgba(255,255,255,0.8); }
                }
                .animate-float {
                    animation: float linear infinite;
                }
                .animate-glow {
                    animation: glow 2s ease-in-out infinite;
                }
            `}</style>
        </div>
    )
}