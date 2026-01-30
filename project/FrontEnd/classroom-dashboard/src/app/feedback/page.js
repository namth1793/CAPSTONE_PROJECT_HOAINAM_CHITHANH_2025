// src/app/feedback/page.js
'use client'
import { useEffect, useRef, useState } from 'react'
import { buildApiUrl, buildWebSocketUrl } from '../config/api'

export default function FeedbackPage() {
    const [feedbacks, setFeedbacks] = useState([])
    const [filteredFeedbacks, setFilteredFeedbacks] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [lastUpdate, setLastUpdate] = useState(null)
    const [newDataCount, setNewDataCount] = useState(0)
    const [connectionStatus, setConnectionStatus] = useState('connecting')
    const [currentPage, setCurrentPage] = useState(1)
    const [selectedFeedback, setSelectedFeedback] = useState(null)
    const [selectedStudent, setSelectedStudent] = useState(null)
    const [studentFeedbacks, setStudentFeedbacks] = useState([])
    const [filters, setFilters] = useState({
        feedbackType: 'all',
        emotion: 'all',
        rating: 'all',
        dateRange: 'all',
        searchQuery: ''
    })
    const [stats, setStats] = useState({
        total: 0,
        text: 0,
        voice: 0,
        emotionDistribution: {},
        ratingDistribution: {},
        recentCount: 0
    })
    const [processingFeedback, setProcessingFeedback] = useState(null)
    const [audioPlaying, setAudioPlaying] = useState(null)

    const socketRef = useRef(null)
    const pollingIntervalRef = useRef(null)
    const audioRef = useRef(null)
    const itemsPerPage = 10

    useEffect(() => {
        // Lấy data lần đầu
        fetchFeedbacksData()

        // Khởi tạo WebSocket connection
        initWebSocket()

        // Khởi tạo polling (mỗi 15 giây)
        initPolling()

        // Cleanup khi component unmount
        return () => {
            cleanupConnections()
            if (audioRef.current) {
                audioRef.current.pause()
                audioRef.current = null
            }
        }
    }, [])

    useEffect(() => {
        // Áp dụng filters khi có thay đổi
        applyFilters()
    }, [feedbacks, filters])

    // Helper function để kiểm tra bản ghi có phải mới không (trong 5 phút)
    const isRecordRecent = (timestamp) => {
        if (!timestamp) return false
        try {
            const recordTime = new Date(timestamp)
            const now = new Date()
            const diffMinutes = (now - recordTime) / (1000 * 60)
            return diffMinutes <= 5
        } catch {
            return false
        }
    }

    const initWebSocket = () => {
        try {
            setConnectionStatus('connecting')

            // Kết nối WebSocket với API server
            const socket = new WebSocket(buildWebSocketUrl('/ws/live'))
            socketRef.current = socket

            socket.onopen = () => {
                console.log('✅ WebSocket connected for feedback')
                setConnectionStatus('connected')
                
                // Subscribe to feedback updates
                socket.send(JSON.stringify({
                    type: 'subscribe',
                    channel: 'feedback_updates'
                }))
            }

            socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data)
                    console.log('📡 Feedback WebSocket message:', data.type || 'unknown')

                    // Xử lý các loại message liên quan đến feedback
                    switch (data.type) {
                        case 'feedback_created':
                            console.log('📝 New feedback:', data.data?.student_name)
                            handleNewFeedback(data.data)
                            setNewDataCount(prev => prev + 1)
                            fetchFeedbacksData()
                            break

                        case 'voice_feedback_processed':
                            console.log('🎤 Voice feedback processed:', data.data?.student_name)
                            handleVoiceProcessed(data.data)
                            setNewDataCount(prev => prev + 1)
                            fetchFeedbacksData()
                            break

                        case 'feedback_updated':
                            console.log('🔄 Feedback updated:', data.data?.id)
                            handleFeedbackUpdated(data.data)
                            setNewDataCount(prev => prev + 1)
                            fetchFeedbacksData()
                            break

                        case 'stt_processed':
                            console.log('🗣️ STT processed for feedback:', data.feedback_id)
                            handleSTTProcessed(data)
                            setNewDataCount(prev => prev + 1)
                            fetchFeedbacksData()
                            break

                        case 'batch_processed':
                            // Có thể có feedback trong batch data
                            if (data.message?.includes('feedback')) {
                                console.log('🔄 Batch data may contain feedback')
                                fetchFeedbacksData()
                            }
                            break

                        case 'heartbeat':
                            socket.send(JSON.stringify({ 
                                type: 'heartbeat_ack', 
                                timestamp: new Date().toISOString() 
                            }))
                            break

                        default:
                            // Bỏ qua các message không liên quan
                            break
                    }
                } catch (error) {
                    console.error('❌ Error parsing WebSocket message:', error, event.data)
                }
            }

            socket.onerror = (error) => {
                console.error('❌ WebSocket error:', error)
                setConnectionStatus('error')
            }

            socket.onclose = () => {
                console.log('🔌 WebSocket disconnected')
                setConnectionStatus('disconnected')

                // Thử reconnect sau 5s
                setTimeout(() => {
                    if (socketRef.current?.readyState === WebSocket.CLOSED) {
                        console.log('🔄 Attempting WebSocket reconnection...')
                        initWebSocket()
                    }
                }, 5000)
            }
        } catch (error) {
            console.error('❌ Failed to initialize WebSocket:', error)
            setConnectionStatus('error')
        }
    }

    const initPolling = () => {
        // Polling cho feedback data mỗi 15 giây
        pollingIntervalRef.current = setInterval(() => {
            console.log('🔄 Auto-refreshing feedback data...')
            fetchFeedbacksData()
        }, 15000) // 15 giây
    }

    const cleanupConnections = () => {
        // Đóng WebSocket
        if (socketRef.current) {
            socketRef.current.close()
            socketRef.current = null
        }

        // Xóa polling intervals
        if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current)
            pollingIntervalRef.current = null
        }
    }

    const fetchFeedbacksData = async (forceRefresh = false) => {
        try {
            setError(null)
            setLoading(true)
            console.log('🔄 Fetching feedbacks data...')

            // Gọi endpoint feedback với các tham số
            const url = buildApiUrl('/api/feedback?limit=100&sort=desc')
            const response = await fetch(url)

            if (!response.ok) {
                throw new Error(`Feedback API Error: ${response.status}`)
            }

            const data = await response.json()
            console.log('📝 Feedback API response:', {
                status: data.status,
                count: data.count || 0,
                total: data.total || 0
            })

            if (data.status === 'success' && data.feedbacks && data.feedbacks.length > 0) {
                console.log(`✅ Processing ${data.feedbacks.length} feedback records`)
                processFeedbacksData(data.feedbacks)
                calculateStats(data.feedbacks)
            } else {
                console.log('📭 No feedback data, using fallback...')
                useFallbackFeedbackData()
            }

        } catch (error) {
            console.error('❌ Error fetching feedback data:', error)
            setError(error.message || 'Failed to load feedback data')
            useFallbackFeedbackData()
        } finally {
            setLoading(false)
        }
    }

    const fetchFeedbackStats = async () => {
        try {
            const response = await fetch(buildApiUrl('/api/feedback/stats?days=7'))
            if (response.ok) {
                const data = await response.json()
                if (data.status === 'success') {
                    setStats(prev => ({
                        ...prev,
                        ...data.stats,
                        recentCount: data.stats?.total_feedbacks || 0
                    }))
                }
            }
        } catch (error) {
            console.error('❌ Error fetching feedback stats:', error)
        }
    }

    const processFeedbacksData = (feedbacksArray) => {
        if (!feedbacksArray || feedbacksArray.length === 0) {
            console.log('⚠️ No feedback data to process')
            return
        }

        console.log(`📊 Processing ${feedbacksArray.length} feedback records...`)

        const processedFeedbacks = feedbacksArray.map(feedback => {
            // Kiểm tra xem có phải feedback mới không
            const isNew = isRecordRecent(feedback.created_at)
            
            // Xử lý transcribed text
            let displayText = feedback.feedback_text || ''
            if (feedback.feedback_type === 'voice' && feedback.transcribed_text) {
                displayText = feedback.transcribed_text
            }

            // Xử lý emotion display
            let emotionDisplay = feedback.emotion || 'unknown'
            if (feedback.emotion_confidence && feedback.emotion_confidence > 0) {
                emotionDisplay += ` (${Math.round(feedback.emotion_confidence * 100)}%)`
            }

            return {
                id: feedback.id,
                student_id: feedback.student_id,
                student_name: feedback.student_name,
                feedback_text: displayText,
                transcribed_text: feedback.transcribed_text,
                original_text: feedback.feedback_text,
                feedback_type: feedback.feedback_type,
                audio_path: feedback.audio_path,
                audio_duration: feedback.audio_duration,
                confidence: feedback.confidence,
                emotion: feedback.emotion,
                emotion_confidence: feedback.emotion_confidence,
                emotion_display: emotionDisplay,
                rating: feedback.rating,
                class_name: feedback.class_name || 'Chưa xác định',
                session_id: feedback.session_id,
                created_at: feedback.created_at,
                updated_at: feedback.updated_at,
                isNew: isNew,
                has_audio: !!feedback.audio_path,
                is_processed: feedback.feedback_type === 'text' || 
                            (feedback.feedback_type === 'voice' && feedback.transcribed_text),
                needs_processing: feedback.feedback_type === 'voice' && 
                                (!feedback.transcribed_text || feedback.transcribed_text.includes('STT failed'))
            }
        })

        // Sắp xếp theo thời gian mới nhất
        const sortedFeedbacks = processedFeedbacks.sort((a, b) => 
            new Date(b.created_at) - new Date(a.created_at)
        )

        setFeedbacks(sortedFeedbacks)
        setLastUpdate(new Date())
        setCurrentPage(1)

        // Tính toán stats
        calculateStats(sortedFeedbacks)
    }

    const calculateStats = (feedbacksArray) => {
        if (!feedbacksArray || feedbacksArray.length === 0) {
            setStats({
                total: 0,
                text: 0,
                voice: 0,
                emotionDistribution: {},
                ratingDistribution: {},
                recentCount: 0
            })
            return
        }

        const now = new Date()
        const sevenDaysAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000)

        const stats = {
            total: feedbacksArray.length,
            text: feedbacksArray.filter(f => f.feedback_type === 'text').length,
            voice: feedbacksArray.filter(f => f.feedback_type === 'voice').length,
            emotionDistribution: {},
            ratingDistribution: {},
            recentCount: feedbacksArray.filter(f => 
                new Date(f.created_at) >= sevenDaysAgo
            ).length,
            unprocessedVoice: feedbacksArray.filter(f => 
                f.feedback_type === 'voice' && 
                (!f.transcribed_text || f.transcribed_text.includes('STT failed'))
            ).length
        }

        // Tính phân phối cảm xúc
        feedbacksArray.forEach(feedback => {
            const emotion = feedback.emotion || 'unknown'
            stats.emotionDistribution[emotion] = (stats.emotionDistribution[emotion] || 0) + 1
        })

        // Tính phân phối rating
        feedbacksArray.forEach(feedback => {
            if (feedback.rating) {
                stats.ratingDistribution[feedback.rating] = (stats.ratingDistribution[feedback.rating] || 0) + 1
            }
        })

        setStats(stats)
    }

    const useFallbackFeedbackData = () => {
        console.log('🔄 Using fallback feedback data...')
        const fallbackData = getFallbackFeedbackData()
        setFeedbacks(fallbackData)
        calculateStats(fallbackData)
        setLastUpdate(new Date())
        setCurrentPage(1)
    }

    const getFallbackFeedbackData = () => {
        const now = new Date()
        return [
            {
                id: 1,
                student_id: 'SV001',
                student_name: 'Nguyễn Văn A',
                feedback_text: 'Bài giảng hôm nay rất thú vị, em hiểu bài ngay trên lớp',
                feedback_type: 'text',
                emotion: 'happy',
                rating: 5,
                class_name: 'Lớp 10A1',
                created_at: new Date(now.getTime() - 3600000).toISOString(),
                isNew: false,
                has_audio: false,
                is_processed: true
            },
            {
                id: 2,
                student_id: 'SV002',
                student_name: 'Trần Thị B',
                feedback_text: 'Em cảm thấy phần bài tập hơi khó, cần thêm thời gian làm',
                feedback_type: 'text',
                emotion: 'neutral',
                rating: 3,
                class_name: 'Lớp 10A1',
                created_at: new Date(now.getTime() - 7200000).toISOString(),
                isNew: false,
                has_audio: false,
                is_processed: true
            },
            {
                id: 3,
                student_id: 'SV003',
                student_name: 'Lê Văn C',
                transcribed_text: 'Em muốn thầy giảng chậm hơn một chút ở phần công thức',
                feedback_type: 'voice',
                audio_duration: 12.5,
                confidence: 0.85,
                emotion: 'sad',
                rating: 4,
                class_name: 'Lớp 10A1',
                created_at: new Date(now.getTime() - 10800000).toISOString(),
                isNew: false,
                has_audio: true,
                is_processed: true
            },
            {
                id: 4,
                student_id: 'SV004',
                student_name: 'Phạm Thị D',
                feedback_text: 'Phòng học hơi nóng, em đề xuất bật điều hòa',
                feedback_type: 'text',
                emotion: 'neutral',
                rating: 4,
                class_name: 'Lớp 10A1',
                created_at: new Date(now.getTime() - 14400000).toISOString(),
                isNew: false,
                has_audio: false,
                is_processed: true
            },
            {
                id: 5,
                student_id: 'SV005',
                student_name: 'Hoàng Văn E',
                transcribed_text: 'Em rất thích cách dạy của thầy, dễ hiểu và sinh động',
                feedback_type: 'voice',
                audio_duration: 8.2,
                confidence: 0.92,
                emotion: 'happy',
                rating: 5,
                class_name: 'Lớp 10A1',
                created_at: new Date(now.getTime() - 18000000).toISOString(),
                isNew: true,
                has_audio: true,
                is_processed: true
            }
        ]
    }

    const applyFilters = () => {
        let filtered = [...feedbacks]

        // Filter by type
        if (filters.feedbackType !== 'all') {
            filtered = filtered.filter(f => f.feedback_type === filters.feedbackType)
        }

        // Filter by emotion
        if (filters.emotion !== 'all') {
            filtered = filtered.filter(f => f.emotion === filters.emotion)
        }

        // Filter by rating
        if (filters.rating !== 'all') {
            filtered = filtered.filter(f => f.rating === parseInt(filters.rating))
        }

        // Filter by date range
        if (filters.dateRange !== 'all') {
            const now = new Date()
            let startDate = new Date()
            
            switch (filters.dateRange) {
                case 'today':
                    startDate.setHours(0, 0, 0, 0)
                    break
                case 'week':
                    startDate.setDate(now.getDate() - 7)
                    break
                case 'month':
                    startDate.setMonth(now.getMonth() - 1)
                    break
            }
            
            filtered = filtered.filter(f => new Date(f.created_at) >= startDate)
        }

        // Filter by search query
        if (filters.searchQuery) {
            const query = filters.searchQuery.toLowerCase()
            filtered = filtered.filter(f => 
                f.student_name.toLowerCase().includes(query)
            )
        }

        setFilteredFeedbacks(filtered)
        setCurrentPage(1)
    }

    const handleNewFeedback = (data) => {
        console.log('Processing new feedback:', data)
        // Cập nhật UI để hiển thị feedback mới
        setNewDataCount(prev => prev + 1)
    }

    const handleVoiceProcessed = (data) => {
        console.log('Voice feedback processed:', data)
        // Có thể cập nhật UI cho feedback đã được xử lý
    }

    const handleFeedbackUpdated = (data) => {
        console.log('Feedback updated:', data)
        // Cập nhật feedback cụ thể
    }

    const handleSTTProcessed = (data) => {
        console.log('STT processed:', data)
        // Cập nhật text cho voice feedback
    }

    const handleFilterChange = (filterName, value) => {
        setFilters(prev => ({
            ...prev,
            [filterName]: value
        }))
    }

    const handleResetFilters = () => {
        setFilters({
            feedbackType: 'all',
            emotion: 'all',
            rating: 'all',
            dateRange: 'all',
            searchQuery: ''
        })
    }

    const fetchStudentFeedbacks = async (studentId) => {
        try {
            console.log(`🔍 Fetching feedbacks for student: ${studentId}`)
            const response = await fetch(buildApiUrl(`/api/feedback?student_id=${studentId}&limit=20`))
            if (response.ok) {
                const data = await response.json()
                if (data.status === 'success' && data.feedbacks) {
                    setStudentFeedbacks(data.feedbacks)
                    setSelectedStudent(studentId)
                }
            }
        } catch (error) {
            console.error('❌ Error fetching student feedbacks:', error)
        }
    }

    const processSTT = async (feedbackId) => {
        try {
            setProcessingFeedback(feedbackId)
            console.log(`🔄 Processing STT for feedback: ${feedbackId}`)
            
            const response = await fetch(buildApiUrl(`/api/feedback/process-stt/${feedbackId}`), {
                method: 'POST'
            })

            if (response.ok) {
                const data = await response.json()
                console.log('✅ STT processed:', data)
                
                // Refresh data
                setTimeout(() => {
                    fetchFeedbacksData()
                }, 1000)
            }
        } catch (error) {
            console.error('❌ Error processing STT:', error)
        } finally {
            setTimeout(() => setProcessingFeedback(null), 2000)
        }
    }

    const playAudio = async (audioPath) => {
        try {
            if (audioPlaying === audioPath) {
                // Stop audio if already playing
                if (audioRef.current) {
                    audioRef.current.pause()
                    audioRef.current = null
                }
                setAudioPlaying(null)
                return
            }

            // Construct full URL to audio file
            const audioUrl = audioPath?.startsWith('http')
                ? audioPath
                : audioPath
                    ? buildApiUrl(audioPath)
                    : ''
            console.log('🎵 Playing audio:', audioUrl)

            // Create audio element
            const audio = new Audio(audioUrl)
            audioRef.current = audio

            audio.onplay = () => {
                setAudioPlaying(audioPath)
            }

            audio.onended = () => {
                setAudioPlaying(null)
                audioRef.current = null
            }

            audio.onerror = () => {
                console.error('❌ Error playing audio')
                setAudioPlaying(null)
                audioRef.current = null
            }

            await audio.play()
        } catch (error) {
            console.error('❌ Error playing audio:', error)
            setAudioPlaying(null)
        }
    }

    // Các hàm helper cho UI
    const getEmotionColor = (emotion) => {
        const colors = {
            happy: { text: 'text-green-400', bg: 'bg-green-900/30', border: 'border-green-800/50', icon: '😊', label: 'Vui vẻ' },
            neutral: { text: 'text-blue-400', bg: 'bg-blue-900/30', border: 'border-blue-800/50', icon: '😐', label: 'Bình thường' },
            sad: { text: 'text-red-400', bg: 'bg-red-900/30', border: 'border-red-800/50', icon: '😢', label: 'Buồn' },
            angry: { text: 'text-orange-400', bg: 'bg-orange-900/30', border: 'border-orange-800/50', icon: '😠', label: 'Tức giận' },
            surprised: { text: 'text-purple-400', bg: 'bg-purple-900/30', border: 'border-purple-800/50', icon: '😮', label: 'Ngạc nhiên' },
            fearful: { text: 'text-pink-400', bg: 'bg-pink-900/30', border: 'border-pink-800/50', icon: '😨', label: 'Sợ hãi' },
            disgusted: { text: 'text-yellow-600', bg: 'bg-yellow-900/30', border: 'border-yellow-800/50', icon: '🤢', label: 'Chán ghét' },
            unknown: { text: 'text-gray-400', bg: 'bg-gray-900/30', border: 'border-gray-800/50', icon: '❓', label: 'Chưa xác định' }
        }
        return colors[emotion] || colors.unknown
    }

    const getFeedbackTypeColor = (type) => {
        const colors = {
            text: { text: 'text-blue-400', bg: 'bg-blue-900/20', border: 'border-blue-700/30', icon: '📝', label: 'Văn bản' },
            voice: { text: 'text-purple-400', bg: 'bg-purple-900/20', border: 'border-purple-700/30', icon: '🎤', label: 'Giọng nói' }
        }
        return colors[type] || colors.text
    }

    const getRatingStars = (rating) => {
        if (!rating) return 'Chưa đánh giá'
        return '⭐'.repeat(rating) + '☆'.repeat(5 - rating)
    }

    const getRatingColor = (rating) => {
        if (!rating) return 'text-gray-400'
        if (rating >= 4) return 'text-green-400'
        if (rating >= 3) return 'text-yellow-400'
        return 'text-red-400'
    }

    const formatTime = (timestamp) => {
        if (!timestamp) return 'Vừa xong'
        try {
            const date = new Date(timestamp)
            const now = new Date()
            const diffMs = now - date
            const diffMins = Math.floor(diffMs / 60000)

            if (diffMins < 1) return 'Vừa xong'
            if (diffMins < 60) return `${diffMins} phút trước`
            if (diffMins < 1440) return `${Math.floor(diffMins / 60)} giờ trước`
            return date.toLocaleDateString('vi-VN', {
                hour: '2-digit',
                minute: '2-digit'
            })
        } catch (error) {
            return 'Gần đây'
        }
    }

    const formatDate = (timestamp) => {
        if (!timestamp) return ''
        try {
            const date = new Date(timestamp)
            return date.toLocaleDateString('vi-VN', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            })
        } catch (error) {
            return timestamp
        }
    }

    const truncateText = (text, maxLength = 100) => {
        if (!text) return ''
        if (text.length <= maxLength) return text
        return text.substring(0, maxLength) + '...'
    }

    const getConnectionStatusColor = () => {
        switch (connectionStatus) {
            case 'connected': return 'bg-green-500'
            case 'connecting': return 'bg-yellow-500 animate-pulse'
            case 'disconnected': return 'bg-red-500'
            case 'error': return 'bg-red-500'
            default: return 'bg-gray-500'
        }
    }

    const getConnectionStatusText = () => {
        switch (connectionStatus) {
            case 'connected': return 'Live Connected'
            case 'connecting': return 'Connecting...'
            case 'disconnected': return 'Disconnected'
            case 'error': return 'Connection Error'
            default: return 'Unknown'
        }
    }

    // Tính toán pagination
    const totalPages = Math.ceil(filteredFeedbacks.length / itemsPerPage)
    const startIndex = (currentPage - 1) * itemsPerPage
    const endIndex = startIndex + itemsPerPage
    const currentFeedbacks = filteredFeedbacks.slice(startIndex, endIndex)

    const handlePageChange = (pageNumber) => {
        setCurrentPage(pageNumber)
        const tableElement = document.querySelector('.feedbacks-table')
        if (tableElement) {
            tableElement.scrollIntoView({ behavior: 'smooth' })
        }
    }

    const handleRefresh = () => {
        setLoading(true)
        setNewDataCount(0)
        fetchFeedbacksData()
    }

    if (loading && feedbacks.length === 0) {
        return (
            <div className="feedback-page flex justify-center items-center min-h-screen bg-black">
                <div className="text-center">
                    <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500 mb-4"></div>
                    <span className="text-lg text-white block">Loading Feedback Analytics...</span>
                    <p className="text-gray-400 mt-2">Fetching feedback data from API...</p>
                    <div className="mt-4 flex justify-center items-center">
                        <div className={`w-3 h-3 rounded-full ${getConnectionStatusColor()} mr-2`}></div>
                        <span className="text-sm text-gray-400">{getConnectionStatusText()}</span>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="feedback-page bg-[#B39858] p-6 min-h-screen">
            <div className="max-w-7xl mx-auto">
                {/* Header với status */}
                <div className="text-center mb-8">
                    <h1 className="text-4xl font-bold text-white mb-3">Student Feedback</h1>
                    <div className="flex justify-center items-center space-x-4 mb-2">
                        <p className="text-blue-100 text-lg">Real-time feedback analysis and monitoring</p>
                    </div>
                </div>


                {/* Filters Section */}
                <div className="bg-gray-900/50 rounded-xl p-4 mb-8 border border-gray-800">
                    <div className="flex flex-wrap gap-4 items-center">
                        <div className="flex-1">
                            <input
                                type="text"
                                placeholder="Tìm theo tên học sinh..."
                                value={filters.searchQuery}
                                onChange={(e) => handleFilterChange('searchQuery', e.target.value)}
                                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            />
                        </div>

                        <select
                            value={filters.feedbackType}
                            onChange={(e) => handleFilterChange('feedbackType', e.target.value)}
                            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white"
                        >
                            <option value="all">All Types</option>
                            <option value="text">Text Only</option>
                            <option value="voice">Voice Only</option>
                        </select>

                        <select
                            value={filters.dateRange}
                            onChange={(e) => handleFilterChange('dateRange', e.target.value)}
                            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white"
                        >
                            <option value="all">All Time</option>
                            <option value="today">Today</option>
                            <option value="week">Last 7 Days</option>
                            <option value="month">Last 30 Days</option>
                        </select>

                        <button
                            onClick={handleResetFilters}
                            className="bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition"
                        >
                            Reset Filters
                        </button>

                        <button
                            onClick={handleRefresh}
                            className="bg-gradient-to-r from-blue-600 to-purple-600 hover:opacity-90 text-white px-4 py-2 rounded-lg transition flex items-center"
                        >
                            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                            </svg>
                            Refresh
                        </button>
                    </div>

                    {/* Active filters display */}
                    <div className="mt-4 flex flex-wrap gap-2">
                        {filters.feedbackType !== 'all' && (
                            <span className="bg-blue-900/50 text-blue-300 px-3 py-1 rounded-full text-sm">
                                Type: {filters.feedbackType}
                            </span>
                        )}
                        {filters.emotion !== 'all' && (
                            <span className="bg-purple-900/50 text-purple-300 px-3 py-1 rounded-full text-sm">
                                Emotion: {filters.emotion}
                            </span>
                        )}
                        {filters.rating !== 'all' && (
                            <span className="bg-yellow-900/50 text-yellow-300 px-3 py-1 rounded-full text-sm">
                                Rating: {filters.rating} ⭐
                            </span>
                        )}
                        {filters.dateRange !== 'all' && (
                            <span className="bg-green-900/50 text-green-300 px-3 py-1 rounded-full text-sm">
                                Period: {filters.dateRange}
                            </span>
                        )}
                        {filters.searchQuery && (
                            <span className="bg-gray-800 text-gray-300 px-3 py-1 rounded-full text-sm">
                                Search: "{filters.searchQuery}"
                            </span>
                        )}
                    </div>
                </div>

                {/* Feedbacks Table */}
                <div className="bg-gray-900 rounded-2xl shadow-xl overflow-hidden border border-gray-800 mb-8 feedbacks-table">
                    <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-6">
                        <div className="flex justify-between items-center">
                            <h2 className="text-2xl font-bold text-white flex items-center">
                                <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                                </svg>
                                Student Feedbacks
                                {newDataCount > 0 && (
                                    <span className="ml-3 px-2 py-1 bg-green-900/50 text-green-400 text-xs rounded-full animate-pulse">
                                        {newDataCount} feedback mới
                                    </span>
                                )}
                            </h2>
                            <div className="flex items-center space-x-2">
                                <span className="text-sm text-gray-300">
                                    Hiển thị {startIndex + 1}-{Math.min(endIndex, filteredFeedbacks.length)} của {filteredFeedbacks.length} feedbacks
                                </span>
                            </div>
                        </div>
                    </div>
                    <div className="p-6">
                        {filteredFeedbacks.length === 0 ? (
                            <div className="text-center py-8">
                                <p className="text-gray-400">Không tìm thấy feedback nào phù hợp với bộ lọc</p>
                                <button
                                    onClick={handleResetFilters}
                                    className="mt-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white py-2 px-4 rounded-lg hover:opacity-90 transition"
                                >
                                    Xóa Bộ Lọc
                                </button>
                            </div>
                        ) : (
                            <>
                                <div className="overflow-x-auto">
                                    <table className="w-full min-w-full">
                                        <thead className="bg-gray-800">
                                            <tr>
                                                <th className="text-left p-4 text-gray-300 font-semibold">Học Sinh</th>
                                                <th className="text-left p-4 text-gray-300 font-semibold">Feedback</th>
                                                <th className="text-left p-4 text-gray-300 font-semibold">Loại</th>
                                                <th className="text-left p-4 text-gray-300 font-semibold">Thời Gian</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-gray-800">
                                            {currentFeedbacks.map((feedback, index) => {
                                                const emotionStyle = getEmotionColor(feedback.emotion)
                                                const typeStyle = getFeedbackTypeColor(feedback.feedback_type)
                                                const ratingColor = getRatingColor(feedback.rating)

                                                return (
                                                    <tr
                                                        key={`${feedback.id}_${index}`}
                                                        className={`hover:bg-gray-800/50 transition duration-150 ${feedback.isNew ? 'animate-pulse-once bg-gradient-to-r from-green-900/20 to-emerald-900/10' : ''
                                                            }`}
                                                    >
                                                        <td className="p-4">
                                                            <div className="flex items-center">
                                                                <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center text-white font-semibold mr-3 border border-blue-700 relative">
                                                                    {feedback.student_name?.charAt(0) || '?'}
                                                                    {feedback.isNew && (
                                                                        <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                                                                    )}
                                                                </div>
                                                                <div>
                                                                    <div className="font-semibold text-gray-200 cursor-pointer hover:text-blue-300"
                                                                        onClick={() => fetchStudentFeedbacks(feedback.student_id)}>
                                                                        {feedback.student_name}
                                                                    </div>
                                                                    <div className="text-xs text-gray-400">ID: {feedback.student_id}</div>
                                                                    <div className="text-xs text-gray-500 mt-1">{feedback.class_name}</div>
                                                                </div>
                                                            </div>
                                                        </td>
                                                        <td className="p-4">
                                                            <div className="max-w-md">
                                                                <p className="text-gray-300">
                                                                    {truncateText(feedback.feedback_text, 120)}
                                                                </p>
                                                                {feedback.feedback_type === 'voice' && (
                                                                    <div className="mt-2 flex items-center text-sm text-gray-400">
                                                                        <span className="mr-2">🎤 Voice</span>
                                                                        {feedback.audio_duration && (
                                                                            <span className="mr-3">{feedback.audio_duration.toFixed(1)}s</span>
                                                                        )}
                                                                        {feedback.confidence && (
                                                                            <span className={`px-2 py-1 rounded ${feedback.confidence >= 0.7 ? 'bg-green-900/30 text-green-400' : 'bg-yellow-900/30 text-yellow-400'}`}>
                                                                                {Math.round(feedback.confidence * 100)}% confidence
                                                                            </span>
                                                                        )}
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </td>
                                                        <td className="p-4">
                                                            <div className="space-y-2">
                                                                <div className="flex items-center">
                                                                    <div className={`w-8 h-8 ${typeStyle.bg} rounded-full flex items-center justify-center text-lg mr-2 border ${typeStyle.border}`}>
                                                                        {typeStyle.icon}
                                                                    </div>
                                                                    <span className={`font-medium ${typeStyle.text}`}>
                                                                        {typeStyle.label}
                                                                    </span>
                                                                </div>
                                                            </div>
                                                        </td>
                                                        <td className="p-4">
                                                            <div className="text-sm text-gray-400">
                                                                {formatTime(feedback.created_at)}
                                                            </div>
                                                            <div className="text-xs text-gray-500">
                                                                {formatDate(feedback.created_at)}
                                                            </div>
                                                        </td>
                                                    </tr>
                                                )
                                            })}
                                        </tbody>
                                    </table>
                                </div>

                                {/* Pagination Controls */}
                                {totalPages > 1 && (
                                    <div className="mt-6 flex justify-center items-center space-x-2">
                                        <button
                                            onClick={() => handlePageChange(currentPage - 1)}
                                            disabled={currentPage === 1}
                                            className={`px-4 py-2 rounded-lg transition duration-200 ${currentPage === 1
                                                ? 'bg-gray-800 text-gray-500 cursor-not-allowed'
                                                : 'bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white'
                                                }`}
                                        >
                                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                                            </svg>
                                        </button>

                                        {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                                            let pageNumber
                                            if (totalPages <= 5) {
                                                pageNumber = i + 1
                                            } else if (currentPage <= 3) {
                                                pageNumber = i + 1
                                            } else if (currentPage >= totalPages - 2) {
                                                pageNumber = totalPages - 4 + i
                                            } else {
                                                pageNumber = currentPage - 2 + i
                                            }

                                            return (
                                                <button
                                                    key={pageNumber}
                                                    onClick={() => handlePageChange(pageNumber)}
                                                    className={`px-4 py-2 rounded-lg transition duration-200 ${currentPage === pageNumber
                                                        ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold'
                                                        : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                                                        }`}
                                                >
                                                    {pageNumber}
                                                </button>
                                            )
                                        })}

                                        <button
                                            onClick={() => handlePageChange(currentPage + 1)}
                                            disabled={currentPage === totalPages}
                                            className={`px-4 py-2 rounded-lg transition duration-200 ${currentPage === totalPages
                                                ? 'bg-gray-800 text-gray-500 cursor-not-allowed'
                                                : 'bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white'
                                                }`}
                                        >
                                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                            </svg>
                                        </button>

                                        <span className="ml-4 text-sm text-gray-400">
                                            Trang {currentPage} của {totalPages}
                                        </span>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                </div>

                {/* Student Feedbacks Modal */}
                {selectedStudent && studentFeedbacks.length > 0 && (
                    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
                        <div className="bg-gray-900 rounded-2xl shadow-2xl max-w-4xl w-full max-h-[80vh] overflow-hidden">
                            <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-6">
                                <div className="flex justify-between items-center">
                                    <h3 className="text-2xl font-bold text-white">
                                        Tất cả feedbacks của: {studentFeedbacks[0]?.student_name}
                                    </h3>
                                    <button
                                        onClick={() => {
                                            setSelectedStudent(null)
                                            setStudentFeedbacks([])
                                        }}
                                        className="text-gray-400 hover:text-white"
                                    >
                                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                        </svg>
                                    </button>
                                </div>
                            </div>
                            <div className="p-6 overflow-y-auto max-h-[60vh]">
                                <div className="space-y-4">
                                    {studentFeedbacks.map((feedback, index) => {
                                        const emotionStyle = getEmotionColor(feedback.emotion)
                                        const typeStyle = getFeedbackTypeColor(feedback.feedback_type)
                                        
                                        return (
                                            <div key={index} className="bg-gray-800/50 p-4 rounded-xl border border-gray-700">
                                                <div className="flex justify-between items-start mb-2">
                                                    <div className="flex items-center">
                                                        <div className={`w-8 h-8 ${typeStyle.bg} rounded-full flex items-center justify-center text-lg mr-2 border ${typeStyle.border}`}>
                                                            {typeStyle.icon}
                                                        </div>
                                                        <div>
                                                            <div className={`text-sm ${typeStyle.text}`}>
                                                                {typeStyle.label}
                                                            </div>
                                                            <div className="text-xs text-gray-400">
                                                                {formatDate(feedback.created_at)}
                                                            </div>
                                                        </div>
                                                    </div>
                                                    <div className="flex items-center">
                                                        <div className={`w-8 h-8 ${emotionStyle.bg} rounded-full flex items-center justify-center text-lg mr-2 border ${emotionStyle.border}`}>
                                                            {emotionStyle.icon}
                                                        </div>
                                                        <div className={`text-xl ${getRatingColor(feedback.rating)}`}>
                                                            {getRatingStars(feedback.rating)}
                                                        </div>
                                                    </div>
                                                </div>
                                                <div className="mt-3">
                                                    <p className="text-gray-300">
                                                        {feedback.feedback_text || feedback.transcribed_text}
                                                    </p>
                                                    {feedback.feedback_type === 'voice' && (
                                                        <div className="mt-2 text-sm text-gray-400 flex items-center">
                                                            <span className="mr-4">🎤 Voice ({feedback.audio_duration?.toFixed(1)}s)</span>
                                                            {feedback.confidence && (
                                                                <span className={`px-2 py-1 rounded ${feedback.confidence >= 0.7 ? 'bg-green-900/30 text-green-400' : 'bg-yellow-900/30 text-yellow-400'}`}>
                                                                    {Math.round(feedback.confidence * 100)}% confidence
                                                                </span>
                                                            )}
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        )
                                    })}
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}