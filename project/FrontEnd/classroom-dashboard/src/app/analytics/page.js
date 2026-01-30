// src/app/analytics/page.js
'use client'
import { useEffect, useRef, useState } from 'react'
import { buildApiUrl, buildWebSocketUrl } from '../config/api'

export default function AnalyticsPage() {
    const [studentsData, setStudentsData] = useState([])
    const [focusData, setFocusData] = useState({})
    const [dominantEmotion, setDominantEmotion] = useState({})
    const [dominantBehavior, setDominantBehavior] = useState({})
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [lastUpdate, setLastUpdate] = useState(null)
    const [newDataCount, setNewDataCount] = useState(0)
    const [connectionStatus, setConnectionStatus] = useState('connecting')
    const [currentPage, setCurrentPage] = useState(1)
    const [selectedStudent, setSelectedStudent] = useState(null)
    const [studentDetails, setStudentDetails] = useState([])
    const [batchProcessStatus, setBatchProcessStatus] = useState({
        active: false,
        lastProcessed: null,
        successCount: 0
    })
    const itemsPerPage = 10

    const socketRef = useRef(null)
    const pollingIntervalRef = useRef(null)
    const previousDataRef = useRef([])
    const batchPollingRef = useRef(null)

    useEffect(() => {
        // Lấy data lần đầu từ batch-process
        fetchBatchProcessData()

        // Khởi tạo WebSocket connection
        initWebSocket()

        // Khởi tạo polling cho batch data (mỗi 10 giây)
        initBatchPolling()

        // Cleanup khi component unmount
        return () => {
            cleanupConnections()
        }
    }, [])

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
                console.log('✅ WebSocket connected')
                setConnectionStatus('connected')
            }

            socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data)
                    console.log('📡 WebSocket message:', data.type || 'unknown')

                    // Xử lý các loại message
                    switch (data.type) {
                        case 'student_data_update':
                            console.log('📊 Student data update:', data.data?.student_name)
                            handleStudentDataUpdate(data.data)
                            setNewDataCount(prev => prev + 1)
                            fetchBatchProcessData()
                            break

                        case 'attendance_update':
                            console.log('✅ Attendance update:', data.data?.student_name)
                            handleAttendanceUpdate(data.data)
                            setNewDataCount(prev => prev + 1)
                            fetchBatchProcessData()
                            break

                        case 'emotion_update':
                            console.log('😊 Emotion update:', data.data?.student_name, data.data?.emotion)
                            handleEmotionUpdate(data.data)
                            setNewDataCount(prev => prev + 1)
                            fetchBatchProcessData()
                            break

                        case 'behavior_update':
                            console.log('👥 Behavior update:', data.data?.student_name)
                            handleBehaviorUpdate(data.data)
                            setNewDataCount(prev => prev + 1)
                            fetchBatchProcessData()
                            break

                        case 'focus_update':
                            console.log('🎯 Focus update:', data.data?.student_name, data.data?.focus_score)
                            processFocusUpdate(data.data)
                            setNewDataCount(prev => prev + 1)
                            fetchBatchProcessData()
                            break

                        case 'engagement_update':
                            console.log('🧠 Engagement update:', data.data?.student_name)
                            processFocusUpdate(data.data)
                            setNewDataCount(prev => prev + 1)
                            fetchBatchProcessData()
                            break

                        case 'checkout_update':
                            console.log('🚪 Check-out:', data.data?.student_name)
                            handleCheckoutUpdate(data.data)
                            setNewDataCount(prev => prev + 1)
                            fetchBatchProcessData()
                            break

                        case 'batch_processed':
                            console.log('🔄 Batch processed:', data.processed_count, 'items')
                            handleBatchProcessed(data)
                            setBatchProcessStatus(prev => ({
                                ...prev,
                                lastProcessed: new Date(),
                                successCount: data.processed_count
                            }))
                            setNewDataCount(prev => prev + 1)
                            fetchBatchProcessData() // Lấy dữ liệu mới sau khi batch được xử lý
                            break

                        case 'database_reset':
                            console.log('🔄 Database reset notification')
                            handleDatabaseReset()
                            fetchBatchProcessData()
                            break

                        case 'ai_detection':
                            console.log('🤖 AI detection:', data.data?.student_name)
                            handleAIDetection(data.data)
                            setNewDataCount(prev => prev + 1)
                            fetchBatchProcessData()
                            break

                        case 'unknown_student_filtered':
                            console.log('🚫 Unknown student filtered:', data.data?.student_name)
                            handleUnknownFiltered(data.data)
                            break

                        case 'realtime_analysis':
                            console.log('📈 Realtime analysis:', data.timestamp)
                            handleRealtimeAnalysis(data.data)
                            break

                        case 'class_statistics':
                            console.log('📊 Class statistics update')
                            handleClassStatistics(data.data)
                            break

                        case 'system_status':
                            console.log('⚙️ System status:', data.status)
                            handleSystemStatus(data.data)
                            break

                        case 'error_notification':
                            console.error('❌ Server error:', data.message)
                            handleServerError(data)
                            break

                        case 'heartbeat':
                            console.log('❤️ Heartbeat from server')
                            socket.send(JSON.stringify({ type: 'heartbeat_ack', timestamp: new Date().toISOString() }))
                            break

                        default:
                            console.log('📨 Unknown WebSocket message type:', data.type || 'unknown', data)
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

    // Các hàm xử lý cho từng loại message
    const handleStudentDataUpdate = (data) => {
        console.log('Processing student data update:', data)
    }

    const handleAttendanceUpdate = (data) => {
        console.log('Processing attendance update:', data)
    }

    const handleEmotionUpdate = (data) => {
        console.log('Processing emotion update:', data)
    }

    const handleBehaviorUpdate = (data) => {
        console.log('Processing behavior update:', data)
    }

    const handleCheckoutUpdate = (data) => {
        console.log('Processing checkout:', data)
    }

    const handleBatchProcessed = (data) => {
        console.log('Batch processed:', data)
        if (data.success_count > 0) {
            console.log(`✅ ${data.success_count} items processed successfully`)
            // Có thể hiển thị toast notification ở đây
        }
    }

    const handleDatabaseReset = () => {
        console.log('Database was reset, refreshing all data...')
        fetchBatchProcessData(true)
    }

    const handleAIDetection = (data) => {
        console.log('AI detection:', data)
    }

    const handleUnknownFiltered = (data) => {
        console.log('Unknown student filtered:', data)
    }

    const handleRealtimeAnalysis = (data) => {
        console.log('Realtime analysis:', data)
    }

    const handleClassStatistics = (data) => {
        console.log('Class statistics:', data)
    }

    const handleSystemStatus = (data) => {
        console.log('System status:', data)
    }

    const handleServerError = (data) => {
        console.error('Server error:', data)
    }

    const initBatchPolling = () => {
        // Polling cho batch data mỗi 10 giây
        batchPollingRef.current = setInterval(() => {
            console.log('🔄 Auto-refreshing batch data...')
            fetchBatchProcessData()
        }, 10000) // 10 giây
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

        if (batchPollingRef.current) {
            clearInterval(batchPollingRef.current)
            batchPollingRef.current = null
        }
    }

    const fetchBatchProcessData = async (forceRefresh = false) => {
        try {
            setError(null)
            setLoading(true)
            console.log('🔄 Fetching batch process data...')

            // Gọi GET endpoint của batch-process để lấy dữ liệu mới nhất
            const response = await fetch(buildApiUrl('/api/ai/batch-process?method=GET'))

            if (!response.ok) {
                throw new Error(`Batch-process API Error: ${response.status}`)
            }

            const data = await response.json()
            console.log('📦 Batch-process API response:', {
                status: data.status,
                method: data.method,
                recent_count: data.recent_ai_data?.length || 0
            })

            if (data.status === 'success' && data.recent_ai_data && data.recent_ai_data.length > 0) {
                console.log(`✅ Processing ${data.recent_ai_data.length} recent AI records`)
                processBatchData(data.recent_ai_data)

                // Cập nhật thông tin batch process
                setBatchProcessStatus(prev => ({
                    ...prev,
                    active: true,
                    lastProcessed: new Date(),
                    stats: {
                        total_records: data.stats?.total_ai_records || 0,
                        today_records: data.stats?.today_ai_records || 0
                    }
                }))
            } else {
                // Fallback: thử lấy dữ liệu từ student-data
                console.log('📭 No batch data, trying student-data...')
                await fetchStudentData()
            }

        } catch (error) {
            console.error('❌ Error fetching batch process data:', error)
            setError(error.message || 'Failed to load batch data')
            await fetchStudentData() // Fallback
        } finally {
            setLoading(false)
        }
    }

    const processBatchData = (batchArray) => {
        if (!batchArray || batchArray.length === 0) {
            console.log('⚠️ No batch data to process')
            return
        }

        console.log(`📊 Processing ${batchArray.length} batch records...`)

        setStudentsData(prev => {
            // Chuyển đổi batch data
            const newStudents = batchArray
                .map((record, index) => ({
                    student_id: record.student_id || `AI_${index}`,
                    student_name: record.student_name || 'AI Student',
                    focus_score: record.focus_score || record.engagement_score || 0,
                    concentration_level: record.concentration_level ||
                        ((record.focus_score || 75) >= 80 ? 'high' :
                            (record.focus_score || 75) >= 60 ? 'medium' : 'low'),
                    emotion: record.emotion || 'unknown',
                    emotion_confidence: record.emotion_confidence || 0.5,
                    behavior_type: record.behavior_type || 'unknown',
                    behavior_details: record.behavior_details || record.details || 'AI detected',
                    recorded_at: record.recorded_at || new Date().toISOString(),
                    class_name: record.class_name || 'AI Classroom',
                    isNew: isRecordRecent(record.recorded_at),
                    data_source: 'batch_process'
                }))
                .filter(student => student.student_name && student.student_name !== 'Unknown Student')

            // Merge với dữ liệu hiện tại (UPDATE khi trùng tên)
            const existingMap = new Map()

            // Thêm tất cả học sinh hiện tại vào map
            prev.forEach(student => {
                const key = student.student_name
                if (key && !existingMap.has(key)) {
                    existingMap.set(key, student)
                }
            })

            // Cập nhật hoặc thêm học sinh mới
            newStudents.forEach(newStudent => {
                const key = newStudent.student_name
                if (key) {
                    if (existingMap.has(key)) {
                        // Cập nhật dữ liệu cũ với dữ liệu mới
                        const oldStudent = existingMap.get(key)

                        // Chỉ cập nhật nếu dữ liệu mới hơn
                        const oldTime = new Date(oldStudent.recorded_at || 0)
                        const newTime = new Date(newStudent.recorded_at || 0)

                        if (newTime > oldTime) {
                            console.log(`🔄 Updating ${key} with newer data`)
                            existingMap.set(key, {
                                ...oldStudent,
                                ...newStudent,
                                isNew: true
                            })
                        }
                    } else {
                        // Thêm học sinh mới
                        existingMap.set(key, newStudent)
                    }
                }
            })

            // Chuyển map thành array và sắp xếp
            const mergedStudents = Array.from(existingMap.values())
                .sort((a, b) => new Date(b.recorded_at) - new Date(a.recorded_at))
                .slice(0, 100)

            console.log(`✅ Merged: ${prev.length} -> ${mergedStudents.length} students`)

            return mergedStudents
        })

        setLastUpdate(new Date())
        setCurrentPage(1)
        calculateAnalytics(studentsData)
    }


    const fetchStudentData = async () => {
        try {
            console.log('🔄 Trying /api/student-data endpoint...')
            const response = await fetch(buildApiUrl('/api/student-data?limit=50&sort=desc&recent_minutes=30'))

            if (!response.ok) {
                throw new Error(`Student-data API Error: ${response.status}`)
            }

            const data = await response.json()
            console.log('👨‍🎓 Student-data API response:', {
                status: data.status,
                count: data.count || 0
            })

            if (data.status === 'success' && data.student_data && data.student_data.length > 0) {
                console.log(`✅ Found ${data.student_data.length} student records`)
                processStudentData(data.student_data)
            } else {
                console.log('📭 No student data, using fallback...')
                useFallbackData()
            }

        } catch (error) {
            console.error('❌ Error fetching student data:', error)
            useFallbackData()
        }
    }

    const processStudentData = (studentArray) => {
        if (!studentArray || studentArray.length === 0) {
            console.log('⚠️ No student data to process')
            return
        }

        console.log(`📊 Processing ${studentArray.length} student records...`)

        // Chuyển đổi student data sang format focus data
        const processedStudents = studentArray
            .map((record, index) => {
                // Map các field từ student_data
                return {
                    student_id: record.student_id || `RECORD_${index}`,
                    student_name: record.student_name || 'Student',
                    focus_score: record.focus_score || record.behavior_score || 70,
                    concentration_level: record.concentration_level ||
                        ((record.focus_score || 70) >= 80 ? 'high' :
                            (record.focus_score || 70) >= 60 ? 'medium' : 'low'),
                    emotion: record.emotion || 'neutral',
                    emotion_confidence: record.emotion_confidence || 0.5,
                    behavior_type: record.behavior_type || 'normal',
                    behavior_details: record.behavior_details || '',
                    recorded_at: record.recorded_at || new Date().toISOString(),
                    class_name: record.class_name || 'Classroom',
                    isNew: isRecordRecent(record.recorded_at),
                    data_source: 'student_data'
                }
            })
            .filter(student => student.student_name && !student.student_name.toLowerCase().includes('unknown'))
            .slice(0, 100) // Giới hạn 100 bản ghi

        console.log(`✅ Converted ${processedStudents.length} student records`)

        setStudentsData(processedStudents)
        setLastUpdate(new Date())
        setCurrentPage(1)

        // Tính toán analytics
        calculateAnalytics(processedStudents)
    }

    const useFallbackData = () => {
        console.log('🔄 Using fallback data...')
        const fallbackData = getFallbackData()
            .sort((a, b) => new Date(b.recorded_at) - new Date(a.recorded_at))
        setStudentsData(fallbackData)
        calculateAnalytics(fallbackData)
        setLastUpdate(new Date())
        setCurrentPage(1)
    }

    const getFallbackData = () => {
        const now = new Date()
        return [
            {
                student_id: 'SV001',
                student_name: 'Nguyễn Văn A',
                class_name: 'Lớp 10A1',
                focus_score: 85.0,
                concentration_level: 'high',
                focus_duration: 45.5,
                emotion: 'happy',
                emotion_confidence: 0.85,
                behavior_type: 'writing',
                behavior_details: 'Đang viết bài tập',
                recorded_at: new Date(now.getTime() - 5000).toISOString(),
                isNew: true,
                data_source: 'batch_fallback'
            },
            {
                student_id: 'SV002',
                student_name: 'Trần Thị B',
                class_name: 'Lớp 10A1',
                focus_score: 72.5,
                concentration_level: 'medium',
                focus_duration: 38.0,
                emotion: 'neutral',
                emotion_confidence: 0.72,
                behavior_type: 'participation',
                behavior_details: 'Phát biểu xây dựng bài',
                recorded_at: new Date(now.getTime() - 10000).toISOString(),
                isNew: true,
                data_source: 'batch_fallback'
            },
            {
                student_id: 'SV003',
                student_name: 'Lê Văn C',
                class_name: 'Lớp 10A1',
                focus_score: 60.0,
                concentration_level: 'low',
                focus_duration: 25.5,
                emotion: 'sad',
                emotion_confidence: 0.65,
                behavior_type: 'discipline',
                behavior_details: 'Thỉnh thoảng mất tập trung',
                recorded_at: new Date(now.getTime() - 15000).toISOString(),
                isNew: true,
                data_source: 'batch_fallback'
            },
            {
                student_id: 'AI_001',
                student_name: 'Nam',
                class_name: 'AI Classroom',
                focus_score: 82.5,
                concentration_level: 'high',
                focus_duration: 45.0,
                emotion: 'neutral',
                emotion_confidence: 0.75,
                behavior_type: 'engagement',
                behavior_details: 'AI detected engagement',
                recorded_at: new Date(now.getTime() - 20000).toISOString(),
                isNew: false,
                data_source: 'batch_fallback'
            },
            {
                student_id: 'AI_002',
                student_name: 'Student 1',
                class_name: 'AI Classroom',
                focus_score: 78.3,
                concentration_level: 'medium',
                focus_duration: 40.0,
                emotion: 'happy',
                emotion_confidence: 0.85,
                behavior_type: 'normal',
                behavior_details: 'AI detected normal behavior',
                recorded_at: new Date(now.getTime() - 25000).toISOString(),
                isNew: false,
                data_source: 'batch_fallback'
            }
        ]
    }

    const calculateAnalytics = (studentArray) => {
        if (!studentArray || studentArray.length === 0) {
            console.log('⚠️ No students data for analytics')
            return
        }

        console.log(`📈 Calculating analytics for ${studentArray.length} students...`)

        // Tính toán dominant emotion
        const emotionCount = {}
        studentArray.forEach(student => {
            const emotion = student.emotion || 'unknown'
            emotionCount[emotion] = (emotionCount[emotion] || 0) + 1
        })

        const dominantEmotionEntry = Object.entries(emotionCount)
            .sort((a, b) => b[1] - a[1])[0] || ['neutral', 0]

        // Tính toán dominant behavior
        const behaviorCount = {}
        studentArray.forEach(student => {
            const behavior = student.behavior_type || 'unknown'
            behaviorCount[behavior] = (behaviorCount[behavior] || 0) + 1
        })

        const dominantBehaviorEntry = Object.entries(behaviorCount)
            .sort((a, b) => b[1] - a[1])[0] || ['normal', 0]

        // Tính average focus
        const validFocusScores = studentArray
            .filter(s => typeof s.focus_score === 'number' && !isNaN(s.focus_score))
            .map(s => s.focus_score)

        const totalFocus = validFocusScores.reduce((sum, score) => sum + score, 0)
        const avgFocus = validFocusScores.length > 0 ? totalFocus / validFocusScores.length : 0

        // Tính tổng focus duration
        const totalDuration = studentArray.reduce((sum, student) => {
            const duration = parseFloat(student.focus_duration) || 0
            return sum + duration
        }, 0)

        console.log('📊 Analytics results:', {
            avgFocus: avgFocus.toFixed(1),
            totalStudents: studentArray.length,
            dominantEmotion: dominantEmotionEntry,
            dominantBehavior: dominantBehaviorEntry
        })

        // Cập nhật state
        setFocusData({
            avg_focus: avgFocus,
            total_students: studentArray.length,
            total_duration: totalDuration
        })

        setDominantEmotion({
            emotion: dominantEmotionEntry[0],
            count: dominantEmotionEntry[1],
            percentage: Math.round((dominantEmotionEntry[1] / Math.max(studentArray.length, 1)) * 100) || 0
        })

        setDominantBehavior({
            behavior: dominantBehaviorEntry[0],
            count: dominantBehaviorEntry[1],
            percentage: Math.round((dominantBehaviorEntry[1] / Math.max(studentArray.length, 1)) * 100) || 0
        })

        // Lưu data hiện tại để so sánh lần sau
        previousDataRef.current = studentArray
    }

    const fetchStudentDetails = async (studentId) => {
        try {
            console.log(`🔍 Fetching details for student: ${studentId}`)
            // Dùng batch-process endpoint để lấy chi tiết
            const response = await fetch(buildApiUrl('/api/ai/batch-process'))
            if (response.ok) {
                const data = await response.json()
                if (data.status === 'success' && data.recent_ai_data) {
                    // Lọc ra records của student cụ thể
                    const studentRecords = data.recent_ai_data.filter(
                        record => record.student_id === studentId
                    )

                    console.log(`✅ Found ${studentRecords.length} batch records for student ${studentId}`)

                    // Process data để có cấu trúc thống nhất
                    const processedDetails = studentRecords
                        .map(record => ({
                            student_id: record.student_id,
                            student_name: record.student_name,
                            focus_score: record.focus_score,
                            concentration_level: record.concentration_level,
                            emotion: record.emotion,
                            emotion_confidence: record.emotion_confidence,
                            behavior_type: record.behavior_type,
                            behavior_details: record.behavior_details,
                            focus_duration: record.focus_duration,
                            recorded_at: record.recorded_at,
                            class_name: record.class_name
                        }))
                        .sort((a, b) => new Date(b.recorded_at) - new Date(a.recorded_at))

                    setStudentDetails(processedDetails)
                    setSelectedStudent(studentId)
                }
            }
        } catch (error) {
            console.error('❌ Error fetching student details:', error)
        }
    }

    const processFocusUpdate = (focusData) => {
        if (!focusData) return

        setStudentsData(prev => {
            const newStudent = {
                student_id: focusData.student_id || `FOCUS_${Date.now()}`,
                student_name: focusData.student_name || 'Unknown Student',
                focus_score: focusData.focus_score || focusData.engagement_score || 75,
                concentration_level: focusData.concentration_level || 'medium',
                emotion: focusData.emotion || 'neutral',
                emotion_confidence: focusData.emotion_confidence || 0.5,
                behavior_type: focusData.behavior_type || focusData.behavior || 'normal',
                behavior_details: focusData.behavior_details || '',
                recorded_at: new Date().toISOString(),
                class_name: focusData.class_name || 'N/A',
                isNew: true,
                data_source: 'websocket_update'
            }

            console.log(`🔄 Processing update for: ${newStudent.student_name}`)

            // Tìm học sinh hiện có cùng tên
            const existingStudentIndex = prev.findIndex(student =>
                student.student_name === newStudent.student_name
            )

            let updatedStudents = [...prev]

            if (existingStudentIndex !== -1) {
                console.log(`✅ Replacing old data for ${newStudent.student_name}`)
                // THAY THẾ dữ liệu cũ bằng dữ liệu mới (giữ vị trí)
                updatedStudents[existingStudentIndex] = {
                    ...updatedStudents[existingStudentIndex], // Giữ các thuộc tính cũ không bị mất
                    ...newStudent, // Cập nhật với dữ liệu mới
                    isNew: true // Đánh dấu là mới
                }
            } else {
                console.log(`➕ Adding new student: ${newStudent.student_name}`)
                // Thêm học sinh mới vào đầu
                updatedStudents = [newStudent, ...prev]
            }

            // Sắp xếp theo thời gian mới nhất
            const sortedStudents = updatedStudents
                .sort((a, b) => new Date(b.recorded_at) - new Date(a.recorded_at))
                .slice(0, 50) // Giới hạn 50 bản ghi

            return sortedStudents
        })

        setLastUpdate(new Date())
        setNewDataCount(prev => prev + 1)
    }

    // Các hàm helper cho UI (giữ nguyên từ code cũ)
    const getEmotionColor = (emotion) => {
        const colors = {
            happy: { text: 'text-green-400', bg: 'bg-green-900/30', border: 'border-green-800/50', icon: '😊', label: 'Vui vẻ' },
            neutral: { text: 'text-blue-400', bg: 'bg-blue-900/30', border: 'border-blue-800/50', icon: '😐', label: 'Bình thường' },
            sad: { text: 'text-red-400', bg: 'bg-red-900/30', border: 'border-red-800/50', icon: '😢', label: 'Buồn' },
            excited: { text: 'text-yellow-400', bg: 'bg-yellow-900/30', border: 'border-yellow-800/50', icon: '🤩', label: 'Hào hứng' },
            surprised: { text: 'text-purple-400', bg: 'bg-purple-900/30', border: 'border-purple-800/50', icon: '😲', label: 'Ngạc nhiên' },
            angry: { text: 'text-orange-400', bg: 'bg-orange-900/30', border: 'border-orange-800/50', icon: '😠', label: 'Tức giận' },
            distracted: { text: 'text-gray-400', bg: 'bg-gray-900/30', border: 'border-gray-800/50', icon: '😐', label: 'Mất tập trung' },
            fear: { text: 'text-pink-400', bg: 'bg-pink-900/30', border: 'border-pink-800/50', icon: '😨', label: 'Sợ hãi' },
            disgust: { text: 'text-yellow-600', bg: 'bg-yellow-900/30', border: 'border-yellow-800/50', icon: '🤢', label: 'Ghê tởm' },
            unknown: { text: 'text-gray-400', bg: 'bg-gray-900/30', border: 'border-gray-800/50', icon: '❓', label: 'Chưa xác định' }
        }
        return colors[emotion] || colors.unknown
    }

    const getBehaviorColor = (behavior) => {
        const colors = {
            engagement: { text: 'text-green-400', bg: 'bg-green-900/20', border: 'border-green-700/30', icon: '💪', label: 'Tham gia' },
            participation: { text: 'text-blue-400', bg: 'bg-blue-900/20', border: 'border-blue-700/30', icon: '🗣️', label: 'Phát biểu' },
            raising_one_hand: { text: 'text-green-500', bg: 'bg-green-900/25', border: 'border-green-600/40', icon: '✋', label: 'Giơ tay' },
            writing: { text: 'text-emerald-400', bg: 'bg-emerald-900/20', border: 'border-emerald-700/30', icon: '✍️', label: 'Đang viết' },
            look_straight: { text: 'text-teal-400', bg: 'bg-teal-900/20', border: 'border-teal-700/30', icon: '👁️', label: 'Nhìn thẳng' },
            look_around: { text: 'text-yellow-400', bg: 'bg-yellow-900/20', border: 'border-yellow-700/30', icon: '👀', label: 'Nhìn quanh' },
            discipline: { text: 'text-indigo-400', bg: 'bg-indigo-900/20', border: 'border-indigo-700/30', icon: '📚', label: 'Kỷ luật' },
            normal: { text: 'text-gray-400', bg: 'bg-gray-900/20', border: 'border-gray-700/30', icon: '👍', label: 'Bình thường' },
            unknown: { text: 'text-gray-400', bg: 'bg-gray-900/20', border: 'border-gray-700/30', icon: '❓', label: 'Chưa xác định' }
        }

        const behaviorKey = Object.keys(colors).find(key =>
            behavior.toLowerCase().includes(key.toLowerCase()) ||
            key.toLowerCase().includes(behavior.toLowerCase())
        )

        return colors[behaviorKey] || colors.unknown
    }

    const getFocusColor = (score) => {
        if (!score && score !== 0) return {
            text: 'text-gray-400',
            bg: 'bg-gray-900/20',
            border: 'border-gray-700/30',
            label: 'N/A',
            icon: '❓'
        }

        if (score >= 50) return {
            text: 'text-green-400',
            bg: 'bg-green-900/20',
            border: 'border-green-700/30',
            label: 'Tập trung',
            icon: '✅'
        }
        return {
            text: 'text-red-400',
            bg: 'bg-red-900/20',
            border: 'border-red-700/30',
            label: 'Không tập trung',
            icon: '❌'
        }
    }

    const getConcentrationColor = (level) => {
        const colors = {
            very_high: { text: 'text-green-400', bg: 'bg-green-900/20', border: 'border-green-700/30', icon: '🚀', label: 'Rất cao' },
            high: { text: 'text-green-300', bg: 'bg-green-900/15', border: 'border-green-700/25', icon: '⭐', label: 'Cao' },
            medium: { text: 'text-yellow-400', bg: 'bg-yellow-900/20', border: 'border-yellow-700/30', icon: '📊', label: 'Trung bình' },
            low: { text: 'text-orange-400', bg: 'bg-orange-900/20', border: 'border-orange-700/30', icon: '📉', label: 'Thấp' },
            very_low: { text: 'text-red-400', bg: 'bg-red-900/20', border: 'border-red-700/30', icon: '⚠️', label: 'Rất thấp' },
            unknown: { text: 'text-gray-400', bg: 'bg-gray-900/20', border: 'border-gray-700/30', icon: '❓', label: 'Chưa xác định' }
        }
        return colors[level] || colors.unknown
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
    const totalPages = Math.ceil(studentsData.length / itemsPerPage)
    const startIndex = (currentPage - 1) * itemsPerPage
    const endIndex = startIndex + itemsPerPage
    const currentStudents = studentsData.slice(startIndex, endIndex)

    const handlePageChange = (pageNumber) => {
        setCurrentPage(pageNumber)
        const tableElement = document.querySelector('.students-table')
        if (tableElement) {
            tableElement.scrollIntoView({ behavior: 'smooth' })
        }
    }

    const handleRefresh = () => {
        setLoading(true)
        setNewDataCount(0)
        fetchBatchProcessData()
    }

    if (loading && studentsData.length === 0) {
        return (
            <div className="analytics-page flex justify-center items-center min-h-screen bg-black">
                <div className="text-center">
                    <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500 mb-4"></div>
                    <span className="text-lg text-white block">Loading Batch Process Analytics...</span>
                    <p className="text-gray-400 mt-2">Fetching data from Batch-Process API...</p>
                    <div className="mt-4 flex justify-center items-center">
                        <div className={`w-3 h-3 rounded-full ${getConnectionStatusColor()} mr-2`}></div>
                        <span className="text-sm text-gray-400">{getConnectionStatusText()}</span>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="analytics-page bg-[#B39858] p-6 min-h-screen">
            <div className="max-w-7xl mx-auto">
                {/* Header với status */}
                <div className="text-center mb-8">
                    <h1 className="text-4xl font-bold text-white mb-3">Analytics Dashboard</h1>
                    <div className="flex justify-center items-center space-x-4 mb-2">
                        <p className="text-blue-100 text-lg">Real-time data from AI Batch Process</p>
                    </div>
                </div>

                {/* Students Table - Focus Data */}
                <div className="bg-gray-900 rounded-2xl shadow-xl overflow-hidden border border-gray-800 mb-8 students-table">
                    <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-6">
                        <div className="flex justify-between items-center">
                            <h2 className="text-2xl font-bold text-white flex items-center">
                                <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                                </svg>
                                Student Analytics
                                {newDataCount > 0 && (
                                    <span className="ml-3 px-2 py-1 bg-green-900/50 text-green-400 text-xs rounded-full animate-pulse">
                                        {newDataCount} bản ghi mới
                                    </span>
                                )}
                            </h2>
                            <div className="flex items-center space-x-2">
                                <span className="text-sm text-gray-300">
                                    Hiển thị {startIndex + 1}-{Math.min(endIndex, studentsData.length)} của {studentsData.length} học sinh
                                </span>
                            </div>
                        </div>
                    </div>
                    <div className="p-6">
                        {studentsData.length === 0 ? (
                            <div className="text-center py-8">
                                <p className="text-gray-400">Chưa có dữ liệu từ Batch Process API</p>
                                <button
                                    onClick={handleRefresh}
                                    className="mt-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white py-2 px-4 rounded-lg hover:opacity-90 transition"
                                >
                                    Tải Lại Dữ Liệu
                                </button>
                            </div>
                        ) : (
                            <>
                                <div className="overflow-x-auto">
                                    <table className="w-full min-w-full">
                                        <thead className="bg-gray-800">
                                            <tr>
                                                <th className="text-left p-4 text-gray-300 font-semibold">Học Sinh</th>
                                                <th className="text-left p-4 text-gray-300 font-semibold">Đánh Giá</th>
                                                <th className="text-left p-4 text-gray-300 font-semibold">Cảm Xúc</th>
                                                <th className="text-left p-4 text-gray-300 font-semibold">Hành Vi</th>
                                                <th className="text-left p-4 text-gray-300 font-semibold">Cập Nhật</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-gray-800">
                                            {currentStudents.map((student, index) => {
                                                const emotionStyle = getEmotionColor(student.emotion)
                                                const behaviorStyle = getBehaviorColor(student.behavior_type)
                                                const focusStyle = getFocusColor(student.focus_score)

                                                return (
                                                    <tr
                                                        key={`${student.student_id}_${index}_${student.recorded_at}`}
                                                        className={`hover:bg-gray-800/50 transition duration-150 cursor-pointer ${student.isNew ? 'animate-pulse-once bg-gradient-to-r from-green-900/20 to-emerald-900/10' : ''
                                                            }`}
                                                        onClick={() => fetchStudentDetails(student.student_id)}
                                                    >
                                                        <td className="p-4">
                                                            <div className="flex items-center">
                                                                <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center text-white font-semibold mr-3 border border-blue-700 relative">
                                                                    {student.student_name?.charAt(0) || '?'}
                                                                    {student.isNew && (
                                                                        <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                                                                    )}
                                                                </div>
                                                                <div>
                                                                    <div className="font-semibold text-gray-200">{student.student_name}</div>
                                                                    <div className="text-xs text-gray-400">ID: {student.student_id}</div>
                                                                    {student.isNew && (
                                                                        <div className="text-xs text-green-400 mt-1 animate-pulse">
                                                                            ⚡ Mới cập nhật
                                                                        </div>
                                                                    )}
                                                                </div>
                                                            </div>
                                                        </td>
                                                        <td className="p-4">
                                                            <div className="flex flex-col">
                                                                <div className={`text-xl font-bold ${focusStyle.text} mb-1`}>
                                                                    {student.focus_score >= 50 ? 'Tập trung' : 'Không tập trung'}
                                                                </div>
                                                            </div>
                                                        </td>
                                                        <td className="p-4">
                                                            <div className="flex items-center">
                                                                <div className={`w-8 h-8 ${emotionStyle.bg} rounded-full flex items-center justify-center text-lg mr-2 border ${emotionStyle.border}`}>
                                                                    {emotionStyle.icon}
                                                                </div>
                                                                <div>
                                                                    <div className={`font-medium ${emotionStyle.text}`}>
                                                                        {emotionStyle.label}
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </td>
                                                        <td className="p-4">
                                                            <div className="flex items-center">
                                                                <div className={`w-8 h-8 ${behaviorStyle.bg} rounded-full flex items-center justify-center text-lg mr-2 border ${behaviorStyle.border}`}>
                                                                    {behaviorStyle.icon}
                                                                </div>
                                                                <div>
                                                                    <div className={`font-medium ${behaviorStyle.text}`}>
                                                                        {behaviorStyle.label}
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </td>
                                                        <td className="p-4">
                                                            <div className="text-sm text-gray-400">
                                                                {formatTime(student.recorded_at)}
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

                        {/* Summary Statistics */}
                        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div className="bg-gray-800/50 p-4 rounded-xl border border-gray-700">
                                <h3 className="text-gray-300 font-semibold mb-2">Phân Bố Tập Trung</h3>
                                <div className="space-y-2">
                                    <div className="flex justify-between items-center">
                                        <span className="text-green-400">Tập trung (≥50%)</span>
                                        <span className="text-gray-400">
                                            {studentsData.filter(s => s.focus_score >= 50).length}
                                        </span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="text-red-400">Không tập trung (&lt;50%)</span>
                                        <span className="text-gray-400">
                                            {studentsData.filter(s => s.focus_score < 50).length}
                                        </span>
                                    </div>
                                </div>
                            </div>

                            <div className="bg-gray-800/50 p-4 rounded-xl border border-gray-700">
                                <h3 className="text-gray-300 font-semibold mb-2">Mức Độ Tập Trung</h3>
                                <div className="space-y-2">
                                    {Object.entries(
                                        studentsData.reduce((acc, student) => {
                                            const level = student.concentration_level || 'unknown'
                                            acc[level] = (acc[level] || 0) + 1
                                            return acc
                                        }, {})
                                    )
                                        .sort((a, b) => b[1] - a[1])
                                        .map(([level, count]) => {
                                            const style = getConcentrationColor(level)
                                            return (
                                                <div key={level} className="flex justify-between items-center">
                                                    <div className="flex items-center">
                                                        <span className="mr-2">{style.icon}</span>
                                                        <span className="text-gray-300">{style.label}</span>
                                                    </div>
                                                    <span className="text-gray-400">{count}</span>
                                                </div>
                                            )
                                        })}
                                </div>
                            </div>

                            <div className="bg-gray-800/50 p-4 rounded-xl border border-gray-700">
                                <h3 className="text-gray-300 font-semibold mb-2">Top Học Sinh</h3>
                                <div className="space-y-2">
                                    {studentsData
                                        .sort((a, b) => (b.focus_score || 0) - (a.focus_score || 0))
                                        .slice(0, 3)
                                        .map((student, index) => (
                                            <div key={student.student_id} className="flex justify-between items-center">
                                                <div className="flex items-center">
                                                    <span className="text-yellow-400 mr-2">#{index + 1}</span>
                                                    <span className="text-gray-300 truncate max-w-[100px]">{student.student_name}</span>
                                                </div>
                                                <span className={`font-semibold ${student.focus_score >= 50 ? 'text-green-400' : 'text-red-400'}`}>
                                                    {student.focus_score >= 50 ? 'Tập trung' : 'Không tập trung'}
                                                </span>
                                            </div>
                                        ))}
                                </div>
                            </div>
                        </div>

                        {/* Student Details Modal */}
                        {selectedStudent && studentDetails.length > 0 && (
                            <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
                                <div className="bg-gray-900 rounded-2xl shadow-2xl max-w-4xl w-full max-h-[80vh] overflow-hidden">
                                    <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-6">
                                        <div className="flex justify-between items-center">
                                            <h3 className="text-2xl font-bold text-white">
                                                Chi tiết học sinh: {studentDetails[0]?.student_name}
                                            </h3>
                                            <button
                                                onClick={() => {
                                                    setSelectedStudent(null)
                                                    setStudentDetails([])
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
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                                            <div className="bg-gray-800/50 p-4 rounded-xl">
                                                <h4 className="text-gray-300 font-semibold mb-3">Thống Kê</h4>
                                                <div className="space-y-2">
                                                    <div className="flex justify-between">
                                                        <span className="text-gray-400">Số lần ghi nhận:</span>
                                                        <span className="text-gray-300">{studentDetails.length}</span>
                                                    </div>
                                                    <div className="flex justify-between">
                                                        <span className="text-gray-400">Điểm TB:</span>
                                                        <span className="text-green-400 font-semibold">
                                                            {(studentDetails.reduce((sum, item) => sum + (item.focus_score || 0), 0) / studentDetails.length).toFixed(1)}%
                                                        </span>
                                                    </div>
                                                    <div className="flex justify-between">
                                                        <span className="text-gray-400">Thời gian TB:</span>
                                                        <span className="text-blue-400">
                                                            {(studentDetails.reduce((sum, item) => sum + (item.focus_duration || 0), 0) / studentDetails.length).toFixed(1)} phút
                                                        </span>
                                                    </div>
                                                </div>
                                            </div>
                                            <div className="bg-gray-800/50 p-4 rounded-xl">
                                                <h4 className="text-gray-300 font-semibold mb-3">Phân Bố Cảm Xúc</h4>
                                                <div className="space-y-2">
                                                    {Object.entries(
                                                        studentDetails.reduce((acc, item) => {
                                                            const emotion = item.emotion || 'unknown'
                                                            acc[emotion] = (acc[emotion] || 0) + 1
                                                            return acc
                                                        }, {})
                                                    ).map(([emotion, count]) => {
                                                        const style = getEmotionColor(emotion)
                                                        return (
                                                            <div key={emotion} className="flex justify-between items-center">
                                                                <div className="flex items-center">
                                                                    <span className="mr-2">{style.icon}</span>
                                                                    <span className="text-gray-300">{style.label}</span>
                                                                </div>
                                                                <span className="text-gray-400">{count}</span>
                                                            </div>
                                                        )
                                                    })}
                                                </div>
                                            </div>
                                        </div>
                                        <h4 className="text-gray-300 font-semibold mb-3">Lịch Sử Focus</h4>
                                        <div className="overflow-x-auto">
                                            <table className="w-full">
                                                <thead className="bg-gray-800">
                                                    <tr>
                                                        <th className="p-3 text-gray-300 text-left">Thời Gian</th>
                                                        <th className="p-3 text-gray-300 text-left">Trạng Thái</th>
                                                        <th className="p-3 text-gray-300 text-left">Mức Độ</th>
                                                        <th className="p-3 text-gray-300 text-left">Cảm Xúc</th>
                                                        <th className="p-3 text-gray-300 text-left">Hành Vi</th>
                                                    </tr>
                                                </thead>
                                                <tbody className="divide-y divide-gray-800">
                                                    {studentDetails.slice(0, 10).map((detail, index) => (
                                                        <tr key={index} className="hover:bg-gray-800/50">
                                                            <td className="p-3 text-gray-400">
                                                                {formatTime(detail.recorded_at)}
                                                            </td>
                                                            <td className="p-3">
                                                                <span className={getFocusColor(detail.focus_score).text}>
                                                                    {detail.focus_score >= 50 ? 'Tập trung' : 'Không tập trung'}
                                                                </span>
                                                            </td>
                                                            <td className="p-3">
                                                                <span className={getConcentrationColor(detail.concentration_level).text}>
                                                                    {getConcentrationColor(detail.concentration_level).label}
                                                                </span>
                                                            </td>
                                                            <td className="p-3">
                                                                <span className={getEmotionColor(detail.emotion).text}>
                                                                    {getEmotionColor(detail.emotion).label}
                                                                </span>
                                                            </td>
                                                            <td className="p-3">
                                                                <span className={getBehaviorColor(detail.behavior_type).text}>
                                                                    {getBehaviorColor(detail.behavior_type).label}
                                                                </span>
                                                            </td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}