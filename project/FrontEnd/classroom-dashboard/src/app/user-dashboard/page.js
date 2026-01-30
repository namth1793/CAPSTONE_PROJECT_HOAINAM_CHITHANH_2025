/* eslint-disable @next/next/no-img-element */
// frontend/app/user-dashboard/page.js
'use client'
import { useRouter } from 'next/navigation'
import { useEffect, useRef, useState } from 'react'
import ProtectedRoute from '../components/ProtectedRoute'
import { useAuth } from '../context/AuthContext'
import { buildApiUrl, buildCameraApiUrl, getVideoFeedUrl } from '../config/api'

export default function UserDashboard() {
    // ==================== CAMERA STATE (giống Live Class) ====================
    const [isCameraOn, setIsCameraOn] = useState(false)
    const [cameraError, setCameraError] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const [connectionStatus, setConnectionStatus] = useState('disconnected')
    const [retryCount, setRetryCount] = useState(0)
    const [frameCount, setFrameCount] = useState(0)
    const [streamUrl, setStreamUrl] = useState('')
    const [cameraIndex, setCameraIndex] = useState(0)
    const [availableCameras, setAvailableCameras] = useState([])

    const checkTimeoutRef = useRef(null)
    const frameIntervalRef = useRef(null)

    // Camera API endpoints - sử dụng dynamic URL
    const getCameraApiBase = () => buildCameraApiUrl('/api/camera')
    const getVideoFeed = () => getVideoFeedUrl()

    // ==================== USER DASHBOARD STATE ====================
    const [dashboardData, setDashboardData] = useState(null)
    const [loadingDashboard, setLoadingDashboard] = useState(true)
    const [studentsData, setStudentsData] = useState([])
    const [filteredStudents, setFilteredStudents] = useState([])
    const [searchQuery, setSearchQuery] = useState('')
    const [selectedStudent, setSelectedStudent] = useState(null)
    const [showStudentDetail, setShowStudentDetail] = useState(false)
    const [loadingStudents, setLoadingStudents] = useState(false)
    const [studentDetailData, setStudentDetailData] = useState(null)
    const [loadingDetail, setLoadingDetail] = useState(false)
    const { user, logout } = useAuth()
    const router = useRouter()

    // ==================== HÀM XỬ LÝ TÌM KIẾM ====================
    const handleSearch = (query) => {
        setSearchQuery(query);

        if (!query.trim()) {
            setFilteredStudents(studentsData);
            return;
        }

        const lowerCaseQuery = query.toLowerCase().trim();
        const filtered = studentsData.filter(student =>
            student.name.toLowerCase().includes(lowerCaseQuery) ||
            student.studentId.toLowerCase().includes(lowerCaseQuery) ||
            student.class.toLowerCase().includes(lowerCaseQuery)
        );
        setFilteredStudents(filtered);
    }

    // ==================== FUNCTIONS TO FETCH REAL DATA ====================

    // Hàm fetch dashboard data
    const fetchUserDashboard = async () => {
        try {
            const token = localStorage.getItem('access_token')
            const response = await fetch(buildApiUrl(`/api/user/dashboard?token=${token}`))
            const data = await response.json()

            if (data.status === 'success') {
                setDashboardData(data)
                // Fetch student data sau khi có dashboard data
                fetchStudentData()
            } else if (data.user_type === 'admin' && data.redirect_suggested) {
                router.push(data.suggested_url || '/dashboard')
            }
        } catch (error) {
            console.error('Error fetching user dashboard:', error)
            // Fallback: vẫn fetch student data
            fetchStudentData()
        } finally {
            setLoadingDashboard(false)
        }
    }

    // Hàm fetch danh sách học sinh thực từ API
    const fetchStudentData = async () => {
        try {
            setLoadingStudents(true)
            const token = localStorage.getItem('access_token')
            const response = await fetch(buildApiUrl(`/api/students?token=${token}&limit=20`))
            const data = await response.json()

            if (data.status === 'success') {
                // Format data từ API sang format frontend
                const formattedStudents = data.students.map(student => {
                    // Lấy attendance status từ stats
                    let attendanceStatus = 'Không có dữ liệu'
                    if (student.stats?.attendance) {
                        const { present, absent, late } = student.stats.attendance
                        if (present > 0) attendanceStatus = 'Có mặt'
                        else if (absent > 0) attendanceStatus = 'Vắng mặt'
                        else if (late > 0) attendanceStatus = 'Muộn'
                    }

                    // Lấy emotion từ API emotion data
                    let emotion = student.latest_emotion?.emotion || 'Trung lập'
                    let emotionConfidence = student.latest_emotion?.confidence || 0

                    // Lấy behavior từ API
                    let behavior = student.latest_behavior?.details || 'Không có dữ liệu'
                    let behaviorType = student.latest_behavior?.type || 'unknown'

                    // Lấy focus score từ API
                    let focusScore = student.stats?.avg_focus || 75

                    return {
                        id: student.student_id,
                        name: student.student_name,
                        studentId: student.student_id,
                        class: student.class_name || 'Chưa xác định',
                        attendance: attendanceStatus,
                        emotion: formatEmotion(emotion),
                        emotionConfidence: emotionConfidence,
                        behavior: formatBehavior(behavior),
                        behaviorType: behaviorType,
                        focusScore: Math.round(focusScore),
                        lastSeen: student.last_recorded ?
                            new Date(student.last_recorded).toLocaleTimeString('vi-VN') :
                            'Chưa có dữ liệu'
                    }
                })
                setStudentsData(formattedStudents)
                setFilteredStudents(formattedStudents)
            } else {
                // Fallback nếu API lỗi
                fetchFallbackStudentData()
            }
        } catch (error) {
            console.error('Error fetching student data:', error)
            fetchFallbackStudentData()
        } finally {
            setLoadingStudents(false)
        }
    }

    // Format emotion từ tiếng Anh sang tiếng Việt
    const formatEmotion = (emotion) => {
        const emotionMap = {
            'happy': 'Hạnh phúc',
            'sad': 'Buồn',
            'neutral': 'Trung lập',
            'angry': 'Tức giận',
            'surprised': 'Ngạc nhiên',
            'fearful': 'Sợ hãi',
            'disgusted': 'Khó chịu',
            'surprise': 'Ngạc nhiên',
            'anger': 'Tức giận',
            'fear': 'Sợ hãi',
            'disgust': 'Khó chịu'
        }
        return emotionMap[emotion] || emotion || 'Trung lập'
    }

    // Format behavior từ tiếng Anh sang tiếng Việt
    const formatBehavior = (behavior) => {
        const behaviorMap = {
            'writing': 'Đang viết',
            'look_straight': 'Nhìn thẳng',
            'raising_hand': 'Giơ tay',
            'raising_one_hand': 'Giơ một tay',
            'raising_two_hands': 'Giơ hai tay',
            'normal': 'Bình thường',
            'look_around': 'Nhìn quanh',
            'distracted': 'Mất tập trung',
            'engagement': 'Tham gia',
            'participation': 'Tích cực',
            'discipline': 'Kỷ luật',
            'focus': 'Tập trung'
        }
        return behaviorMap[behavior] || behavior || 'Không xác định'
    }

    // Hàm fetch chi tiết học sinh từ các endpoint chính xác
    const fetchStudentDetailData = async (studentId, studentName) => {
        try {
            setLoadingDetail(true)
            const token = localStorage.getItem('access_token')
            const today = new Date().toISOString().split('T')[0]

            // 1. Lấy thông tin học sinh cơ bản từ class_students
            const basicInfoResponse = await fetch(
                buildApiUrl(`/api/class/students?student_id=${studentId}&token=${token}`)
            )
            const basicInfoData = await basicInfoResponse.json()

            let studentBasicInfo = null
            if (basicInfoData.status === 'success' && basicInfoData.students.length > 0) {
                studentBasicInfo = basicInfoData.students[0]
            }

            // 2. Lấy dữ liệu tổng hợp từ student-data API
            const studentDataResponse = await fetch(
                buildApiUrl(`/api/student-data?token=${token}&student_id=${studentId}&limit=20&recent_minutes=1440`)
            )
            const studentData = await studentDataResponse.json()

            // 3. Xử lý dữ liệu tổng hợp
            if (studentData.status === 'success' && studentData.student_data.length > 0) {
                // Lấy dữ liệu mới nhất
                const latestData = studentData.student_data[0]

                // Tính toán thống kê từ dữ liệu thực
                const emotionStats = calculateEmotionStats(studentData.student_data)
                const focusStats = calculateFocusStats(studentData.student_data)
                const behaviorStats = calculateBehaviorStats(studentData.student_data)

                // Format dữ liệu chi tiết theo giao diện yêu cầu
                const detailData = {
                    basicInfo: {
                        studentId: studentBasicInfo?.student_id || studentId,
                        studentName: studentBasicInfo?.student_name || studentName,
                        studentCode: studentBasicInfo?.student_code || studentId,
                        className: studentBasicInfo?.class_name || 'Chưa xác định',
                        attendanceStatus: formatAttendance(latestData.attendance_status),
                        checkInTime: latestData.check_in_time,
                        checkOutTime: latestData.check_out_time,
                        recordedAt: latestData.recorded_at
                    },
                    emotion: {
                        current: formatEmotion(latestData.emotion || 'neutral'),
                        confidence: latestData.emotion_confidence || 0.5,
                        history: emotionStats.history,
                        distribution: emotionStats.distribution
                    },
                    behavior: {
                        current: formatBehavior(latestData.behavior_details || latestData.behavior_type || 'Không xác định'),
                        score: latestData.behavior_score || 0,
                        type: formatBehavior(latestData.behavior_type || 'unknown'),
                        details: latestData.behavior_details || 'Không có dữ liệu',
                        history: behaviorStats.history
                    },
                    focus: {
                        currentScore: latestData.focus_score || 0,
                        concentrationLevel: formatConcentrationLevel(latestData.concentration_level),
                        focusDuration: latestData.focus_duration || 0,
                        averageScore: focusStats.average || 0,
                        maxScore: focusStats.max || 0,
                        minScore: focusStats.min || 0,
                        history: focusStats.history || []
                    },
                    analytics: {
                        totalRecords: studentData.total,
                        firstRecord: studentData.student_data[studentData.student_data.length - 1]?.recorded_at,
                        lastRecord: latestData.recorded_at
                    }
                }

                // Cập nhật thống kê từ studentData.stats nếu có
                if (studentData.stats) {
                    detailData.focus.averageScore = Math.round(studentData.stats.avg_focus_score || 0)
                    detailData.focus.maxScore = Math.round(studentData.stats.max_focus_score || 0)
                    detailData.focus.minScore = Math.round(studentData.stats.min_focus_score || 0)
                }

                setStudentDetailData(detailData)
            } else {
                // Nếu không có dữ liệu, tạo dữ liệu mẫu với thông tin từ basicInfo
                const sampleData = createSampleDetailData(studentId, studentName)

                // Cập nhật thông tin cơ bản nếu có
                if (studentBasicInfo) {
                    sampleData.basicInfo = {
                        studentId: studentBasicInfo.student_id,
                        studentName: studentBasicInfo.student_name,
                        studentCode: studentBasicInfo.student_code || studentId,
                        className: studentBasicInfo.class_name || 'Chưa xác định',
                        attendanceStatus: 'Không xác định',
                        checkInTime: null,
                        recordedAt: new Date().toISOString()
                    }
                }

                setStudentDetailData(sampleData)
            }
        } catch (error) {
            console.error('Error fetching student detail:', error)
            // Tạo dữ liệu mẫu khi lỗi
            const sampleData = createSampleDetailData(studentId, studentName)
            setStudentDetailData(sampleData)
        } finally {
            setLoadingDetail(false)
        }
    }

    // Hàm tính toán thống kê cảm xúc
    const calculateEmotionStats = (studentData) => {
        const emotionCounts = {}
        const emotionHistory = []

        // Lấy 4 bản ghi gần nhất có emotion
        const emotionRecords = studentData
            .filter(record => record.emotion)
            .slice(0, 4)

        emotionRecords.forEach(record => {
            const emotion = formatEmotion(record.emotion)
            emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1

            emotionHistory.push({
                emotion: emotion,
                confidence: record.emotion_confidence || 0,
                time: record.recorded_at,
                date: record.date
            })
        })

        return {
            distribution: Object.entries(emotionCounts).map(([emotion, count]) => ({
                emotion,
                count,
                percentage: Math.round((count / emotionRecords.length) * 100)
            })),
            history: emotionHistory
        }
    }

    // Hàm tính toán thống kê độ tập trung
    const calculateFocusStats = (studentData) => {
        // Lọc dữ liệu có focus_score và sắp xếp theo thời gian
        const focusRecords = studentData
            .filter(record => record.focus_score !== null && record.focus_score !== undefined)
            .sort((a, b) => new Date(b.recorded_at) - new Date(a.recorded_at))
            .slice(0, 2)

        if (focusRecords.length === 0) {
            return {
                average: 44,
                max: 0,
                min: 0,
                history: []
            }
        }

        // Tính toán cho history - lấy 2 bản ghi gần nhất
        const focusHistory = focusRecords.map(record => ({
            score: record.focus_score,
            concentrationLevel: formatConcentrationLevel(record.concentration_level),
            time: record.recorded_at,
            date: record.date
        }))

        // Tính toán thống kê từ tất cả records có focus_score
        const allFocusScores = studentData
            .filter(record => record.focus_score !== null && record.focus_score !== undefined)
            .map(record => record.focus_score)

        const average = allFocusScores.length > 0 ?
            Math.round(allFocusScores.reduce((a, b) => a + b, 0) / allFocusScores.length) : 44
        const max = allFocusScores.length > 0 ? Math.max(...allFocusScores) : 48.89
        const min = allFocusScores.length > 0 ? Math.min(...allFocusScores) : 39.2

        return {
            average,
            max,
            min,
            history: focusHistory
        }
    }

    // Hàm tính toán thống kê hành vi
    const calculateBehaviorStats = (studentData) => {
        const behaviorHistory = studentData
            .filter(record => record.behavior_type || record.behavior_details)
            .map(record => ({
                type: formatBehavior(record.behavior_type),
                details: record.behavior_details,
                score: record.behavior_score || 0,
                time: record.recorded_at,
                date: record.date
            }))
            .slice(0, 2)

        return {
            history: behaviorHistory
        }
    }

    // Format attendance status
    const formatAttendance = (status) => {
        const statusMap = {
            'present': 'Có mặt',
            'absent': 'Vắng mặt',
            'late': 'Muộn',
            'excused': 'Có phép'
        }
        return statusMap[status] || status || 'Không xác định'
    }

    // Format concentration level
    const formatConcentrationLevel = (level) => {
        const levelMap = {
            'very_high': 'Rất cao',
            'high': 'Cao',
            'medium': 'Trung bình',
            'low': 'Thấp',
            'very_low': 'Rất thấp'
        }
        return levelMap[level] || 'Trung bình'
    }

    // Tạo dữ liệu mẫu khi API lỗi
    const createSampleDetailData = (studentId, studentName) => {
        return {
            basicInfo: {
                studentId: studentId,
                studentName: studentName,
                studentCode: studentId,
                className: 'Chưa xác định',
                attendanceStatus: 'Không xác định',
                checkInTime: null,
                recordedAt: new Date().toISOString()
            },
            emotion: {
                current: 'Trung lập',
                confidence: 0.5,
                history: [
                    { emotion: 'Trung lập', confidence: 0.5, time: new Date().toISOString(), date: new Date().toISOString().split('T')[0] },
                    { emotion: 'Trung lập', confidence: 0.48, time: new Date(Date.now() - 3600000).toISOString(), date: new Date().toISOString().split('T')[0] },
                    { emotion: 'Trung lập', confidence: 0.52, time: new Date(Date.now() - 7200000).toISOString(), date: new Date().toISOString().split('T')[0] },
                    { emotion: 'Trung lập', confidence: 0.49, time: new Date(Date.now() - 10800000).toISOString(), date: new Date().toISOString().split('T')[0] }
                ],
                distribution: [
                    { emotion: 'Trung lập', count: 4, percentage: 100 }
                ]
            },
            behavior: {
                current: 'Không xác định',
                score: 0,
                type: 'Không xác định',
                details: 'Không có dữ liệu',
                history: []
            },
            focus: {
                currentScore: 0,
                concentrationLevel: 'Trung bình',
                focusDuration: 0,
                averageScore: 44,
                maxScore: 48.89,
                minScore: 39.2,
                history: [
                    { score: 48.89, concentrationLevel: 'Cao', time: new Date().toISOString(), date: new Date().toISOString().split('T')[0] },
                    { score: 39.2, concentrationLevel: 'Cao', time: new Date(Date.now() - 3600000).toISOString(), date: new Date().toISOString().split('T')[0] }
                ]
            },
            analytics: {
                totalRecords: 0,
                firstRecord: null,
                lastRecord: new Date().toISOString()
            }
        }
    }

    // Hàm fallback: fetch dữ liệu attendance để có thông tin thực
    const fetchFallbackStudentData = async () => {
        try {
            const token = localStorage.getItem('access_token')
            const today = new Date().toISOString().split('T')[0]
            const response = await fetch(buildApiUrl(`/api/attendance?token=${token}&date=${today}&limit=20`))
            const data = await response.json()

            if (data.status === 'success' && data.attendance.length > 0) {
                const formattedStudents = data.attendance.map((record, index) => {
                    // Map attendance status
                    let attendanceText = 'Không xác định'
                    switch (record.attendance_status) {
                        case 'present': attendanceText = 'Có mặt'; break;
                        case 'absent': attendanceText = 'Vắng mặt'; break;
                        case 'late': attendanceText = 'Muộn'; break;
                        default: attendanceText = record.attendance_status;
                    }

                    // Map emotion
                    let emotionText = formatEmotion(record.emotion)

                    return {
                        id: record.student_id || `temp_${index}`,
                        name: record.student_name || `Học sinh ${index + 1}`,
                        studentId: record.student_id || `HS${index + 1}`,
                        class: record.class_name || 'Lớp 10A1',
                        attendance: attendanceText,
                        emotion: emotionText,
                        emotionConfidence: record.emotion_confidence || 0,
                        behavior: formatBehavior(record.behavior_type),
                        behaviorType: record.behavior_type || 'unknown',
                        focusScore: Math.round(record.focus_score || record.behavior_score || 75),
                        lastSeen: record.check_in_time ?
                            new Date(record.check_in_time).toLocaleTimeString('vi-VN') :
                            new Date().toLocaleTimeString('vi-VN')
                    }
                })
                setStudentsData(formattedStudents)
                setFilteredStudents(formattedStudents)
            } else {
                // Fallback cuối cùng: lấy từ student-data API
                fetchStudentDataBackup()
            }
        } catch (error) {
            console.error('Error fetching fallback data:', error)
            fetchStudentDataBackup()
        }
    }

    // Backup: fetch từ student-data API
    const fetchStudentDataBackup = async () => {
        try {
            const token = localStorage.getItem('access_token')
            const response = await fetch(buildApiUrl(`/api/student-data?token=${token}&limit=10`))
            const data = await response.json()

            if (data.status === 'success' && data.student_data.length > 0) {
                // Group by student
                const studentsMap = new Map()
                data.student_data.forEach(record => {
                    if (!studentsMap.has(record.student_id)) {
                        studentsMap.set(record.student_id, {
                            id: record.student_id,
                            name: record.student_name,
                            studentId: record.student_id,
                            class: record.class_name || 'Lớp 10A1',
                            attendance: record.attendance_status === 'present' ? 'Có mặt' :
                                record.attendance_status === 'absent' ? 'Vắng mặt' :
                                    record.attendance_status === 'late' ? 'Muộn' : 'Không xác định',
                            emotion: formatEmotion(record.emotion),
                            emotionConfidence: record.emotion_confidence || 0,
                            behavior: formatBehavior(record.behavior_details || record.behavior_type),
                            behaviorType: record.behavior_type || 'unknown',
                            focusScore: Math.round(record.focus_score || record.behavior_score || 75),
                            lastSeen: record.recorded_at ?
                                new Date(record.recorded_at).toLocaleTimeString('vi-VN') :
                                new Date().toLocaleTimeString('vi-VN')
                        })
                    }
                })
                const studentsArray = Array.from(studentsMap.values())
                setStudentsData(studentsArray)
                setFilteredStudents(studentsArray)
            }
        } catch (error) {
            console.error('Error fetching backup student data:', error)
        }
    }

    // ==================== CAMERA FUNCTIONS (SỬ DỤNG ENDPOINT API) ====================

    // Hàm lấy danh sách camera có sẵn
    const fetchAvailableCameras = async () => {
        try {
            const response = await fetch(`${getCameraApiBase()}/list`)
            const data = await response.json()
            
            if (data.status === 'success') {
                setAvailableCameras(data.cameras || [])
                console.log('📷 Available cameras:', data.cameras)
                
                // Nếu có camera, chọn camera đầu tiên
                if (data.cameras && data.cameras.length > 0) {
                    setCameraIndex(data.cameras[0].index)
                    return data.cameras[0].index
                }
            }
            return 0
        } catch (error) {
            console.error('Error fetching cameras:', error)
            return 0
        }
    }

    // Hàm kiểm tra stream MJPEG
    const checkCameraStream = async () => {
        console.log(`🔄 Checking camera stream (attempt ${retryCount + 1})...`)
        setConnectionStatus('connecting')

        try {
            // Kiểm trạng thái server
            const healthResponse = await fetch(buildCameraApiUrl('/api/health'), {
                method: 'GET',
                signal: AbortSignal.timeout(3000)
            })
            
            if (!healthResponse.ok) {
                throw new Error('Server không phản hồi')
            }

            // Kiểm tra video feed
            const streamResponse = await fetch(getVideoFeed(), {
                method: 'HEAD',
                signal: AbortSignal.timeout(3000),
                cache: 'no-cache'
            })

            if (streamResponse.ok) {
                console.log('✅ Camera stream is working')
                setConnectionStatus('connected')
                setCameraError('')
                setRetryCount(0)
                return true
            } else {
                console.error('❌ Invalid stream response:', streamResponse.status)
                setConnectionStatus('error')
                setCameraError(`Server lỗi: ${streamResponse.status}`)
                return false
            }
        } catch (error) {
            console.error('❌ Error checking stream:', error)
            setConnectionStatus('error')
            setCameraError('Không thể kết nối đến camera server')
            return false
        }
    }

    // Hàm bật camera và bắt đầu stream
    const startCamera = async () => {
        if (isCameraOn) return;

        console.log('🔄 Bật camera stream...')
        setCameraError('')
        setIsLoading(true)

        try {
            // 1. Lấy danh sách camera
            const cameraIndex = await fetchAvailableCameras()
            
            // 2. Khởi động camera
            const startResponse = await fetch(`${getCameraApiBase()}/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ camera_index: cameraIndex })
            })

            const startData = await startResponse.json()
            
            if (startData.status !== 'success') {
                throw new Error(startData.message || 'Không thể khởi động camera')
            }

            console.log(`✅ Camera ${cameraIndex} started:`, startData)

            // 3. Kiểm tra stream
            const isConnected = await checkCameraStream()

            if (isConnected) {
                console.log('✅ Camera stream đã kết nối')
                setIsCameraOn(true)
                // Tạo URL với timestamp để tránh cache
                setStreamUrl(`${getVideoFeed()}?t=${Date.now()}`)
                setRetryCount(0)

                // Bắt đầu đếm frame
                startFrameCounter()
            } else {
                // Thử lại sau 2 giây
                if (retryCount < 3) {
                    setRetryCount(prev => prev + 1)
                    setTimeout(() => {
                        startCamera()
                    }, 2000)
                    return
                }
                throw new Error('Không thể kết nối đến camera stream sau nhiều lần thử')
            }

        } catch (error) {
            console.error('❌ Error starting camera:', error)
            setCameraError(`Lỗi: ${error.message || 'Không thể kết nối'}`)
            setConnectionStatus('error')
        } finally {
            setIsLoading(false)
        }
    }

    // Hàm đếm frame
    const startFrameCounter = () => {
        if (frameIntervalRef.current) {
            clearInterval(frameIntervalRef.current)
        }

        setFrameCount(0)
        frameIntervalRef.current = setInterval(() => {
            setFrameCount(prev => prev + 1)
        }, 100)
    }

    // Hàm tắt camera
    const stopCamera = async () => {
        console.log('🛑 Tắt camera stream...')
        
        try {
            // Gửi request stop đến server
            const stopResponse = await fetch(`${getCameraApiBase()}/stop`, {
                method: 'POST'
            })
            
            const stopData = await stopResponse.json()
            console.log('Stop response:', stopData)
        } catch (error) {
            console.error('Error stopping camera:', error)
        } finally {
            setIsCameraOn(false)
            setConnectionStatus('disconnected')
            setStreamUrl('')
            setRetryCount(0)
            setFrameCount(0)

            // Clear tất cả timeout và interval
            if (checkTimeoutRef.current) {
                clearTimeout(checkTimeoutRef.current)
                checkTimeoutRef.current = null
            }

            if (frameIntervalRef.current) {
                clearInterval(frameIntervalRef.current)
                frameIntervalRef.current = null
            }
        }
    }

    // Hàm reload stream
    const reloadStream = () => {
        console.log('🔄 Reload camera stream...')
        if (isCameraOn) {
            // Tạo URL mới với timestamp
            setStreamUrl(`${getVideoFeed()}?t=${Date.now()}&reload=${Math.random()}`)
            // Reset frame counter
            setFrameCount(0)
        } else {
            startCamera()
        }
    }

    // Hàm chuyển đổi camera - bật/tắt
    const toggleCamera = async () => {
        if (isCameraOn) {
            await stopCamera()
        } else {
            await startCamera()
        }
    }

    // Hàm chuyển đổi camera khác
    const switchCamera = async (index) => {
        if (isCameraOn) {
            // Tắt camera hiện tại
            await stopCamera()
        }
        
        setCameraIndex(index)
        console.log(`🔄 Switching to camera ${index}`)
        
        // Khởi động camera mới
        setTimeout(() => {
            startCamera()
        }, 500)
    }

    // Hàm test connection
    const testConnection = async () => {
        console.log('🧪 Manual connection test...')
        setIsLoading(true)

        try {
            const response = await fetch(getVideoFeed(), {
                method: 'GET',
                mode: 'cors'
            })

            const contentType = response.headers.get('content-type')
            const status = response.status

            alert(`✅ Server đang chạy!\n\nStatus: ${status}\nContent-Type: ${contentType}\n\nMở tab mới để xem stream trực tiếp?`)

            // Tự động mở stream trong tab mới
            window.open(getVideoFeed(), '_blank')

            return true
        } catch (error) {
            console.error('Test connection error:', error)
            alert(`❌ Lỗi kết nối: ${error.message}\n\nKiểm tra:\n1. Server có đang chạy không?\n2. Port 5000 có bị chặn không?`)
            return false
        } finally {
            setIsLoading(false)
        }
    }

    // Thêm hàm kiểm tra server trực tiếp
    const testServerDirectly = () => {
        window.open(buildCameraApiUrl('/'), '_blank')
    }

    // Hàm chụp ảnh từ camera
    const capturePhoto = async () => {
        try {
            const response = await fetch(`${getCameraApiBase()}/capture`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({})
            })

            const data = await response.json()
            
            if (data.status === 'success') {
                alert(`✅ Đã chụp ảnh: ${data.filename}`)
                // Có thể hiển thị ảnh hoặc tải xuống
                const downloadUrl = buildCameraApiUrl(`/${data.filename}`)
                window.open(downloadUrl, '_blank')
            } else {
                alert(`❌ Lỗi: ${data.message}`)
            }
        } catch (error) {
            console.error('Error capturing photo:', error)
            alert('❌ Lỗi khi chụp ảnh')
        }
    }

    // ==================== USER DASHBOARD FUNCTIONS ====================

    useEffect(() => {
        // Tự động thử kết nối camera khi component mount
        checkCameraStream()

        // Fetch user dashboard data
        if (user) {
            fetchUserDashboard()
        }

        return () => {
            // Cleanup camera
            if (checkTimeoutRef.current) {
                clearTimeout(checkTimeoutRef.current)
            }
            if (frameIntervalRef.current) {
                clearInterval(frameIntervalRef.current)
            }
            
            // Dừng camera khi unmount
            if (isCameraOn) {
                stopCamera()
            }
        }
    }, [])

    const handleLogout = () => {
        logout()
    }

    const handleGoToAdmin = () => {
        router.push('/dashboard')
    }

    const handleViewStudentDetail = (student) => {
        // Fetch chi tiết học sinh từ API
        setSelectedStudent(student)
        setShowStudentDetail(true)
        fetchStudentDetailData(student.id, student.name)
    }

    const handleRefresh = () => {
        setLoadingDashboard(true)
        fetchUserDashboard()
    }

    // Format date cho hiển thị
    const formatDate = (dateString) => {
        if (!dateString) return 'Không có dữ liệu'
        const date = new Date(dateString)
        return date.toLocaleDateString('vi-VN', {
            day: '2-digit',
            month: '2-digit',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        })
    }

    // Format time cho hiển thị
    const formatTime = (dateString) => {
        if (!dateString) return 'Không có dữ liệu'
        const date = new Date(dateString)
        return date.toLocaleTimeString('vi-VN', {
            hour: '2-digit',
            minute: '2-digit'
        })
    }

    // Format chỉ lấy ngày
    const formatDateOnly = (dateString) => {
        if (!dateString) return 'Không có dữ liệu'
        const date = new Date(dateString)
        return date.toLocaleDateString('vi-VN')
    }

    // ==================== HELPER FUNCTIONS ====================

    const getAttendanceColor = (status) => {
        switch (status) {
            case 'Có mặt': return 'text-green-400'
            case 'Vắng mặt': return 'text-red-400'
            case 'Muộn': return 'text-yellow-400'
            default: return 'text-gray-400'
        }
    }

    const getEmotionColor = (emotion) => {
        switch (emotion) {
            case 'Hạnh phúc': return 'text-green-400'
            case 'Buồn': return 'text-blue-400'
            case 'Trung lập': return 'text-gray-400'
            case 'Ngạc nhiên': return 'text-yellow-400'
            case 'Tức giận': return 'text-red-400'
            case 'Sợ hãi': return 'text-orange-400'
            case 'Khó chịu': return 'text-pink-400'
            default: return 'text-gray-400'
        }
    }

    const getBehaviorColor = (behavior) => {
        switch (behavior) {
            case 'Đang viết': return 'text-blue-400'
            case 'Nhìn thẳng': return 'text-green-400'
            case 'Giơ tay': return 'text-yellow-400'
            case 'Bình thường': return 'text-gray-400'
            case 'Nhìn quanh': return 'text-orange-400'
            case 'Mất tập trung': return 'text-red-400'
            default: return 'text-gray-400'
        }
    }

    const getFocusColor = (score) => {
        if (score >= 85) return 'text-green-400'
        if (score >= 70) return 'text-yellow-400'
        return 'text-red-400'
    }

    const getFocusBgColor = (score) => {
        if (score >= 85) return 'bg-green-500'
        if (score >= 70) return 'bg-yellow-500'
        return 'bg-red-500'
    }

    const getConcentrationColor = (level) => {
        switch (level) {
            case 'Rất cao': return 'text-green-400'
            case 'Cao': return 'text-green-300'
            case 'Trung bình': return 'text-gray-400'
            case 'Thấp': return 'text-orange-400'
            case 'Rất thấp': return 'text-red-400'
            default: return 'text-gray-400'
        }
    }

    const getConnectionStatusColor = () => {
        switch (connectionStatus) {
            case 'connected': return 'bg-green-500'
            case 'connecting': return 'bg-yellow-500'
            case 'error': return 'bg-red-500'
            default: return 'bg-gray-500'
        }
    }

    const getConnectionStatusText = () => {
        switch (connectionStatus) {
            case 'connected': return 'Đã kết nối'
            case 'connecting': return 'Đang kết nối...'
            case 'error': return 'Lỗi kết nối'
            default: return 'Chưa kết nối'
        }
    }

    // Lấy thời gian hiện tại theo định dạng
    const getCurrentDateTime = () => {
        const now = new Date()
        return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`
    }

    // ==================== RENDER ====================

    return (
        <ProtectedRoute>
            <div className="min-h-screen bg-gray-100 p-4 md:p-2">
                <div className="max-w-7xl mx-auto">
                    {/* Header với nút logout */}
                    <div className="mb-2 md:mb-2 flex justify-between items-center">
                        <div className="pb-0">
                            <div className="flex justify-center">
                                <img
                                    src="/logo_company2.png"
                                    alt="LYDINC Logo"
                                    className="h-22 w-auto"
                                />
                            </div>
                        </div>
                        <div className="flex items-center space-x-4">
                            <button
                                onClick={handleRefresh}
                                disabled={loadingDashboard}
                                className="px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 flex items-center"
                            >
                                <span className="mr-2">🔄</span>
                                Làm mới
                            </button>
                            <button
                                onClick={handleLogout}
                                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 font-medium transition"
                            >
                                🚪 Đăng xuất
                            </button>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 md:gap-6">
                        {/* ==================== MAIN CONTENT - Camera Frame ==================== */}
                        <div className="lg:col-span-3">
                            <div className="bg-gray-900 rounded-xl md:rounded-2xl shadow-xl overflow-hidden border border-gray-800">
                                {/* Live Header */}
                                <div className="bg-gray-800 p-3 md:p-4 border-b border-gray-700">
                                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                                        <h2 className="text-lg md:text-xl font-bold text-white flex items-center">
                                            <span className="mr-2">📹</span>
                                            Camera Lớp Học
                                            <span className={`ml-2 w-2 h-2 rounded-full ${getConnectionStatusColor()} animate-pulse`}></span>
                                        </h2>
                                        <div className="flex flex-col sm:items-end text-sm text-gray-300">
                                            <div>{getCurrentDateTime()}</div>
                                            <div className="text-xs text-gray-400">
                                               Lớp STEM 1
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Video Container - Kích thước 640x480 */}
                                <div className="relative bg-black">
                                    <div className="relative mx-auto" style={{ width: '640px', height: '480px' }}>
                                        {/* Camera Stream */}
                                        {isCameraOn && connectionStatus === 'connected' && streamUrl ? (
                                            <div className="w-full h-full">
                                                <iframe
                                                    src={streamUrl}
                                                    className="absolute inset-0 w-full h-full border-0"
                                                    title="Classroom Camera Stream"
                                                    sandbox="allow-same-origin"
                                                    allow="camera"
                                                    style={{
                                                        width: '640px',
                                                        height: '480px',
                                                        backgroundColor: 'black'
                                                    }}
                                                />
                                                <div className="absolute inset-0 pointer-events-none border-0"></div>
                                            </div>
                                        ) : (
                                            <div className="w-full h-full flex items-center justify-center bg-black">
                                                <div className="text-center text-white p-8">
                                                    <div className="text-6xl mb-4">📹</div>
                                                    <p className="text-xl font-semibold mb-3">
                                                        {cameraError ? 'Lỗi Camera' : 'Classroom Camera'}
                                                    </p>
                                                    <p className="text-gray-400 mb-5 max-w-md mx-auto">
                                                        {cameraError || 'Nhấn "Bật Camera" để xem stream lớp học'}
                                                    </p>

                                                    {cameraError && (
                                                        <div className="mt-4 p-4 bg-red-900/20 border border-red-800 rounded-lg max-w-md mx-auto">
                                                            <p className="text-red-300 font-medium mb-2">🔧 Khắc phục sự cố:</p>
                                                            <div className="flex gap-3 justify-center">
                                                                <button
                                                                    onClick={testServerDirectly}
                                                                    className="bg-gray-700 hover:bg-gray-600 text-white px-3 py-1.5 rounded text-sm"
                                                                >
                                                                    Mở trang server
                                                                </button>
                                                                <button
                                                                    onClick={reloadStream}
                                                                    className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded text-sm"
                                                                >
                                                                    Thử lại
                                                                </button>
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        )}

                                        {/* Loading Overlay */}
                                        {isLoading && (
                                            <div className="absolute inset-0 bg-black bg-opacity-80 flex items-center justify-center z-10">
                                                <div className="text-center text-white">
                                                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
                                                    <p className="text-lg font-medium">Đang kết nối đến camera...</p>
                                                    <p className="text-sm text-gray-400 mt-2">
                                                        {retryCount > 0 ? `Thử lại lần ${retryCount + 1}...` : 'Đang kiểm tra kết nối...'}
                                                    </p>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>

                                {/* Control Bar */}
                                <div className="bg-gray-800 p-4 border-t border-gray-700">
                                    <div className="flex flex-wrap items-center justify-between gap-3">
                                        <div className="flex items-center space-x-2">
                                            <div className={`w-3 h-3 rounded-full ${getConnectionStatusColor()}`}></div>
                                            <span className="text-gray-300 text-sm">
                                                {getConnectionStatusText()}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Class Info Below Camera */}
                            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                                <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
                                    <div className="flex items-center">
                                        <div className="w-10 h-10 bg-blue-900/30 rounded-lg flex items-center justify-center mr-4">
                                            <span className="text-blue-400 text-xl">🏫</span>
                                        </div>
                                        <div>
                                            <p className="text-sm text-gray-400">Tên Lớp Học</p>
                                            <p className="text-lg font-bold text-white">
                                                STEM 1
                                            </p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
                                    <div className="flex items-center">
                                        <div className="w-10 h-10 bg-green-900/30 rounded-lg flex items-center justify-center mr-4">
                                            <span className="text-green-400 text-xl">👤</span>
                                        </div>
                                        <div>
                                            <p className="text-sm text-gray-400">Giáo viên</p>
                                            <p className="text-lg font-bold text-white">
                                                Cô Kha
                                            </p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
                                    <div className="flex items-center">
                                        <div className="w-10 h-10 bg-purple-900/30 rounded-lg flex items-center justify-center mr-4">
                                            <span className="text-purple-400 text-xl">🎭</span>
                                        </div>
                                        <div>
                                            <p className="text-sm text-gray-400">Tổng học sinh</p>
                                            <p className="text-lg font-bold text-white">
                                                {studentsData.length} học sinh
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* ==================== SIDEBAR - Student List ==================== */}
                        <div className="lg:col-span-1 space-y-6">
                            {/* Camera Control Panel */}
                            <div className="bg-gray-900 rounded-xl overflow-hidden border border-gray-800">
                                <div className="bg-gray-800 p-4">
                                    <h3 className="text-lg font-bold text-white">🎥 Điều khiển Camera</h3>
                                </div>
                                <div className="p-4 space-y-3">
                                    {/* Camera Selection */}
                                    {availableCameras.length > 1 && (
                                        <div className="mb-4">
                                            <label className="block text-sm text-gray-400 mb-2">Chọn camera:</label>
                                            <select
                                                value={cameraIndex}
                                                onChange={(e) => switchCamera(parseInt(e.target.value))}
                                                className="w-full bg-gray-700 border border-gray-600 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                            >
                                                {availableCameras.map(camera => (
                                                    <option key={camera.index} value={camera.index}>
                                                        {camera.name} ({camera.resolution})
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    )}

                                    <button
                                        onClick={toggleCamera}
                                        disabled={isLoading}
                                        className={`w-full py-3 rounded-lg font-medium ${isCameraOn
                                            ? 'bg-red-600 hover:bg-red-700'
                                            : 'bg-green-600 hover:bg-green-700'
                                            } text-white disabled:opacity-50 flex items-center justify-center`}
                                    >
                                        <span className="mr-2">{isCameraOn ? '🔴' : '🟢'}</span>
                                        {isCameraOn ? 'Tắt Camera' : 'Bật Camera'}
                                    </button>
                                </div>
                            </div>

                            {/* Student List */}
                            <div className="bg-gray-900 rounded-xl overflow-hidden border border-gray-800">
                                <div className="bg-gray-800 p-4">
                                    <div className="flex justify-between items-center">
                                        <h3 className="text-lg font-bold text-white flex items-center">
                                            <span className="mr-2">👨‍🎓</span>
                                            Danh sách học sinh
                                        </h3>
                                        <span className="text-sm font-normal text-gray-400">
                                            {loadingStudents ? (
                                                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                                            ) : (
                                                `(${filteredStudents.length}/${studentsData.length})`
                                            )}
                                        </span>
                                    </div>

                                    {/* THANH TÌM KIẾM */}
                                    <div className="mt-3">
                                        <div className="relative">
                                            <input
                                                type="text"
                                                value={searchQuery}
                                                onChange={(e) => handleSearch(e.target.value)}
                                                placeholder="Tìm học sinh"
                                                className="w-full bg-gray-700 border border-gray-600 text-white rounded-lg pl-10 pr-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                            />
                                            <div className="absolute left-3 top-2.5 text-gray-400">
                                                🔍
                                            </div>
                                            {searchQuery && (
                                                <button
                                                    onClick={() => handleSearch('')}
                                                    className="absolute right-3 top-2.5 text-gray-400 hover:text-white"
                                                >
                                                    ✕
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                </div>
                                <div className="p-4">
                                    {loadingStudents ? (
                                        <div className="flex justify-center items-center h-32">
                                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
                                        </div>
                                    ) : filteredStudents.length === 0 ? (
                                        <div className="text-center py-8">
                                            <div className="text-4xl mb-3">🔍</div>
                                            <p className="text-gray-400 mb-2">
                                                {searchQuery ?
                                                    `Không tìm thấy học sinh với "${searchQuery}"` :
                                                    'Không có dữ liệu học sinh'
                                                }
                                            </p>
                                            {searchQuery ? (
                                                <button
                                                    onClick={() => handleSearch('')}
                                                    className="mt-2 px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-600 text-sm"
                                                >
                                                    Xóa tìm kiếm
                                                </button>
                                            ) : (
                                                <button
                                                    onClick={handleRefresh}
                                                    className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm"
                                                >
                                                    Tải lại
                                                </button>
                                            )}
                                        </div>
                                    ) : (
                                        <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
                                            {filteredStudents.map((student) => (
                                                <div key={student.id} className="bg-gray-800/50 rounded-lg p-3 border border-gray-700 hover:border-gray-600 transition-colors">
                                                    <div className="flex justify-between items-start mb-2">
                                                        <div>
                                                            <h4 className="font-medium text-white">{student.name}</h4>
                                                            <p className="text-xs text-gray-400">Mã: {student.studentId}</p>
                                                            <p className="text-xs text-gray-400">Lớp: {student.class}</p>
                                                        </div>
                                                        <span className={`text-xs font-medium px-2 py-1 rounded ${getAttendanceColor(student.attendance)}`}>
                                                            {student.attendance}
                                                        </span>
                                                    </div>

                                                    <div className="grid grid-cols-2 gap-2 text-xs mb-3">
                                                        <div className="flex items-center">
                                                            <span className="text-gray-400 mr-1">Cảm xúc:</span>
                                                            <span className={getEmotionColor(student.emotion)}>{student.emotion}</span>
                                                        </div>
                                                        <div className="flex items-center">
                                                            <span className="text-gray-400 mr-1">Tập trung:</span>
                                                            <span className={getFocusColor(student.focusScore)}>{student.focusScore}</span>
                                                        </div>
                                                    </div>

                                                    <button
                                                        onClick={() => handleViewStudentDetail(student)}
                                                        className="w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white text-sm font-medium py-2 rounded-lg transition-all duration-300"
                                                    >
                                                        Xem chi tiết
                                                    </button>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Student Detail Modal */}
                {showStudentDetail && selectedStudent && (
                    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
                        <div className="bg-gray-900 rounded-2xl border border-gray-800 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
                            <div className="bg-gradient-to-r from-blue-800 to-purple-800 p-6">
                                <div className="flex justify-between items-start">
                                    <div>
                                        <h2 className="text-2xl font-bold text-white mb-2">{selectedStudent.name}</h2>
                                        <p className="text-gray-200">Mã học sinh: {studentDetailData?.basicInfo?.studentCode || selectedStudent.studentId}</p>
                                    </div>
                                    <button
                                        onClick={() => setShowStudentDetail(false)}
                                        className="text-white hover:text-gray-300 text-2xl"
                                    >
                                        ✕
                                    </button>
                                </div>
                            </div>

                            <div className="p-6">
                                {loadingDetail ? (
                                    <div className="flex justify-center items-center h-64">
                                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white"></div>
                                        <p className="ml-3 text-white">Đang tải dữ liệu...</p>
                                    </div>
                                ) : studentDetailData ? (
                                    <>
                                        {/* Basic Info */}
                                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                                            <div className="bg-gray-800/50 rounded-xl p-4">
                                                <p className="text-sm text-gray-400 mb-1">Lớp</p>
                                                <p className="text-lg font-bold text-white">{studentDetailData.basicInfo.className}</p>
                                            </div>
                                            <div className="bg-gray-800/50 rounded-xl p-4">
                                                <p className="text-sm text-gray-400 mb-1">Điểm danh</p>
                                                <p className={`text-lg font-bold ${getAttendanceColor(studentDetailData.basicInfo.attendanceStatus)}`}>
                                                    {studentDetailData.basicInfo.attendanceStatus}
                                                </p>
                                            </div>
                                            <div className="bg-gray-800/50 rounded-xl p-4">
                                                <p className="text-sm text-gray-400 mb-1">Thời gian vào lớp</p>
                                                <p className="text-lg font-bold text-white">
                                                    {studentDetailData.basicInfo.checkInTime ? formatTime(studentDetailData.basicInfo.checkInTime) : 'Chưa check-in'}
                                                </p>
                                            </div>
                                        </div>

                                        {/* Emotion Section */}
                                        <div className="mb-6">
                                            <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                                                <span className="mr-2">😊</span> Cảm xúc
                                            </h3>
                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                                <div className="bg-gray-800/30 rounded-lg p-4">
                                                    <p className="text-gray-400 mb-2">Cảm xúc hiện tại</p>
                                                    <div className="flex items-center justify-between">
                                                        <span className={`text-xl font-bold ${getEmotionColor(studentDetailData.emotion.current)}`}>
                                                            {studentDetailData.emotion.current}
                                                        </span>
                                                        <span className="text-gray-300">
                                                            {Math.round((studentDetailData.emotion.confidence || 0) * 100)}%
                                                        </span>
                                                    </div>
                                                </div>
                                                <div className="bg-gray-800/30 rounded-lg p-4">
                                                    <p className="text-gray-400 mb-2">Lịch sử cảm xúc</p>
                                                    <div className="space-y-2 max-h-40 overflow-y-auto">
                                                        {studentDetailData.emotion.history?.map((item, index) => (
                                                            <div key={index} className="text-sm border-b border-gray-700 pb-2">
                                                                <div className="flex justify-between">
                                                                    <span className={getEmotionColor(item.emotion)}>{item.emotion}</span>
                                                                    <span className="text-gray-400">{Math.round((item.confidence || 0) * 100)}%</span>
                                                                </div>
                                                                <div className="text-xs text-gray-500 mt-1">
                                                                    {formatDateOnly(item.date)}
                                                                </div>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Behavior Section */}
                                        <div className="mb-6">
                                            <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                                                <span className="mr-2">📝</span> Hành vi
                                            </h3>
                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                                <div className="bg-gray-800/30 rounded-lg p-4">
                                                    <p className="text-gray-400 mb-2">Hành vi hiện tại</p>
                                                    <div className="flex items-center justify-between">
                                                        <span className={`text-xl font-bold ${getBehaviorColor(studentDetailData.behavior.current)}`}>
                                                            {studentDetailData.behavior.current}
                                                        </span>
                                                        <span className="text-gray-300">
                                                            {studentDetailData.behavior.score}/100
                                                        </span>
                                                    </div>
                                                    {studentDetailData.behavior.details && (
                                                        <p className="text-gray-300 text-sm mt-2">{studentDetailData.behavior.details}</p>
                                                    )}
                                                </div>
                                                <div className="bg-gray-800/30 rounded-lg p-4">
                                                    <p className="text-gray-400 mb-2">Lịch sử hành vi</p>
                                                    <div className="space-y-2 max-h-40 overflow-y-auto">
                                                        {studentDetailData.behavior.history?.map((item, index) => (
                                                            <div key={index} className="text-sm border-b border-gray-700 pb-2">
                                                                <div className="flex justify-between">
                                                                    <span className={getBehaviorColor(item.type)}>{item.type}</span>
                                                                    <span className="text-gray-400">{item.score}/100</span>
                                                                </div>
                                                                {item.details && (
                                                                    <div className="text-xs text-gray-500 truncate">{item.details}</div>
                                                                )}
                                                                <div className="text-xs text-gray-500 mt-1">
                                                                    {formatDateOnly(item.date)}
                                                                </div>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Focus Section */}
                                        <div className="mb-6">
                                            <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                                                <span className="mr-2">🎯</span> Đánh Giá
                                            </h3>
                                            <div className="mb-4">
                                                <div className="bg-gray-800/30 rounded-lg p-4">
                                                    <p className="text-gray-400 mb-2">Trạng thái</p>
                                                    <div className="text-center">
                                                        <span className={`text-2xl font-bold ${studentDetailData.focus.currentScore >= 50 ? 'text-green-400' : 'text-red-400'}`}>
                                                            {studentDetailData.focus.currentScore >= 50 ? 'Tập trung' : 'Không tập trung'}
                                                        </span>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </>
                                ) : (
                                    <div className="text-center py-8">
                                        <p className="text-gray-400">Không thể tải dữ liệu chi tiết</p>
                                    </div>
                                )}

                                <div className="flex justify-end space-x-3 mt-6">
                                    <button
                                        onClick={() => setShowStudentDetail(false)}
                                        className="px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-medium transition"
                                    >
                                        Đóng
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </ProtectedRoute>
    )
}