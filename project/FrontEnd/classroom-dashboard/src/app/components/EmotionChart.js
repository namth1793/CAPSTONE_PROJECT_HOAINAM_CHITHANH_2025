// frontend/components/EmotionChart.js
import { useEffect, useState } from 'react'
import { buildApiUrl } from '../config/api'
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'

const EmotionChart = () => {
    const [emotionData, setEmotionData] = useState([])
    const [loading, setLoading] = useState(false)
    const [summary, setSummary] = useState({
        totalDetections: 0,
        dominantEmotion: 'neutral',
        avgConfidence: 0,
        dataFreshness: 'demo',
        lastUpdate: new Date().toLocaleTimeString()
    })

    // Màu sắc cho từng cảm xúc
    const EMOTION_COLORS = {
        'happy': '#10B981',      // Green
        'neutral': '#6B7280',    // Gray
        'sad': '#EF4444',        // Red
        'angry': '#DC2626',      // Dark red
        'surprised': '#F59E0B',  // Orange
        'fearful': '#8B5CF6',    // Purple
        'disgusted': '#84CC16',  // Lime green
        'default': '#9CA3AF'     // Default gray
    }

    // Icon cho từng cảm xúc
    const EMOTION_ICONS = {
        'happy': '😊',
        'neutral': '😐',
        'sad': '😢',
        'angry': '😠',
        'surprised': '😲',
        'fearful': '😨',
        'disgusted': '🤢',
        'default': '😐'
    }

    // Tên hiển thị cho từng cảm xúc
    const EMOTION_LABELS = {
        'happy': 'Happy',
        'neutral': 'Neutral',
        'sad': 'Sad',
        'angry': 'Angry',
        'surprised': 'Surprised',
        'fearful': 'Fearful',
        'disgusted': 'Disgusted'
    }

    useEffect(() => {
        fetchEmotionData()

        // Refresh data mỗi 30 giây
        const interval = setInterval(() => {
            fetchEmotionData()
        }, 30000)

        return () => clearInterval(interval)
    }, [])

    const fetchEmotionData = async () => {
        setLoading(true)
        try {
            // 🔴 SỬA: Gọi API engagement real-time để lấy emotion data
            const response = await fetch(buildApiUrl('/api/engagement/realtime?recent_minutes=10'))

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`)
            }

            const data = await response.json()

            if (data.status === 'success') {
                // Tính toán phân bổ cảm xúc từ real-time data
                const emotionDistribution = calculateEmotionDistributionFromRealTime(data.students || [])
                setEmotionData(emotionDistribution)

                // Cập nhật summary
                updateSummary(data.students || [], data.summary || {})
            } else {
                // Fallback nếu không có dữ liệu
                setEmotionData(getDefaultData())
                setSummary(prev => ({
                    ...prev,
                    dataFreshness: 'fallback'
                }))
            }
        } catch (error) {
            console.error('Error fetching emotion data:', error)
            // Fallback nếu API lỗi
            setEmotionData(getDefaultData())
            setSummary(prev => ({
                ...prev,
                dataFreshness: 'offline'
            }))
        } finally {
            setLoading(false)
        }
    }

    const calculateEmotionDistributionFromRealTime = (students) => {
        if (students.length === 0) {
            return getDefaultData()
        }

        // Đếm số lượng từng loại cảm xúc
        const emotionCounts = {}
        let totalConfidence = 0
        let validEmotions = 0

        students.forEach(student => {
            const emotion = (student.latest_emotion || 'neutral').toLowerCase()
            const confidence = student.emotion_confidence || 0.5

            // Chuẩn hóa tên cảm xúc
            let normalizedEmotion = emotion
            if (emotion.includes('surprise')) normalizedEmotion = 'surprised'
            if (emotion.includes('fear')) normalizedEmotion = 'fearful'
            if (emotion.includes('disgust')) normalizedEmotion = 'disgusted'

            // Chỉ đếm các cảm xúc hợp lệ
            if (EMOTION_LABELS[normalizedEmotion]) {
                emotionCounts[normalizedEmotion] = (emotionCounts[normalizedEmotion] || 0) + 1
                totalConfidence += confidence
                validEmotions++
            }
        })

        // Nếu không có cảm xúc hợp lệ, trả về default
        if (Object.keys(emotionCounts).length === 0) {
            return getDefaultData()
        }

        // Chuyển đổi sang mảng và tính phần trăm
        const totalDetections = Object.values(emotionCounts).reduce((a, b) => a + b, 0)
        const avgConfidence = validEmotions > 0 ? totalConfidence / validEmotions : 0

        const emotionArray = Object.entries(emotionCounts).map(([emotion, count]) => ({
            name: EMOTION_LABELS[emotion],
            value: Math.round((count / totalDetections) * 100),
            originalName: emotion,
            count: count,
            percentage: Math.round((count / totalDetections) * 100),
            icon: EMOTION_ICONS[emotion] || EMOTION_ICONS.default
        }))

        // Sắp xếp theo value giảm dần
        emotionArray.sort((a, b) => b.value - a.value)

        // Cập nhật summary
        setSummary(prev => ({
            ...prev,
            totalDetections,
            avgConfidence: Math.round(avgConfidence * 100)
        }))

        return emotionArray
    }

    const updateSummary = (students, apiSummary) => {
        if (students.length === 0) {
            setSummary(prev => ({
                ...prev,
                totalDetections: 0,
                dominantEmotion: 'neutral',
                avgConfidence: 0,
                dataFreshness: apiSummary.data_freshness || 'demo',
                lastUpdate: new Date().toLocaleTimeString()
            }))
            return
        }

        // Tìm cảm xúc chiếm ưu thế
        const emotionCounts = {}
        let totalConfidence = 0
        let validDetections = 0

        students.forEach(student => {
            const emotion = (student.latest_emotion || 'neutral').toLowerCase()
            const confidence = student.emotion_confidence || 0.5

            if (EMOTION_LABELS[emotion]) {
                emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1
                totalConfidence += confidence
                validDetections++
            }
        })

        let dominantEmotion = 'neutral'
        let maxCount = 0
        Object.entries(emotionCounts).forEach(([emotion, count]) => {
            if (count > maxCount) {
                maxCount = count
                dominantEmotion = emotion
            }
        })

        const avgConfidence = validDetections > 0 ? Math.round((totalConfidence / validDetections) * 100) : 0

        setSummary({
            totalDetections: students.length,
            dominantEmotion: EMOTION_LABELS[dominantEmotion] || 'Neutral',
            avgConfidence,
            dataFreshness: apiSummary.data_freshness || 'recent',
            lastUpdate: new Date().toLocaleTimeString()
        })
    }

    const getDefaultData = () => {
        return [
            {
                name: 'Happy',
                value: 35,
                originalName: 'happy',
                count: 7,
                icon: '😊'
            },
            {
                name: 'Neutral',
                value: 40,
                originalName: 'neutral',
                count: 8,
                icon: '😐'
            },
            {
                name: 'Sad',
                value: 10,
                originalName: 'sad',
                count: 2,
                icon: '😢'
            },
            {
                name: 'Surprised',
                value: 15,
                originalName: 'surprised',
                count: 3,
                icon: '😲'
            }
        ]
    }

    // Hàm lấy màu cho cảm xúc
    const getEmotionColor = (emotionName) => {
        const emotionKey = emotionName.toLowerCase()
        return EMOTION_COLORS[emotionKey] || EMOTION_COLORS.default
    }

    const getDataStatusText = () => {
        switch (summary.dataFreshness) {
            case 'live':
                return 'Live AI Data'
            case 'recent':
                return 'Recent AI Data'
            case 'demo':
                return 'Demo Data'
            case 'fallback':
                return 'Fallback Data'
            case 'offline':
                return 'Offline Mode'
            default:
                return 'AI Emotion Data'
        }
    }

    const getDataStatusColor = () => {
        switch (summary.dataFreshness) {
            case 'live':
                return 'bg-green-100 text-green-800 border-green-300'
            case 'recent':
                return 'bg-blue-100 text-blue-800 border-blue-300'
            case 'demo':
            case 'fallback':
                return 'bg-yellow-100 text-yellow-800 border-yellow-300'
            case 'offline':
                return 'bg-red-100 text-red-800 border-red-300'
            default:
                return 'bg-gray-100 text-gray-800 border-gray-300'
        }
    }

    // Format dữ liệu cho tooltip
    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload
            return (
                <div className="bg-gray-900 text-white p-3 rounded-lg shadow-lg border border-gray-700">
                    <div className="flex items-center gap-2 mb-2">
                        <span className="text-xl">{data.icon}</span>
                        <p className="font-semibold text-lg">{data.name}</p>
                    </div>
                    <div className="space-y-1">
                        <p className="text-blue-300 font-medium">
                            {data.value}% of detected emotions
                        </p>
                        <p className="text-gray-300">
                            {data.count} student{data.count !== 1 ? 's' : ''}
                        </p>
                        <div className="flex items-center gap-2 mt-2 pt-2 border-t border-gray-700">
                            <div
                                className="w-3 h-3 rounded-sm"
                                style={{ backgroundColor: getEmotionColor(data.originalName) }}
                            ></div>
                            <p className="text-xs text-gray-400">
                                AI detected with {summary.avgConfidence}% avg confidence
                            </p>
                        </div>
                    </div>
                </div>
            )
        }
        return null
    }

    // Custom legend
    const renderColorfulLegendText = (value, entry) => {
        const { color } = entry
        const emotionKey = Object.keys(EMOTION_LABELS).find(
            key => EMOTION_LABELS[key] === value
        ) || 'default'

        return (
            <div className="flex items-center gap-2">
                <span className="text-lg">{EMOTION_ICONS[emotionKey] || EMOTION_ICONS.default}</span>
                <span style={{ color: '#374151' }} className="text-sm">
                    {value}
                </span>
            </div>
        )
    }

    // Hiển thị loading nếu đang tải dữ liệu
    if (loading) {
        return (
            <div className="bg-white rounded-xl p-6 h-full">
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-xl font-semibold text-black">AI Emotion Distribution</h2>
                    <span className={`px-2 py-1 text-xs font-medium rounded-full border ${getDataStatusColor()}`}>
                        {getDataStatusText()}
                    </span>
                </div>
                <div className="h-64 flex flex-col items-center justify-center">
                    <div className="text-gray-500 mb-2">Detecting emotions from camera feed...</div>
                    <div className="animate-pulse flex space-x-2">
                        <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                        <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                        <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="bg-white rounded-xl pt-10 h-full">
            {/* Chart */}
            <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                        <Pie
                            data={emotionData}
                            cx="50%"
                            cy="50%"
                            innerRadius={45}
                            outerRadius={75}
                            paddingAngle={2}
                            dataKey="value"
                            nameKey="name"
                            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                            labelLine={false}
                        >
                            {emotionData.map((entry, index) => (
                                <Cell
                                    key={`cell-${index}`}
                                    fill={getEmotionColor(entry.originalName)}
                                    stroke="#000000ff"
                                    strokeWidth={1}
                                />
                            ))}
                        </Pie>
                        <Tooltip content={<CustomTooltip />} />
                        <Legend
                            wrapperStyle={{
                                fontSize: '12px',
                                paddingTop: '10px'
                            }}
                            formatter={renderColorfulLegendText}
                            layout="horizontal"
                            verticalAlign="bottom"
                            align="center"
                        />
                    </PieChart>
                </ResponsiveContainer>
            </div>
        </div>
    )
}

export default EmotionChart