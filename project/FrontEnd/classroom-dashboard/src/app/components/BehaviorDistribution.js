// frontend/components/BehaviorDistribution.js
import { useEffect, useState } from 'react'
import { buildApiUrl } from '../config/api'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

const BehaviorDistribution = () => {
    const [behaviorData, setBehaviorData] = useState([])
    const [loading, setLoading] = useState(false)
    const [summary, setSummary] = useState({
        totalBehaviors: 0,
        topBehavior: 'normal',
        dataFreshness: 'demo',
        lastUpdate: new Date().toLocaleTimeString()
    })

    useEffect(() => {
        fetchBehaviorData()

        // Refresh data mỗi 30 giây
        const interval = setInterval(() => {
            fetchBehaviorData()
        }, 30000)

        return () => clearInterval(interval)
    }, [])

    const fetchBehaviorData = async () => {
        setLoading(true)
        try {
            // 🔴 SỬA: Gọi API engagement real-time để lấy behavior data
            const response = await fetch(buildApiUrl('/api/engagement/realtime?recent_minutes=10'))

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`)
            }

            const data = await response.json()

            if (data.status === 'success') {
                // Tính toán phân bổ hành vi từ real-time data
                const behaviorDistribution = calculateBehaviorDistributionFromRealTime(data.students || [])
                setBehaviorData(behaviorDistribution)

                // Cập nhật summary
                updateSummary(data.students || [], data.summary?.data_freshness || 'demo')
            } else {
                // Fallback nếu không có dữ liệu
                setBehaviorData(getDefaultData())
                setSummary(prev => ({
                    ...prev,
                    dataFreshness: 'fallback'
                }))
            }
        } catch (error) {
            console.error('Error fetching behavior data:', error)
            // Fallback nếu API lỗi
            setBehaviorData(getDefaultData())
            setSummary(prev => ({
                ...prev,
                dataFreshness: 'offline'
            }))
        } finally {
            setLoading(false)
        }
    }

    const calculateBehaviorDistributionFromRealTime = (students) => {
        if (students.length === 0) {
            return getDefaultData()
        }

        // Initialize behavior categories từ AI behavior detection
        const behaviorCategories = {
            'writing': 0,
            'raising_hand': 0,
            'look_straight': 0,
            'look_around': 0,
        }

        // Map behavior từ AI system
        students.forEach(student => {
            const behavior = (student.latest_behavior || 'normal').toLowerCase()

            if (behavior.includes('writing') || behavior.includes('write')) {
                behaviorCategories['writing']++
            } else if (behavior.includes('raising') || behavior.includes('raise') || behavior.includes('hand')) {
                behaviorCategories['raising_hand']++
            } else if (behavior.includes('straight') || behavior.includes('looking straight') || behavior.includes('focus')) {
                behaviorCategories['look_straight']++
            } else {
                behaviorCategories['look_around']++
            }
        })

        const totalStudents = students.length
        const totalDetected = Object.values(behaviorCategories).reduce((a, b) => a + b, 0)

        if (totalDetected === 0) {
            return getDefaultData()
        }

        // Chuẩn hóa percentages (tổng 100%)
        const scaleFactor = 100 / totalDetected

        // Tạo data cho chart với friendly names
        return [
            {
                name: 'Writing',
                value: Math.round(behaviorCategories['writing'] * scaleFactor),
                fill: '#10B981',
                description: 'Đang viết/take notes',
                icon: '✍️'
            },
            {
                name: 'Raising\nHand',
                value: Math.round(behaviorCategories['raising_hand'] * scaleFactor),
                fill: '#3B82F6',
                description: 'Giơ tay phát biểu',
                icon: '✋'
            },
            {
                name: 'Look\nStraight',
                value: Math.round(behaviorCategories['look_straight'] * scaleFactor),
                fill: '#8B5CF6',
                description: 'Nhìn thẳng/tập trung',
                icon: '👀'
            },
            {
                name: 'Look\nAround',
                value: Math.round(behaviorCategories['look_around'] * scaleFactor),
                fill: '#EF4444',
                description: 'Nhìn quanh/không tập trung',
                icon: '😕'
            },
        ]
    }

    const updateSummary = (students, freshness) => {
        if (students.length === 0) {
            setSummary({
                totalBehaviors: 0,
                topBehavior: 'normal',
                dataFreshness: freshness,
                lastUpdate: new Date().toLocaleTimeString()
            })
            return
        }

        // Tìm behavior phổ biến nhất
        const behaviorCounts = {}
        students.forEach(student => {
            const behavior = student.latest_behavior || 'normal'
            behaviorCounts[behavior] = (behaviorCounts[behavior] || 0) + 1
        })

        let topBehavior = 'normal'
        let maxCount = 0
        Object.entries(behaviorCounts).forEach(([behavior, count]) => {
            if (count > maxCount) {
                maxCount = count
                topBehavior = behavior
            }
        })

        // Định dạng behavior name cho đẹp
        const formattedTopBehavior = formatBehaviorName(topBehavior)

        setSummary({
            totalBehaviors: students.length,
            topBehavior: formattedTopBehavior,
            dataFreshness: freshness,
            lastUpdate: new Date().toLocaleTimeString()
        })
    }

    const formatBehaviorName = (behavior) => {
        const behaviorMap = {
            'writing': 'Writing',
            'raising_one_hand': 'Raising Hand',
            'raising_two_hands': 'Raising Both Hands',
            'look_straight': 'look_straight',
            'look_around': 'look_around',
        }

        return behaviorMap[behavior] ||
            behavior.split('_').map(word =>
                word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' ') ||
            'Normal'
    }

    const getDefaultData = () => {
        return [
            {
                name: 'Writing',
                value: 25,
                fill: '#10B981',
                description: 'Đang viết/take notes',
                icon: '✍️'
            },
            {
                name: 'Raising\nHand',
                value: 15,
                fill: '#3B82F6',
                description: 'Giơ tay phát biểu',
                icon: '✋'
            },
            {
                name: 'Look\nStraight',
                value: 35,
                fill: '#8B5CF6',
                description: 'Nhìn thẳng/tập trung',
                icon: '👀'
            },
            {
                name: 'Look\nAround',
                value: 15,
                fill: '#EF4444',
                description: 'Nhìn quanh/không tập trung',
                icon: '😕'
            },
        ]
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
                return 'AI Behavior Data'
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

    // Custom tooltip
    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload
            return (
                <div className="bg-gray-900 p-3 rounded-lg shadow-lg border border-gray-700">
                    <div className="flex items-center gap-2 mb-2">
                        <span className="text-xl">{data.icon}</span>
                        <p className="text-white font-medium">{data.name.replace('\n', ' ')}</p>
                    </div>
                    <p className="text-gray-300 text-sm mb-1">{data.description}</p>
                    <p className="text-white font-bold">
                        {data.value}% of students
                    </p>
                    <p className="text-gray-400 text-xs mt-1">
                        {Math.round((summary.totalBehaviors * data.value) / 100)} students
                    </p>
                </div>
            )
        }
        return null
    }

    // Hiển thị loading nếu đang tải dữ liệu
    if (loading) {
        return (
            <div className="bg-white rounded-xl p-6 h-full">
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-xl font-semibold text-black">AI Behavior Distribution</h2>
                    <span className={`px-2 py-1 text-xs font-medium rounded-full border ${getDataStatusColor()}`}>
                        {getDataStatusText()}
                    </span>
                </div>
                <div className="h-64 flex flex-col items-center justify-center">
                    <div className="text-gray-500 mb-2">Loading AI behavior data...</div>
                    <div className="text-xs text-gray-400">Detecting behaviors from camera feed</div>
                </div>
            </div>
        )
    }

    return (
        <div className="bg-white rounded-xl p-6 pt-10 h-full">
            {/* Chart */}
            <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                        data={behaviorData}
                        margin={{ top: 5, right: 10, left: 0, bottom: -15 }}
                    >
                        <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="#E5E7EB"
                            vertical={false}
                        />
                        <XAxis
                            dataKey="name"
                            stroke="#000000ff"
                            fontSize={11}
                            interval={0}
                            tickLine={false}
                            axisLine={false}
                            angle={0}
                            height={50}
                            textAnchor="middle"
                        />
                        <YAxis
                            stroke="#000000ff"
                            fontSize={11}
                            tickLine={false}
                            axisLine={false}
                            width={35}
                            domain={[0, 100]}
                            tickFormatter={(value) => `${value}%`}
                        />
                        <Tooltip content={<CustomTooltip />} />
                        <Bar
                            dataKey="value"
                            name="Percentage"
                            radius={[6, 6, 0, 0]}
                            barSize={45}
                            animationDuration={1000}
                        />
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    )
}

export default BehaviorDistribution