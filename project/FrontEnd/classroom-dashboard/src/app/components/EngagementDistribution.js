// frontend/components/EngagementDistribution.js
import { useEffect, useState } from 'react'
import { buildApiUrl } from '../config/api'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

const EngagementDistribution = () => {
    const [engagementData, setEngagementData] = useState([])
    const [loading, setLoading] = useState(false)
    const [summary, setSummary] = useState({
        totalStudents: 0,
        avgEngagement: 0,
        topEmotion: 'neutral',
        dataFreshness: 'demo'
    })

    useEffect(() => {
        fetchEngagementData()

        // Refresh data mỗi 30 giây để có real-time data
        const interval = setInterval(() => {
            fetchEngagementData()
        }, 30000) // 30 giây

        return () => clearInterval(interval)
    }, [])

    const fetchEngagementData = async () => {
        setLoading(true)
        try {
            // 🔴 SỬA: Gọi API engagement real-time mới
            const response = await fetch(buildApiUrl('/api/engagement/realtime?recent_minutes=10'))

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`)
            }

            const data = await response.json()

            if (data.status === 'success') {
                // Lưu summary data
                if (data.summary) {
                    setSummary({
                        totalStudents: data.summary.total_students || 0,
                        avgEngagement: data.summary.avg_engagement || 0,
                        topEmotion: data.summary.top_emotion || 'neutral',
                        dataFreshness: data.summary.data_freshness || 'demo'
                    })
                }

                // Tính toán engagement distribution từ students data
                const engagementDistribution = calculateEngagementDistributionFromStudents(data.students || [])
                setEngagementData(engagementDistribution)
            } else {
                // Fallback nếu không có dữ liệu
                setEngagementData(getDefaultData())
                setSummary({
                    totalStudents: 8,
                    avgEngagement: 72.5,
                    topEmotion: 'neutral',
                    dataFreshness: 'fallback'
                })
            }
        } catch (error) {
            console.error('Error fetching engagement data:', error)
            // Fallback nếu API lỗi
            setEngagementData(getDefaultData())
            setSummary({
                totalStudents: 8,
                avgEngagement: 72.5,
                topEmotion: 'neutral',
                dataFreshness: 'offline'
            })
        } finally {
            setLoading(false)
        }
    }

    const calculateEngagementDistributionFromStudents = (students) => {
        const ranges = [
            { range: '0-20%', count: 0, color: '#EF4444', description: 'Very Low' },
            { range: '21-40%', count: 0, color: '#F59E0B', description: 'Low' },
            { range: '41-60%', count: 0, color: '#EAB308', description: 'Medium' },
            { range: '61-80%', count: 0, color: '#84CC16', description: 'High' },
            { range: '81-100%', count: 0, color: '#10B981', description: 'Very High' }
        ]

        if (students.length === 0) {
            return getDefaultData()
        }

        // Tính toán distribution từ engagement scores
        students.forEach(student => {
            const engagement = student.latest_engagement || 0

            if (engagement <= 20) ranges[0].count++
            else if (engagement <= 40) ranges[1].count++
            else if (engagement <= 60) ranges[2].count++
            else if (engagement <= 80) ranges[3].count++
            else ranges[4].count++
        })

        return ranges
    }

    const getDefaultData = () => {
        return [
            { range: '0-20%', count: 1, color: '#EF4444', description: 'Very Low' },
            { range: '21-40%', count: 2, color: '#F59E0B', description: 'Low' },
            { range: '41-60%', count: 3, color: '#EAB308', description: 'Medium' },
            { range: '61-80%', count: 2, color: '#84CC16', description: 'High' },
            { range: '81-100%', count: 0, color: '#10B981', description: 'Very High' }
        ]
    }

    const getDataStatusText = () => {
        switch (summary.dataFreshness) {
            case 'live':
                return 'Live Data'
            case 'recent':
                return 'Recent Data'
            case 'demo':
                return 'Demo Data'
            case 'fallback':
                return 'Fallback Data'
            case 'offline':
                return 'Offline Mode'
            default:
                return 'Unknown'
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

    // Hiển thị loading nếu đang tải dữ liệu
    if (loading) {
        return (
            <div className="bg-white rounded-xl p-6 h-full">
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-xl font-semibold text-black">Engagement Distribution</h2>
                    <span className={`px-2 py-1 text-xs font-medium rounded-full border ${getDataStatusColor()}`}>
                        {getDataStatusText()}
                    </span>
                </div>
                <div className="h-64 flex items-center justify-center">
                    <div className="text-gray-500">Loading real-time engagement data...</div>
                </div>
            </div>
        )
    }

    return (
        <div className="bg-white rounded-xl pt-10 h-full">
            {/* Chart */}
            <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={engagementData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                        <XAxis
                            dataKey="range"
                            stroke="#000000ff"
                            fontSize={12}
                            tickLine={false}
                            axisLine={false}
                        />
                        <YAxis
                            stroke="#000000ff"
                            fontSize={12}
                            tickLine={false}
                            axisLine={false}
                            allowDecimals={false}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#000000ff',
                                border: 'none',
                                borderRadius: '8px',
                                color: 'white'
                            }}
                            itemStyle={{ color: 'white' }}
                            formatter={(value, name, props) => {
                                const description = props.payload.description || ''
                                return [
                                    `${value} student${value !== 1 ? 's' : ''}`,
                                    `${description} Engagement`
                                ]
                            }}
                            labelFormatter={(label) => `Engagement: ${label}`}
                        />
                        <Bar
                            dataKey="count"
                            name="Number of Students"
                            radius={[6, 6, 0, 0]}
                            barSize={35}
                        >
                            {engagementData.map((entry, index) => (
                                <Cell
                                    key={`cell-${index}`}
                                    fill={entry.color}
                                    strokeWidth={1}
                                    stroke="#1F2937"
                                />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    )
}

export default EngagementDistribution