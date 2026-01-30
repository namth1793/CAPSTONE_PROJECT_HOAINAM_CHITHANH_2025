// frontend/app/components/DashboardStats.js
export default function DashboardStats({ stats }) {
  // Kiểm tra nếu stats không tồn tại hoặc không có dữ liệu
  if (!stats || !stats.stats) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {[...Array(6)].map((_, index) => (
          <div key={index} className="bg-white rounded-lg shadow p-4 animate-pulse">
            <div className="flex items-center space-x-3">
              <div className="bg-gray-300 w-12 h-12 rounded-lg"></div>
              <div className="space-y-2">
                <div className="h-4 bg-gray-300 rounded w-20"></div>
                <div className="h-6 bg-gray-300 rounded w-16"></div>
              </div>
            </div>
          </div>
        ))}
      </div>
    )
  }

  // Lấy data từ đúng cấu trúc API
  const data = stats.stats

  const statCards = [
    {
      title: 'Total',
      value: data.total_students || 0,
      color: 'bg-blue-500',
      icon: '👥',
    },
    {
      title: 'Present',
      value: data.present_count || 0,
      color: 'bg-green-500',
      icon: '✅',
    },
    {
      title: 'Absent',
      value: data.absent_count || 0,
      color: 'bg-red-500',
      icon: '❌',
    },
    {
      title: 'Attendance Rate',
      value: `${data.attendance_rate || 0}%` || 0,
      color: 'bg-purple-500',
      icon: '📊',
    },
    {
      title: 'Average Focus Score',
      value: `${data.avg_focus_score || 0}%`,
      color: 'bg-orange-500',
      icon: '🎯',
    },
    {
      title: 'Top Emotion',
      value: data.top_emotion || 'neutral',
      color: 'bg-pink-500',
      icon: getEmotionIcon(data.top_emotion),
    }
  ]

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      {statCards.map((card, index) => (
        <div
          key={index}
          className="bg-white text-black rounded-lg p-4 hover:border-gray-600 transition-all hover:shadow-lg hover:shadow-gray-900/30"
        >
          <div className="flex items-center space-x-3">
            <div className={`${card.color} w-12 h-12 rounded-lg flex items-center justify-center text-white shadow-md`}>
              <span className="text-2xl">{card.icon}</span>
            </div>
            <div>
              <h3 className="text-xs font-medium uppercase tracking-wider">
                {card.title}
              </h3>
              <p className="text-xl font-bold mt-1">
                {card.value}
              </p>
              <p className="text-xs mt-1">
                {card.description}
              </p>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

// Helper function để lấy icon cảm xúc
function getEmotionIcon(emotion) {
  const emotionIcons = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'neutral': '😐',
    'surprised': '😲',
    'disgusted': '🤢',
    'fearful': '😨'
  }
  return emotionIcons[emotion] || '😐'
}