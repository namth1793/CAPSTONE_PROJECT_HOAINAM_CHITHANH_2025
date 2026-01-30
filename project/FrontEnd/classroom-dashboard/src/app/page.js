// app/dashboard/page.jsx
'use client'

import { useEffect, useRef, useState } from 'react'
import BehaviorDistribution from './components/BehaviorDistribution'
import DashboardStats from './components/DashboardStats'
import EmotionChart from './components/EmotionChart'
import EngagementDistribution from './components/EngagementDistribution'
import ProtectedRoute from './components/ProtectedRoute'
import { useAuth } from './context/AuthContext'
import { buildApiUrl, buildWebSocketUrl } from './config/api'

export default function Dashboard() {
  const { user, loading: authLoading } = useAuth()
  const [stats, setStats] = useState(null)
  const [students, setStudents] = useState([])
  const [dataLoading, setDataLoading] = useState(true)
  const [modelStatus, setModelStatus] = useState('stopped')

  // WebSocket state
  const [wsConnected, setWsConnected] = useState(false)
  const [fps, setFps] = useState(0)
  const [detectionCount, setDetectionCount] = useState(0)
  const [realtimeUpdates, setRealtimeUpdates] = useState([])
  const wsRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)
  const pollingIntervalsRef = useRef({ stats: null, students: null, status: null })
  const pingIntervalRef = useRef(null)

  // Database Reset state
  const [resetLoading, setResetLoading] = useState(false)

  // ==================== DATABASE RESET FUNCTION ====================

  /**
   * Reset database - Đơn giản không cần token hay mật khẩu
   */
  // Trong file dashboard/page.jsx - Cập nhật hàm resetDatabase
  const resetDatabase = async () => {
    if (!confirm('⚠️ BẠN CÓ CHẮC MUỐN RESET DATABASE?\n\nToàn bộ dữ liệu sẽ bị xóa và tạo lại từ đầu.\nHành động này không thể hoàn tác!')) {
      return
    }

    setResetLoading(true)

    try {
      console.log('🔄 Resetting database...')

      const response = await fetch(buildApiUrl('/api/system/reset-database'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          confirm: true,
          create_sample_data: true
        })
      })

      const result = await response.json()

      if (response.ok) {
        alert(`✅ ${result.message}\n\nTài khoản demo:\n• demo / demo123 (teacher)\n• admin / admin123 (admin)\n\nTrang sẽ tự động reload sau 5 giây...`)

        // Tự động reload sau 5 giây
        setTimeout(() => {
          window.location.reload()
        }, 5000)
      } else {
        // Hiển thị lỗi chi tiết
        alert(`❌ Lỗi: ${result.message || 'Không thể reset database'}\n\n${result.suggestion || 'Vui lòng thử lại sau'}`)
      }
    } catch (error) {
      console.error('❌ Network error:', error)
      alert('❌ Lỗi kết nối đến server. Vui lòng kiểm tra:\n1. Backend có đang chạy không?\n2. Cổng 8000 có bị chặn không?')
    } finally {
      setResetLoading(false)
    }
  }

  // ==================== WEBSOCKET FUNCTIONS ====================

  const connectWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    const wsUrls = [
      buildWebSocketUrl('/ws/live'),
    ]

    let currentUrlIndex = 0

    const tryConnect = () => {
      if (currentUrlIndex >= wsUrls.length) {
        console.log('❌ All WebSocket URLs failed, falling back to polling')
        setWsConnected(false)
        startPolling()
        return
      }

      const url = wsUrls[currentUrlIndex]
      console.log(`🔌 Trying WebSocket: ${url}`)

      try {
        wsRef.current = new WebSocket(url)

        wsRef.current.onopen = () => {
          console.log(`✅ WebSocket connected to: ${url}`)
          setWsConnected(true)
          stopPolling()

          if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current)
            reconnectTimeoutRef.current = null
          }

          startPingInterval()

          wsRef.current.send(JSON.stringify({
            type: 'client_connect',
            client: 'dashboard',
            timestamp: new Date().toISOString(),
            message: 'Dashboard connected'
          }))
        }

        wsRef.current.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            console.log(`📩 WebSocket [${data.type}]:`, data)

            switch (data.type) {
              case 'batch_processed':
                console.log(`✅ Batch processed: ${data.processed_count} items`)
                fetchStudentsPolling()
                break

              case 'student_data_update':
                setRealtimeUpdates(prev => [
                  { type: 'student', data: data.data, timestamp: new Date() },
                  ...prev.slice(0, 9)
                ])
                fetchStudentsPolling()
                break

              case 'attendance_update':
                setRealtimeUpdates(prev => [
                  { type: 'attendance', data: data.data, timestamp: new Date() },
                  ...prev.slice(0, 9)
                ])
                fetchStatsPolling()
                break

              case 'engagement_update':
                setRealtimeUpdates(prev => [
                  { type: 'engagement', data: data.data, timestamp: new Date() },
                  ...prev.slice(0, 9)
                ])
                fetchStudentsPolling()
                break

              case 'database_reset':
                console.log('🔄 Database reset notification:', data.message)
                setRealtimeUpdates(prev => [
                  {
                    type: 'system',
                    data: {
                      student_name: 'System',
                      message: 'Database đã được reset và tạo mới'
                    },
                    timestamp: new Date()
                  },
                  ...prev.slice(0, 9)
                ])
                setTimeout(() => {
                  fetchStatsPolling()
                  fetchStudentsPolling()
                }, 3000)
                break

              case 'detection_update':
                setDetectionCount(data.detections?.length || 0)
                if (data.fps) {
                  setFps(Math.round(data.fps))
                }
                if (data.detections && data.detections.length > 0) {
                  const newStudents = data.detections.map(detection => ({
                    id: detection.id || `det_${Date.now()}_${Math.random()}`,
                    student_id: detection.student_id || detection.id,
                    student_name: detection.name || detection.student_name || 'Unknown Student',
                    engagement: detection.engagement || (detection.focus_score / 100) || 0,
                    focus_score: detection.focus_score || detection.engagement * 100 || 0,
                    emotion: detection.emotion || 'neutral',
                    behavior: detection.behavior || detection.behavior_details || 'normal',
                    emotion_confidence: detection.emotion_confidence || detection.confidence || 0.5,
                    concentration_level: detection.concentration_level ||
                      (detection.engagement > 0.7 ? 'high' :
                        detection.engagement > 0.5 ? 'medium' : 'low')
                  }))
                  setStudents(newStudents)
                }
                break

              case 'status':
                if (data.status) {
                  setModelStatus(data.status)
                }
                break

              case 'pong':
                console.log('🏓 Received pong')
                break

              case 'client_status':
                console.log('👤 Client status:', data)
                break

              default:
                console.log(`ℹ️ Unknown WebSocket message type: ${data.type}`)
                break
            }
          } catch (error) {
            console.error('❌ Error parsing WebSocket message:', error)
          }
        }

        wsRef.current.onclose = (event) => {
          console.log(`🔌 WebSocket disconnected: ${event.code} ${event.reason}`)
          setWsConnected(false)
          stopPingInterval()
          startPolling()

          if (!reconnectTimeoutRef.current) {
            reconnectTimeoutRef.current = setTimeout(() => {
              console.log('🔄 Attempting WebSocket reconnect...')
              currentUrlIndex = (currentUrlIndex + 1) % wsUrls.length
              tryConnect()
            }, 3000)
          }
        }

        wsRef.current.onerror = (error) => {
          console.error(`❌ WebSocket error on ${url}:`, error)
          setWsConnected(false)
          stopPingInterval()
          currentUrlIndex++
          setTimeout(tryConnect, 100)
        }

      } catch (error) {
        console.error(`❌ Error creating WebSocket for ${url}:`, error)
        currentUrlIndex++
        setTimeout(tryConnect, 1000)
      }
    }

    tryConnect()
  }

  const startPingInterval = () => {
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current)
    }

    pingIntervalRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        try {
          wsRef.current.send(JSON.stringify({
            type: 'ping',
            timestamp: new Date().toISOString(),
            client: 'dashboard'
          }))
        } catch (error) {
          console.error('❌ Error sending ping:', error)
        }
      }
    }, 15000)
  }

  const stopPingInterval = () => {
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current)
      pingIntervalRef.current = null
    }
  }

  // ==================== POLLING FUNCTIONS ====================

  const stopPolling = () => {
    Object.keys(pollingIntervalsRef.current).forEach(key => {
      if (pollingIntervalsRef.current[key]) {
        clearInterval(pollingIntervalsRef.current[key])
        pollingIntervalsRef.current[key] = null
      }
    })
    console.log('📡 Stopped all polling')
  }

  const startPolling = () => {
    if (wsConnected) {
      return
    }

    console.log('📡 Starting polling mode')

    if (!pollingIntervalsRef.current.stats) {
      pollingIntervalsRef.current.stats = setInterval(fetchStatsPolling, 15000)
      fetchStatsPolling()
    }

    if (!pollingIntervalsRef.current.students) {
      pollingIntervalsRef.current.students = setInterval(fetchStudentsPolling, 10000)
      fetchStudentsPolling()
    }

    if (!pollingIntervalsRef.current.status) {
      pollingIntervalsRef.current.status = setInterval(checkModelStatus, 5000)
      checkModelStatus()
    }
  }

  const fetchStatsPolling = async () => {
    if (wsConnected) {
      return
    }

    try {
      console.log('📡 Polling: Fetching class dashboard stats...')
      const statsRes = await fetch(buildApiUrl('/api/class/dashboard-stats'))

      if (!statsRes.ok) {
        throw new Error(`HTTP error! status: ${statsRes.status}`)
      }

      const statsData = await statsRes.json()
      console.log('✅ Class dashboard stats loaded')

      setStats({
        status: "success",
        stats: {
          total_students: statsData.summary?.total_students || 0,
          present_count: statsData.summary?.present_count || 0,
          absent_count: statsData.summary?.absent_count || 0,
          attendance_rate: statsData.summary?.attendance_rate || 0,
          detection_rate: statsData.realtime_stats?.detection_rate || 0,
          avg_focus_score: statsData.realtime_stats?.avg_focus_score || 0,
          avg_behavior_score: statsData.realtime_stats?.avg_behavior_score || 0,
          top_emotion: statsData.realtime_stats?.top_emotion || "neutral",
          system_status: "online",
          last_update: new Date().toISOString()
        },
        students: statsData.students || []
      })

    } catch (error) {
      console.error('❌ Fetch class stats error:', error)
      if (!stats) {
        const mockStats = {
          status: "success",
          stats: {
            total_students: 25,
            present_count: 22,
            absent_count: 3,
            attendance_rate: 88.0,
            detection_rate: 80.0,
            avg_focus_score: 78.5,
            avg_behavior_score: 82.3,
            top_emotion: "happy",
            system_status: "online",
            last_update: new Date().toISOString()
          }
        }
        setStats(mockStats)
      }
    }
  }

  const fetchStudentsPolling = async () => {
    if (wsConnected) {
      return
    }

    try {
      console.log('📡 Polling: Fetching students data...')
      const studentsRes = await fetch(buildApiUrl('/api/students?recent_minutes=5&limit=20'))

      if (!studentsRes.ok) {
        throw new Error(`HTTP error! status: ${studentsRes.status}`)
      }

      const studentsData = await studentsRes.json()
      console.log(`✅ Students data loaded: ${studentsData.students?.length || 0} students`)

      if (studentsData.students && studentsData.students.length > 0) {
        const formattedStudents = studentsData.students.map(student => ({
          id: student.student_id,
          student_id: student.student_id,
          student_name: student.student_name,
          engagement: student.stats?.avg_focus / 100 || 0.75,
          focus_score: student.stats?.avg_focus || 75,
          emotion: student.latest_emotion?.emotion || 'neutral',
          behavior: student.latest_behavior?.details || 'normal',
          emotion_confidence: student.latest_emotion?.confidence || 0.5,
          concentration_level: student.stats?.avg_focus > 80 ? 'high' :
            student.stats?.avg_focus > 60 ? 'medium' : 'low',
          class_name: student.class_name || 'AI Class',
          last_recorded: student.last_recorded
        }))
        setStudents(formattedStudents)
        setDetectionCount(formattedStudents.length)
      }

    } catch (error) {
      console.error('❌ Fetch students error:', error)
      if (students.length === 0) {
        const mockStudents = [
          {
            id: 'AI_001',
            student_id: 'AI_001',
            student_name: 'Student 1',
            engagement: 0.85,
            focus_score: 85,
            emotion: 'happy',
            behavior: 'writing',
            emotion_confidence: 0.8,
            concentration_level: 'high',
            class_name: 'AI Class'
          },
          {
            id: 'AI_002',
            student_id: 'AI_002',
            student_name: 'Student 2',
            engagement: 0.72,
            focus_score: 72,
            emotion: 'neutral',
            behavior: 'look_straight',
            emotion_confidence: 0.7,
            concentration_level: 'medium',
            class_name: 'AI Class'
          }
        ]
        setStudents(mockStudents)
        setDetectionCount(mockStudents.length)
      }
    }
  }

  const checkModelStatus = async () => {
    if (wsConnected) {
      return
    }

    try {
      const modelRes = await fetch('http://localhost:5000/api/status')

      if (modelRes.ok) {
        const modelData = await modelRes.json()
        setModelStatus(modelData.status || 'stopped')
      } else {
        try {
          const altRes = await fetch('http://localhost:5000/api/health')
          if (altRes.ok) {
            setModelStatus('running')
          } else {
            setModelStatus('stopped')
          }
        } catch {
          setModelStatus('stopped')
        }
      }
    } catch (error) {
      console.error('Error checking model status:', error)
      setModelStatus('stopped')
    }
  }

  const startModel = async () => {
    if (modelStatus === 'running') {
      alert('Model đã đang chạy!')
      return
    }

    setModelStatus('starting')

    try {
      const response = await fetch('http://localhost:5000/api/control', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ action: 'start' })
      })

      if (response.ok) {
        const data = await response.json()
        setModelStatus('running')
        alert(data.message || 'AI model đã khởi động thành công!')
        fetchStatsPolling()
        fetchStudentsPolling()
      } else {
        const altResponse = await fetch('http://localhost:5000/api/start_ai', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          }
        })

        if (altResponse.ok) {
          setModelStatus('running')
          alert('AI model đã khởi động thành công!')
          fetchStatsPolling()
          fetchStudentsPolling()
        } else {
          setModelStatus('error')
          alert('Không thể khởi động model. Vui lòng thử lại.')
        }
      }
    } catch (error) {
      console.error('Error starting model:', error)
      setModelStatus('error')
      alert('Lỗi kết nối đến AI server.')
    }
  }

  const stopModel = async () => {
    if (modelStatus !== 'running') {
      alert('Model không đang chạy!')
      return
    }

    try {
      const response = await fetch('http://localhost:5000/api/control', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ action: 'stop' })
      })

      if (response.ok) {
        setModelStatus('stopped')
        alert('AI model đã dừng thành công!')
      } else {
        const altResponse = await fetch('http://localhost:5000/api/stop_ai', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          }
        })

        if (altResponse.ok) {
          setModelStatus('stopped')
          alert('AI model đã dừng thành công!')
        } else {
          alert('Không thể dừng model. Vui lòng thử lại.')
        }
      }
    } catch (error) {
      console.error('Error stopping model:', error)
      alert('Lỗi kết nối đến AI server.')
    }
  }

  // ==================== USE EFFECT HOOKS ====================

  useEffect(() => {
    if (user && !authLoading) {
      console.log('🔄 Initializing dashboard...')

      const loadInitialData = async () => {
        try {
          await Promise.all([
            fetchStatsPolling(),
            fetchStudentsPolling(),
            checkModelStatus()
          ])
        } catch (error) {
          console.error('Initial data load error:', error)
        } finally {
          setDataLoading(false)
        }
      }

      loadInitialData()
      connectWebSocket()
    }

    return () => {
      console.log('🧹 Cleaning up dashboard...')
      stopPolling()
      stopPingInterval()

      if (wsRef.current) {
        wsRef.current.close()
      }

      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
    }
  }, [user, authLoading])

  useEffect(() => {
    if (!wsConnected && user && !authLoading) {
      const timeoutId = setTimeout(() => {
        if (!wsConnected) {
          startPolling()
        }
      }, 2000)

      return () => clearTimeout(timeoutId)
    }
  }, [wsConnected, user, authLoading])

  useEffect(() => {
    console.log(`🌐 WebSocket: ${wsConnected ? 'Connected' : 'Disconnected'}`)
    console.log(`📡 Polling: stats=${!!pollingIntervalsRef.current.stats}, students=${!!pollingIntervalsRef.current.students}, status=${!!pollingIntervalsRef.current.status}`)

    if (wsConnected && wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({
          type: 'client_status',
          status: 'connected',
          client: 'dashboard',
          timestamp: new Date().toISOString()
        }))
      } catch (error) {
        console.error('Error sending client status:', error)
      }
    }
  }, [wsConnected])

  // Hiển thị loading khi đang xác thực
  if (authLoading) {
    return (
      <div className="flex justify-center items-center min-h-screen bg-black">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    )
  }

  return (
    <ProtectedRoute>
      <div className="p-6 bg-[#B39858] h-full">
        <div className="max-w-7xl mx-auto">
          {/* Header với nút Reset Database */}
          <div className="flex justify-between items-center mb-8">
            <div>
              <h1 className="text-3xl font-bold text-white">AI Classroom Dashboard</h1>
              <p className="text-white">Chào mừng trở lại, {user?.full_name || user?.username}!</p>
              {realtimeUpdates.length > 0 && (
                <div className="mt-2 text-sm text-gray-400">
                  📢 {realtimeUpdates.length} real-time update{realtimeUpdates.length > 1 ? 's' : ''}
                </div>
              )}
            </div>

            <div className="flex items-center space-x-4">
              {/* Trạng thái kết nối */}
              <div className={`px-3 py-1 rounded-full text-sm font-medium flex items-center ${wsConnected ? 'bg-green-900/30 text-green-400' : 'bg-yellow-900/30 text-yellow-400'
                }`}>
                <span className="mr-2">{wsConnected ? '🔌' : '📡'}</span>
                {wsConnected ? 'Real-time' : 'Polling'}
                {!wsConnected && pollingIntervalsRef.current.stats && (
                  <span className="ml-2 text-xs">(15s/10s)</span>
                )}
              </div>

              {/* Trạng thái model */}
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${modelStatus === 'running' ? 'bg-green-900/30 text-green-400' :
                modelStatus === 'starting' ? 'bg-yellow-900/30 text-yellow-400' :
                  modelStatus === 'error' ? 'bg-red-900/30 text-red-400' :
                    'bg-gray-800 text-gray-400'
                }`}>
                {modelStatus === 'running' ? '🤖 AI Running' :
                  modelStatus === 'starting' ? '🔄 Starting' :
                    modelStatus === 'error' ? '❌ AI Error' :
                      '⏸️ AI Stopped'}
              </div>

              {/* Nút điều khiển model */}
              <div className="flex space-x-2">
                {modelStatus === 'running' ? (
                  <button
                    onClick={stopModel}
                    className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors flex items-center"
                  >
                    <span className="mr-2">⏹️</span>
                    Stop AI
                  </button>
                ) : (
                  <button
                    onClick={startModel}
                    disabled={modelStatus === 'starting'}
                    className={`px-4 py-2 ${modelStatus === 'starting'
                      ? 'bg-green-700 cursor-not-allowed'
                      : 'bg-green-600 hover:bg-green-700'
                      } text-white rounded-lg font-medium transition-colors flex items-center`}
                  >
                    {modelStatus === 'starting' ? (
                      <>
                        <span className="animate-spin mr-2">⟳</span>
                        Starting...
                      </>
                    ) : (
                      <>
                        <span className="mr-2">🚀</span>
                        Start AI Model
                      </>
                    )}
                  </button>
                )}
              </div>
              {/* Nút Reset Database - ĐƠN GIẢN */}
              <button
                onClick={resetDatabase}
                disabled={resetLoading}
                className={`px-4 py-2 rounded-lg font-medium transition-colors flex items-center ${resetLoading
                  ? 'bg-red-800 cursor-not-allowed'
                  : 'bg-red-600 hover:bg-red-700'
                  } text-white`}
              >
                {resetLoading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Resetting...
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
                    </svg>
                    Reset Database
                  </>
                )}
              </button>
            </div>
          </div>

          {dataLoading ? (
            <div className="flex justify-center items-center min-h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
              <span className="ml-4 text-white">Đang tải dữ liệu...</span>
            </div>
          ) : (
            <>
              <DashboardStats stats={stats} />

              {/* Grid layout cho 3 biểu đồ chính */}
              <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 mt-6">
                <div className="xl:col-span-1 mb-10">
                  <h2 className="text-xl font-semibold text-white mb-2">Emotion Distribution</h2>
                  <EmotionChart />
                </div>

                <div className="xl:col-span-1 mb-10">
                  <h2 className="text-xl font-semibold text-white mb-2">Behavior Distribution</h2>
                  <BehaviorDistribution />
                </div>

                <div className="xl:col-span-1 mb-10">
                  <h2 className="text-xl font-semibold text-white mb-2">Engagement Distribution</h2>
                  <EngagementDistribution students={students} />
                </div>
              </div>

              {/* Real-time Updates Panel */}
              {realtimeUpdates.length > 0 && (
                <div className="mt-6 p-4 bg-gray-900/50 rounded-lg border border-gray-800">
                  <h3 className="text-lg font-semibold text-white mb-3 flex items-center">
                    <span className="mr-2">📢</span>
                    Real-time Updates
                  </h3>
                  <div className="space-y-2 max-h-40 overflow-y-auto">
                    {realtimeUpdates.map((update, index) => (
                      <div key={index} className="p-3 bg-gray-800/50 rounded border border-gray-700">
                        <div className="flex justify-between items-start">
                          <div>
                            <span className={`inline-block px-2 py-1 rounded text-xs mr-2 ${update.type === 'student' ? 'bg-blue-900/50 text-blue-400' :
                              update.type === 'attendance' ? 'bg-green-900/50 text-green-400' :
                                'bg-purple-900/50 text-purple-400'
                              }`}>
                              {update.type === 'student' ? '👤 Student' :
                                update.type === 'attendance' ? '✅ Attendance' :
                                  '🎯 Engagement'}
                            </span>
                            <span className="text-white text-sm">
                              {update.data?.student_name || 'Unknown'}
                            </span>
                          </div>
                          <span className="text-xs text-gray-500">
                            {new Date(update.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </span>
                        </div>
                        {update.data?.emotion && (
                          <div className="mt-1 text-sm text-gray-400">
                            😊 {update.data.emotion} | 🎯 {update.data.behavior || 'normal'}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Thông tin hệ thống */}
              <div className="mt-6 p-4 bg-gray-900/50 rounded-lg border border-gray-800">
                <h3 className="text-lg font-semibold text-white mb-3">📊 System Status</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-4 bg-gray-800/50 rounded-lg">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">AI Model</span>
                      <span className={`px-2 py-1 rounded text-xs ${modelStatus === 'running' ? 'bg-green-900/50 text-green-400' :
                        modelStatus === 'starting' ? 'bg-yellow-900/50 text-yellow-400' :
                          'bg-gray-800 text-gray-400'
                        }`}>
                        {modelStatus === 'running' ? 'Running' :
                          modelStatus === 'starting' ? 'Starting' :
                            'Stopped'}
                      </span>
                    </div>
                    <p className="text-sm text-gray-500 mt-2">
                      Face recognition + Emotion + Behavior
                    </p>
                    <div className="mt-2 text-xs text-gray-600">
                      Port: 5000 | <a href="http://localhost:5000/api/status" target="_blank" className="text-blue-400 hover:underline">Status</a>
                    </div>
                  </div>

                  <div className="p-4 bg-gray-800/50 rounded-lg">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Backend API</span>
                      <span className="px-2 py-1 rounded text-xs bg-green-900/50 text-green-400">
                        Online
                      </span>
                    </div>
                    <p className="text-sm text-gray-500 mt-2">
                      Database + Real-time processing
                    </p>
                    <div className="mt-2 text-xs text-gray-600">
                      Port: 8000 | <a href={buildApiUrl('/api/health')} target="_blank" className="text-blue-400 hover:underline">Health</a>
                    </div>
                  </div>

                  <div className="p-4 bg-gray-800/50 rounded-lg">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">WebSocket</span>
                      <span className={`px-2 py-1 rounded text-xs ${wsConnected ? 'bg-green-900/50 text-green-400' : 'bg-yellow-900/50 text-yellow-400'
                        }`}>
                        {wsConnected ? 'Connected' : 'Disconnected'}
                      </span>
                    </div>
                    <p className="text-sm text-gray-500 mt-2">
                      Real-time data streaming
                    </p>
                    <div className="mt-2 text-xs text-gray-600">
                      {wsConnected ? buildWebSocketUrl('/ws/live') : 'Reconnecting...'}
                      {!wsConnected && (
                        <div className="mt-1 text-yellow-500 text-xs">
                          Fallback to polling mode
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </ProtectedRoute>
  )
}