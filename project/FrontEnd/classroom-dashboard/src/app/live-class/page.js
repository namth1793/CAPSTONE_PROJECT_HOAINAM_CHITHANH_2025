'use client'

import { useEffect, useRef, useState } from 'react'
import { Camera, Power, RefreshCw, Zap, AlertCircle, CheckCircle, XCircle, Settings } from 'lucide-react'
import { buildCameraApiUrl, getVideoFeedUrl } from '../config/api'

export default function LiveCameraStream() {
    const [isCameraOn, setIsCameraOn] = useState(false)
    const [cameraError, setCameraError] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const [connectionStatus, setConnectionStatus] = useState('disconnected')
    const [retryCount, setRetryCount] = useState(0)
    const [frameCount, setFrameCount] = useState(0)
    const [fps, setFps] = useState(0)
    
    const checkTimeoutRef = useRef(null)
    const frameIntervalRef = useRef(null)
    const statusCheckRef = useRef(null)
    const lastFrameTimeRef = useRef(0)
    const frameTimesRef = useRef([])
    const streamImgRef = useRef(null)

    // Flask server URLs - dynamic based on current hostname
    const getVideoFeed = () => getVideoFeedUrl()
    const getApiBase = () => buildCameraApiUrl('/api')

    // Hàm chuyển đổi camera - bật/tắt
    const toggleCamera = async () => {
        if (isCameraOn) {
            await stopCamera()
        } else {
            await startCamera()
        }
    }

    // Hàm kiểm tra Flask server
    const checkServerConnection = async () => {
        console.log('🔄 Checking Flask server connection...')
        setConnectionStatus('connecting')

        try {
            const controller = new AbortController()
            const timeoutId = setTimeout(() => controller.abort(), 3000)

            const response = await fetch(`${getApiBase()}/health`, {
                method: 'GET',
                signal: controller.signal,
                mode: 'cors',
                cache: 'no-cache'
            })

            clearTimeout(timeoutId)

            if (response.ok) {
                const data = await response.json()
                console.log('✅ Flask server is running:', data)
                setConnectionStatus('connected')
                setCameraError('')
                setRetryCount(0)
                return true
            } else {
                console.error('❌ Invalid server response:', response.status)
                setConnectionStatus('error')
                setCameraError(`Server lỗi: ${response.status}`)
                return false
            }
        } catch (error) {
            console.error('❌ Error checking server:', error)
            setConnectionStatus('error')
            setCameraError('Không thể kết nối đến Flask server')
            return false
        }
    }

    // Hàm bật camera và bắt đầu stream
    const startCamera = async () => {
        if (isCameraOn) return;

        console.log('🔄 Starting camera stream...')
        setCameraError('')
        setIsLoading(true)

        try {
            // Kiểm tra server trước
            const isConnected = await checkServerConnection()

            if (isConnected) {
                console.log('✅ Flask server đã kết nối')
                setIsCameraOn(true)
                setRetryCount(0)

                // Bắt đầu các interval check
                startFrameCounter()
                startStatusCheck()
            } else {
                // Thử lại sau 2 giây
                if (retryCount < 3) {
                    setRetryCount(prev => prev + 1)
                    setTimeout(() => {
                        startCamera()
                    }, 2000)
                    return
                }
                throw new Error('Không thể kết nối đến Flask server sau nhiều lần thử')
            }

        } catch (error) {
            console.error('❌ Error starting camera:', error)
            setCameraError(`Lỗi: ${error.message || 'Không thể kết nối'}`)
            setConnectionStatus('error')
        } finally {
            setIsLoading(false)
        }
    }

    // Hàm đếm FPS
    const updateFps = () => {
        const now = Date.now()
        if (lastFrameTimeRef.current > 0) {
            const delta = now - lastFrameTimeRef.current
            frameTimesRef.current.push(delta)

            // Giữ 60 frame times gần nhất
            if (frameTimesRef.current.length > 60) {
                frameTimesRef.current.shift()
            }

            // Tính FPS trung bình
            if (frameTimesRef.current.length > 0) {
                const avgDelta = frameTimesRef.current.reduce((a, b) => a + b) / frameTimesRef.current.length
                const currentFps = Math.round(1000 / avgDelta)
                setFps(currentFps)
            }
        }
        lastFrameTimeRef.current = now
    }

    // Hàm đếm frame và update stream image
    const startFrameCounter = () => {
        if (frameIntervalRef.current) {
            clearInterval(frameIntervalRef.current)
        }

        setFrameCount(0)
        lastFrameTimeRef.current = 0
        frameTimesRef.current = []

        // Auto-refresh stream image để tránh cache
        frameIntervalRef.current = setInterval(() => {
            setFrameCount(prev => prev + 1)
            updateFps()

            // Force refresh image src để tránh cache
            if (streamImgRef.current) {
                streamImgRef.current.src = `${getVideoFeed()}?t=${Date.now()}`
            }
        }, 100) // Update mỗi 100ms
    }

    // Hàm kiểm tra status định kỳ
    const startStatusCheck = () => {
        if (statusCheckRef.current) {
            clearInterval(statusCheckRef.current)
        }

        statusCheckRef.current = setInterval(async () => {
            await checkServerConnection()
        }, 5000) // Kiểm tra mỗi 5 giây
    }

    // Hàm tắt camera
    const stopCamera = async () => {
        console.log('🛑 Stopping camera...')
        setIsCameraOn(false)
        setConnectionStatus('disconnected')
        setRetryCount(0)
        setFrameCount(0)
        setFps(0)

        // Clear tất cả timeout và interval
        if (checkTimeoutRef.current) {
            clearTimeout(checkTimeoutRef.current)
            checkTimeoutRef.current = null
        }

        if (frameIntervalRef.current) {
            clearInterval(frameIntervalRef.current)
            frameIntervalRef.current = null
        }

        if (statusCheckRef.current) {
            clearInterval(statusCheckRef.current)
            statusCheckRef.current = null
        }
    }

    // Hàm reload stream
    const reloadStream = () => {
        console.log('🔄 Reloading stream...')
        if (isCameraOn) {
            // Reset frame counter
            setFrameCount(0)
            lastFrameTimeRef.current = 0
            frameTimesRef.current = []
        }
    }

    // Hàm test connection
    const testConnection = async () => {
        console.log('🧪 Manual connection test...')
        setIsLoading(true)

        try {
            const response = await fetch(`${getApiBase()}/health`, {
                method: 'GET',
                mode: 'cors'
            })

            const data = await response.json()

            alert(`✅ Server đang chạy!\n\nStatus: ${data.status}\nCamera: ${data.camera_source || 'Webcam'}`)

            return true
        } catch (error) {
            console.error('Test connection error:', error)
            alert(`❌ Lỗi kết nối: ${error.message}\n\nKiểm tra:\n1. Flask server có đang chạy không?\n2. Port 5000 có bị chặn không?`)
            return false
        } finally {
            setIsLoading(false)
        }
    }

    // Hàm lấy danh sách camera
    const listCameras = async () => {
        try {
            setIsLoading(true)
            const response = await fetch(`${getApiBase()}/camera/list`, {
                method: 'GET',
                mode: 'cors'
            })

            if (response.ok) {
                const data = await response.json()
                alert(`📷 Available Cameras:\n\n${data.cameras.map((cam, idx) => 
                    `${idx + 1}. Camera ${cam.index} - ${cam.resolution} @ ${cam.fps}fps`
                ).join('\n')}`)
            }
        } catch (error) {
            console.error('Error listing cameras:', error)
            alert('Không thể lấy danh sách camera')
        } finally {
            setIsLoading(false)
        }
    }

    // Tự động thử kết nối khi component mount
    useEffect(() => {
        checkServerConnection()

        return () => {
            // Cleanup
            if (checkTimeoutRef.current) {
                clearTimeout(checkTimeoutRef.current)
            }
            if (frameIntervalRef.current) {
                clearInterval(frameIntervalRef.current)
            }
            if (statusCheckRef.current) {
                clearInterval(statusCheckRef.current)
            }
        }
    }, [])

    // Helper functions cho UI
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

    // Lấy thời gian hiện tại
    const getCurrentDateTime = () => {
        const now = new Date()
        return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`
    }

    // Thêm fallback khi stream không hiển thị
    const renderStreamFallback = () => {
        return (
            <div className="w-full h-full flex flex-col items-center justify-center bg-gradient-to-br from-gray-900 to-black">
                <div className="text-6xl mb-4 animate-pulse">📹</div>
                <p className="text-white text-lg mb-2 font-semibold">Camera Stream</p>
                <p className="text-gray-400 mb-4 text-center max-w-md">
                    {cameraError || 'Nhấn "Bật Camera" để xem stream trực tiếp từ Flask server'}
                </p>
                {cameraError && (
                    <div className="mt-4 p-4 bg-red-900/20 border border-red-800 rounded-lg max-w-md">
                        <p className="text-red-300 font-medium mb-2">🔧 Khắc phục sự cố:</p>
                        <div className="flex gap-3 justify-center">
                            <button
                                onClick={testConnection}
                                className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded text-sm flex items-center"
                            >
                                <Zap className="w-4 h-4 mr-1" />
                                Test Server
                            </button>
                            <button
                                onClick={reloadStream}
                                className="bg-gray-700 hover:bg-gray-600 text-white px-3 py-1.5 rounded text-sm flex items-center"
                            >
                                <RefreshCw className="w-4 h-4 mr-1" />
                                Thử lại
                            </button>
                        </div>
                    </div>
                )}
            </div>
        )
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black p-4 md:p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="mb-6 md:mb-8">
                    <h1 className="text-2xl md:text-3xl font-bold text-white mb-2">📹 Camera Live Stream</h1>
                    <p className="text-gray-400">Real-time camera feed without AI processing</p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 md:gap-6">
                    {/* Main Video Feed */}
                    <div className="lg:col-span-3">
                        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl md:rounded-2xl shadow-2xl overflow-hidden border border-gray-700">
                            {/* Live Header */}
                            <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-4 border-b border-gray-700">
                                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                                    <h2 className="text-lg md:text-xl font-bold text-white flex items-center">
                                        <Camera className="mr-2 w-5 h-5" />
                                        Live Camera Stream
                                        <span className={`ml-2 w-2 h-2 rounded-full ${getConnectionStatusColor()} animate-pulse`}></span>
                                    </h2>
                                    <div className="flex flex-wrap items-center gap-3 text-sm">
                                        <div className="text-gray-300">{getCurrentDateTime()}</div>
                                    </div>
                                </div>
                            </div>

                            {/* Video Container */}
                            <div className="relative bg-black">
                                <div className="relative mx-auto" style={{ width: '640px', height: '480px' }}>
                                    {isCameraOn && connectionStatus === 'connected' ? (
                                        <div className="w-full h-full bg-black">
                                            {/* MJPEG Stream Image */}
                                            <img
                                                ref={streamImgRef}
                                                src={`${getVideoFeed()}?t=${Date.now()}`}
                                                className="w-full h-full object-contain"
                                                alt="Camera Stream"
                                                onError={(e) => {
                                                    console.error('Stream image error')
                                                    e.currentTarget.src = '/api/placeholder/640/480'
                                                }}
                                                onLoad={updateFps}
                                            />

                                            {/* Overlay information */}
                                            <div className="absolute top-0 left-0 right-0 p-3 bg-gradient-to-b from-black/70 to-transparent">
                                                <div className="flex justify-between text-sm">
                                                    <div className="flex items-center gap-3">
                                                        <div className="flex items-center gap-1">
                                                            <div className={`w-2 h-2 rounded-full ${getConnectionStatusColor()}`}></div>
                                                            <span className="text-white font-medium">LIVE</span>
                                                        </div>
                                                        <span className="text-gray-300">FPS: {fps}</span>
                                                        <span className="text-gray-300">Frames: {frameCount}</span>
                                                    </div>
                                                    <div className="text-gray-300">
                                                        Flask Server: {connectionStatus === 'connected' ? 'Online' : 'Offline'}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    ) : (
                                        renderStreamFallback()
                                    )}

                                    {/* Loading Overlay */}
                                    {isLoading && (
                                        <div className="absolute inset-0 bg-black/90 flex items-center justify-center z-10">
                                            <div className="text-center text-white">
                                                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
                                                <p className="text-lg font-medium">
                                                    {connectionStatus === 'connecting' ? 'Đang kết nối đến server...' : 'Đang xử lý...'}
                                                </p>
                                                {retryCount > 0 && (
                                                    <p className="text-sm text-gray-400 mt-2">
                                                        Thử lại lần {retryCount + 1}...
                                                    </p>
                                                )}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Control Bar */}
                            <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-4 border-t border-gray-700">
                                <div className="flex flex-wrap items-center justify-between gap-3">
                                    <div className="flex items-center gap-4">
                                        <div className="flex items-center gap-2">
                                            <div className={`w-3 h-3 rounded-full ${getConnectionStatusColor()}`}></div>
                                            <span className="text-gray-300">
                                                Server: {getConnectionStatusText()}
                                            </span>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <span className="text-gray-300 text-sm">
                                            Resolution: 640x480
                                        </span>
                                        <span className="text-gray-300 text-sm">
                                            Frame rate: 30 fps
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Camera Info Panel */}
                        <div className="mt-4 bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl overflow-hidden border border-gray-700">
                            <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-4 border-b border-gray-700">
                                <h3 className="text-lg font-bold text-white flex items-center">
                                    <Settings className="mr-2 w-5 h-5" />
                                    Camera Information
                                </h3>
                            </div>
                            <div className="p-4">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div className="space-y-3">
                                        <div className="flex justify-between items-center">
                                            <span className="text-gray-400">Stream Type:</span>
                                            <span className="text-white font-medium">MJPEG</span>
                                        </div>
                                        <div className="flex justify-between items-center">
                                            <span className="text-gray-400">Server URL:</span>
                                            <span className="text-blue-400 text-sm">http://localhost:5000</span>
                                        </div>
                                        <div className="flex justify-between items-center">
                                            <span className="text-gray-400">Endpoint:</span>
                                            <span className="text-green-400 font-medium">/video_feed</span>
                                        </div>
                                    </div>
                                    <div className="space-y-3">
                                        <div className="flex justify-between items-center">
                                            <span className="text-gray-400">Connection:</span>
                                            <span className={`font-medium ${connectionStatus === 'connected' ? 'text-green-400' : 'text-red-400'}`}>
                                                {getConnectionStatusText()}
                                            </span>
                                        </div>
                                        <div className="flex justify-between items-center">
                                            <span className="text-gray-400">Current FPS:</span>
                                            <span className="text-white font-medium">{fps}</span>
                                        </div>
                                        <div className="flex justify-between items-center">
                                            <span className="text-gray-400">Frames processed:</span>
                                            <span className="text-white font-medium">{frameCount}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Sidebar */}
                    <div className="lg:col-span-1 space-y-6">
                        {/* System Status */}
                        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl overflow-hidden border border-gray-700">
                            <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-4">
                                <h3 className="text-lg font-bold text-white">System Status</h3>
                            </div>
                            <div className="p-4 space-y-4">
                                <div className="flex items-center justify-between">
                                    <span className="text-gray-400">Flask Server</span>
                                    <div className="flex items-center gap-2">
                                        {connectionStatus === 'connected' ? (
                                            <CheckCircle className="w-4 h-4 text-green-500" />
                                        ) : connectionStatus === 'error' ? (
                                            <XCircle className="w-4 h-4 text-red-500" />
                                        ) : (
                                            <AlertCircle className="w-4 h-4 text-yellow-500" />
                                        )}
                                        <span className={`text-sm ${connectionStatus === 'connected' ? 'text-green-400' : connectionStatus === 'error' ? 'text-red-400' : 'text-yellow-400'}`}>
                                            {getConnectionStatusText()}
                                        </span>
                                    </div>
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-gray-400">Camera Stream</span>
                                    <div className="flex items-center gap-2">
                                        {isCameraOn ? (
                                            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                                        ) : (
                                            <div className="w-2 h-2 rounded-full bg-red-500"></div>
                                        )}
                                        <span className="text-white text-sm">{isCameraOn ? 'Active' : 'Inactive'}</span>
                                    </div>
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-gray-400">Resolution</span>
                                    <span className="text-blue-400 text-sm">640x480</span>
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-gray-400">Current FPS</span>
                                    <span className="text-white text-sm">{fps}</span>
                                </div>
                                <div className="pt-4 border-t border-gray-700">
                                    <div className="text-center text-gray-400 text-sm">
                                        Stream: {frameCount} frames processed
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Control Panel */}
                        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl overflow-hidden border border-gray-700">
                            <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-4">
                                <h3 className="text-lg font-bold text-white">Control Panel</h3>
                            </div>
                            <div className="p-4 space-y-3">
                                <button
                                    onClick={toggleCamera}
                                    disabled={isLoading}
                                    className={`w-full py-3 rounded-lg font-medium flex items-center justify-center gap-2 ${isCameraOn
                                            ? 'bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800'
                                            : 'bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800'
                                        } text-white disabled:opacity-50 disabled:cursor-not-allowed transition-all`}
                                >
                                    <Power className="w-4 h-4" />
                                    {isCameraOn ? 'Tắt Camera' : 'Bật Camera'}
                                </button>

                                <button
                                    onClick={testConnection}
                                    disabled={isLoading}
                                    className="w-full py-3 rounded-lg font-medium bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
                                >
                                    <Zap className="w-4 h-4" />
                                    Test Connection
                                </button>

                                <button
                                    onClick={reloadStream}
                                    disabled={!isCameraOn || isLoading}
                                    className="w-full py-3 rounded-lg font-medium bg-gradient-to-r from-gray-700 to-gray-800 hover:from-gray-800 hover:to-gray-900 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
                                >
                                    <RefreshCw className="w-4 h-4" />
                                    Refresh Stream
                                </button>

                                <button
                                    onClick={listCameras}
                                    disabled={isLoading}
                                    className="w-full py-3 rounded-lg font-medium bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
                                >
                                    <Settings className="w-4 h-4" />
                                    List Cameras
                                </button>

                                <a
                                    href={getVideoFeed()}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="block w-full py-3 rounded-lg font-medium bg-gradient-to-r from-gray-700 to-gray-800 hover:from-gray-800 hover:to-gray-900 text-white text-center transition-all flex items-center justify-center gap-2"
                                >
                                    <Camera className="w-4 h-4" />
                                    Open in New Tab
                                </a>
                            </div>
                        </div>

                        {/* Stream Stats */}
                        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl overflow-hidden border border-gray-700">
                            <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-4">
                                <h3 className="text-lg font-bold text-white">Stream Stats</h3>
                            </div>
                            <div className="p-4 space-y-4">
                                <div className="flex items-center justify-between">
                                    <span className="text-gray-400">Current FPS</span>
                                    <span className="text-green-400 font-medium">{fps}</span>
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-gray-400">Total Frames</span>
                                    <span className="text-white font-medium">{frameCount}</span>
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-gray-400">Resolution</span>
                                    <span className="text-blue-400 font-medium">640x480</span>
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-gray-400">Connection</span>
                                    <span className={`font-medium ${connectionStatus === 'connected' ? 'text-green-400' : 'text-red-400'}`}>
                                        {connectionStatus === 'connected' ? 'Stable' : 'Unstable'}
                                    </span>
                                </div>
                                <div className="pt-2">
                                    <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                                        <div 
                                            className="h-full bg-gradient-to-r from-green-500 to-blue-500 transition-all duration-300"
                                            style={{ width: `${Math.min(fps / 30 * 100, 100)}%` }}
                                        ></div>
                                    </div>
                                    <div className="text-xs text-gray-400 mt-1 text-center">
                                        {Math.round(fps / 30 * 100)}% of target FPS (30)
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <div className="mt-6 text-center text-gray-500 text-sm">
                    <p>Camera Stream System | Real-time MJPEG Video Feed</p>
                    <p>Server running on http://localhost:5000</p>
                </div>
            </div>
        </div>
    )
}