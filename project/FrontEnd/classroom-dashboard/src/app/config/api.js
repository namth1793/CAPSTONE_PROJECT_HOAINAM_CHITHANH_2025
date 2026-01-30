// Shared API URL helpers
export const getApiBaseUrl = () => {
    const envUrl = process.env.NEXT_PUBLIC_API_BASE_URL
    if (envUrl && typeof envUrl === 'string' && envUrl.trim().length > 0) {
        return envUrl.replace(/\/$/, '')
    }

    if (typeof window !== 'undefined') {
        const { protocol, hostname } = window.location
        const defaultPort = '8000'
        return `${protocol}//${hostname}:${defaultPort}`
    }

    return 'http://localhost:8000'
}

// Camera API URL (Flask server on port 5000)
export const getCameraApiBaseUrl = () => {
    const envUrl = process.env.NEXT_PUBLIC_CAMERA_API_URL
    if (envUrl && typeof envUrl === 'string' && envUrl.trim().length > 0) {
        return envUrl.replace(/\/$/, '')
    }

    if (typeof window !== 'undefined') {
        const { protocol, hostname } = window.location
        const cameraPort = '5000'
        return `${protocol}//${hostname}:${cameraPort}`
    }

    return 'http://localhost:5000'
}

export const buildCameraApiUrl = (path = '') => {
    const baseUrl = getCameraApiBaseUrl()
    const normalizedPath = path.startsWith('/') ? path : `/${path}`
    return `${baseUrl}${normalizedPath}`
}

export const getVideoFeedUrl = () => {
    return buildCameraApiUrl('/video_feed')
}

export const buildApiUrl = (path = '') => {
    const baseUrl = getApiBaseUrl()
    const normalizedPath = path.startsWith('/') ? path : `/${path}`
    return `${baseUrl}${normalizedPath}`
}

export const buildWebSocketUrl = (path = '') => {
    const normalizedPath = path.startsWith('/') ? path : `/${path}`
    try {
        const apiBase = getApiBaseUrl()
        const parsed = new URL(apiBase)
        const protocol = parsed.protocol === 'https:' ? 'wss:' : 'ws:'
        return `${protocol}//${parsed.host}${normalizedPath}`
    } catch (error) {
        if (typeof window !== 'undefined') {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
            const port = window.location.port || '3000'
            return `${protocol}//${window.location.hostname}:${port}${normalizedPath}`
        }
        return `ws://localhost:8000${normalizedPath}`
    }
}
