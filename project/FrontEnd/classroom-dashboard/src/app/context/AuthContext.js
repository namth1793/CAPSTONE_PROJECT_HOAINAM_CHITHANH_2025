// frontend/app/context/AuthContext.js
'use client'
import { createContext, useContext, useEffect, useState } from 'react'
import { buildApiUrl } from '../config/api'

const AuthContext = createContext()

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null)
    const [loading, setLoading] = useState(true)
    const [authChecked, setAuthChecked] = useState(false) // Thêm state này

    // Hàm check auth với backend - cải thiện
    const checkAuthWithBackend = async () => {
        const token = localStorage.getItem('access_token')
        if (!token) {
            setUser(null)
            setAuthChecked(true)
            setLoading(false)
            return null
        }

        try {
            const response = await fetch(buildApiUrl(`/api/auth/check?token=${token}`), {
                cache: 'no-store'
            })
            const data = await response.json()

            if (data.authenticated && data.user) {
                setUser(data.user)
                // Lưu thêm vào localStorage để đồng bộ
                localStorage.setItem('user', JSON.stringify(data.user))
                localStorage.setItem('is_admin', data.user.is_admin)
            } else {
                // Token không hợp lệ
                localStorage.removeItem('access_token')
                localStorage.removeItem('user')
                localStorage.removeItem('is_admin')
                setUser(null)
            }
            return data.user
        } catch (error) {
            console.error('Error checking auth:', error)
            setUser(null)
            return null
        } finally {
            setAuthChecked(true)
            setLoading(false)
        }
    }

    useEffect(() => {
        checkAuthWithBackend()
    }, [])

    const login = async (userData, token) => {
        // Lưu vào localStorage trước
        localStorage.setItem('access_token', token)
        localStorage.setItem('user', JSON.stringify(userData))
        localStorage.setItem('is_admin', userData.is_admin)

        // Cập nhật state ngay lập tức
        setUser(userData)

        // Kiểm tra với backend để xác nhận
        setTimeout(() => {
            checkAuthWithBackend()
        }, 100)

        return userData
    }

    const logout = () => {
        localStorage.removeItem('access_token')
        localStorage.removeItem('user')
        localStorage.removeItem('is_admin')
        setUser(null)
        setAuthChecked(true)
        // Chuyển hướng đến trang login
        window.location.href = '/login'
    }

    // Thêm hàm kiểm tra quyền admin
    const isAdmin = () => {
        if (user) {
            return user.is_admin === true || user.is_admin === 1
        }
        return false
    }

    // Thêm hàm refresh auth
    const refreshAuth = async () => {
        setLoading(true)
        const user = await checkAuthWithBackend()
        setLoading(false)
        return user
    }

    const value = {
        user,
        login,
        logout,
        loading,
        authChecked, // Thêm này
        isAdmin,
        refreshAuth
    }

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    )
}

export function useAuth() {
    const context = useContext(AuthContext)
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider')
    }
    return context
}