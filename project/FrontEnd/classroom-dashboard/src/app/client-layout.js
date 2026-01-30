// app/client-layout.js
'use client'

import { useEffect, useState } from 'react'
import Sidebar from './components/Sidebar'
import { AuthProvider, useAuth } from './context/AuthContext'

function LayoutContent({ children }) {
    const { user, loading, authChecked } = useAuth()
    const [pathname, setPathname] = useState('')
    const [hasToken, setHasToken] = useState(false)

    useEffect(() => {
        // Chỉ chạy trên client side
        if (typeof window !== 'undefined') {
            setPathname(window.location.pathname)

            // Lấy token từ localStorage
            const token = localStorage.getItem('access_token')
            setHasToken(!!token)
        }
    }, [])

    console.log('🔍 Layout Debug:', {
        user,
        loading,
        authChecked,
        pathname,
        hasToken
    })

    // Kiểm tra các trang đặc biệt
    const isLoginPage = pathname === '/login'

    if (loading && !authChecked) {
        return (
            <div className="min-h-screen bg-gray-900 flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
                    <p className="mt-4 text-gray-400">Đang tải...</p>
                </div>
            </div>
        )
    }

    // LOGIC HIỂN THỊ SIDEBAR ĐƠN GIẢN:
    // 1. Đã đăng nhập (user có tồn tại)
    // 2. Không phải trang login
    // 3. User không phải là admin thì ẩn sidebar trên user-dashboard
    const shouldShowSidebar = user &&
        !isLoginPage &&
        !(pathname === '/user-dashboard' && user.is_admin === false)

    console.log('📌 Should show sidebar?', shouldShowSidebar)

    return (
        <div className="min-h-screen">
            {shouldShowSidebar ? (
                <div className="flex">
                    <Sidebar />
                    <main className="flex-1 transition-all duration-300">
                        {children}
                    </main>
                </div>
            ) : (
                <main>{children}</main>
            )}
        </div>
    )
}

export function ClientLayout({ children }) {
    return (
        <AuthProvider>
            <LayoutContent>{children}</LayoutContent>
        </AuthProvider>
    )
}