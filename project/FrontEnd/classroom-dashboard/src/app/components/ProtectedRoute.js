// frontend/app/components/ProtectedRoute.js
'use client'
import { useRouter } from 'next/navigation'
import { useEffect } from 'react'
import { useAuth } from '../context/AuthContext'

export default function ProtectedRoute({
    children,
    requireAdmin = false,  // Thêm prop để yêu cầu quyền admin
    redirectTo = '/login'  // Thêm prop để custom redirect
}) {
    const { user, loading, isAdmin } = useAuth()
    const router = useRouter()

    useEffect(() => {
        if (!loading) {
            // Nếu không có user, redirect đến login
            if (!user) {
                router.push(redirectTo)
                return
            }

            // Nếu route yêu cầu admin nhưng user không phải admin
            if (requireAdmin && !isAdmin()) {
                // Redirect về trang user dashboard
                router.push('/user-dashboard')
                return
            }
        }
    }, [user, loading, router, requireAdmin, redirectTo])

    if (loading) {
        return (
            <div className="min-h-screen bg-gray-900 flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
                    <p className="text-white mt-4">Đang tải...</p>
                </div>
            </div>
        )
    }

    // Kiểm tra quyền nếu requireAdmin
    if (requireAdmin && user && !isAdmin()) {
        return (
            <div className="min-h-screen bg-gray-900 flex items-center justify-center">
                <div className="text-center text-white">
                    <h2 className="text-2xl font-bold mb-4">Không có quyền truy cập</h2>
                    <p className="mb-4">Bạn không có quyền truy cập trang này.</p>
                    <button
                        onClick={() => router.push('/user-dashboard')}
                        className="px-4 py-2 bg-blue-500 rounded hover:bg-blue-600"
                    >
                        Về trang người dùng
                    </button>
                </div>
            </div>
        )
    }

    if (!user) {
        return null
    }

    return children
}