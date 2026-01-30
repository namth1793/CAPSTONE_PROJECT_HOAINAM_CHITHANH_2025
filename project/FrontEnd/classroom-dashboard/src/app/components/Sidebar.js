/* eslint-disable @next/next/no-img-element */
// frontend/app/components/Sidebar.js
'use client'
import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
import { useEffect, useState } from 'react'
import { useAuth } from '../context/AuthContext'

export default function Sidebar() {
    const { user, logout } = useAuth()
    const pathname = usePathname()
    const router = useRouter()
    const [isWelcomePage, setIsWelcomePage] = useState(false)

    // Kiểm tra xem có phải trang Welcome không
    useEffect(() => {
        setIsWelcomePage(pathname === '/welcome')
    }, [pathname])

    // Ẩn sidebar khi đang ở trang login hoặc khi chưa đăng nhập hoặc ở trang Welcome
    if (pathname === '/login' || !user || isWelcomePage) {
        return null
    }

    const menuItems = [
        { href: '/', label: 'Dashboard', icon: '📊', type: 'link' },
        { href: '/live-class', label: 'Live Class', icon: '🎥', type: 'link' },
        { href: '/attendance', label: 'Attendance', icon: '👥', type: 'link' },
        { href: '/analytics', label: 'Analytics', icon: '📈', type: 'link' },
        { href: '/reports', label: 'Reports', icon: '📝', type: 'link' },
        { href: '/welcome', label: 'Welcome', icon: '👋', type: 'link' }, // CHANGED: Sử dụng href thay vì action
        { href: '/feedback', label: 'Feedback', icon: '📈', type: 'link' }, // CHANGED: Sử dụng href thay vì action
    ]

    const handleLogout = () => {
        // Xóa tất cả localStorage liên quan trước khi logout
        localStorage.removeItem('showWelcome')
        localStorage.removeItem('detectedStudent')
        logout()
    }

    return (
        <div className="w-64 sidebar min-h-screen flex flex-col">
            {/* Header Sidebar */}
            <div className="p-6 border-b border-gray-800">
                <div className="flex justify-center">
                    <img
                        src="/logo_company.png"
                        alt="LYDINC Logo"
                        className="h-24 w-auto"
                    />
                </div>
            </div>



            {/* User Info */}
            <div className="p-4 border-b border-gray-800">
                <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full flex items-center justify-center">
                        <span className="text-xl font-bold">
                            {user.full_name?.charAt(0) || user.username?.charAt(0) || 'U'}
                        </span>
                    </div>
                    <div className="flex-1 min-w-0 font-bold">
                        <p className="text-lg truncate">
                            Administrator
                        </p>
                    </div>
                </div>
            </div>

            {/* Navigation Menu */}
            <nav className="p-4 text-lg mt-10 mb-10">
                <ul className="space-y-2">
                    {menuItems.map((item) => (
                        <li key={item.label}>
                            <Link
                                href={item.href}
                                className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors duration-200 ${pathname === item.href
                                    ? 'bg-[#B39858] text-white'
                                    : 'hover:bg-[#B39858] hover:text-white'
                                    }`}
                            >
                                <span className="text-lg">{item.icon}</span>
                                <span className="font-medium">{item.label}</span>
                            </Link>
                        </li>
                    ))}
                </ul>
            </nav>

            {/* Footer với Logout */}
            <div className="p-4 border-t border-gray-800">
                <div className="space-y-3">

                    {/* Logout Button */}
                    <button
                        onClick={handleLogout}
                        className="px-4 py-2 ml-10 bg-red-600 text-white rounded-lg hover:bg-red-700 font-medium transition"
                    >
                        🚪 Đăng xuất
                    </button>
                </div>
            </div>
        </div>
    )
}