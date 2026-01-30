/* eslint-disable @next/next/no-img-element */
// frontend/app/login/page.js
'use client'
import { useRouter } from 'next/navigation'
import { useEffect, useState } from 'react'
import { useAuth } from '../context/AuthContext'
import { buildApiUrl } from '../config/api'

export default function Login() {
    const [isLogin, setIsLogin] = useState(true)
    const [formData, setFormData] = useState({
        username: '',
        email: '',
        password: '',
        full_name: ''
    })
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const { user, login, loading: authLoading, authChecked } = useAuth()
    const router = useRouter()

    useEffect(() => {
        // Chỉ redirect khi đã check auth xong VÀ có user
        if (authChecked && user && !authLoading) {
            console.log('Redirecting user:', user)
            if (user.is_admin) {
                // Đảm bảo reload để sidebar hiển thị
                window.location.href = '/'
            } else {
                window.location.href = '/user-dashboard'
            }
        }
    }, [user, authLoading, authChecked, router])

    // Hiển thị loading khi đang kiểm tra authentication
    if (authLoading && !authChecked) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-blue-600 to-purple-700 flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto"></div>
                    <p className="text-white mt-4">Đang kiểm tra đăng nhập...</p>
                </div>
            </div>
        )
    }

    // Nếu đã đăng nhập và đã check auth, không hiển thị trang login
    if (authChecked && user) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-blue-600 to-purple-700 flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto"></div>
                    <p className="text-white mt-4">Đang chuyển hướng...</p>
                </div>
            </div>
        )
    }

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        })
    }

    const handleSubmit = async (e) => {
        e.preventDefault()
        setLoading(true)
        setError('')

        try {
            const endpoint = isLogin ? '/api/auth/login' : '/api/auth/register'
            const response = await fetch(buildApiUrl(endpoint), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData),
                cache: 'no-store'
            })

            const data = await response.json()

            if (!response.ok) {
                throw new Error(data.detail || 'Có lỗi xảy ra')
            }

            if (isLogin) {
                // Đăng nhập thành công
                const userData = data.user || {
                    id: data.user_id,
                    username: data.username,
                    email: data.email,
                    full_name: data.full_name || data.username,
                    is_admin: data.user?.is_admin || false
                }

                // Gọi login function từ context
                login(userData, data.access_token)

                // Không redirect ở đây nữa, để useEffect xử lý
            } else {
                // Đăng ký thành công, chuyển sang đăng nhập
                setIsLogin(true)
                setError('Đăng ký thành công! Vui lòng đăng nhập.')
                setFormData({
                    username: '',
                    email: '',
                    password: '',
                    full_name: ''
                })
            }
        } catch (error) {
            setError(error.message)
        } finally {
            setLoading(false)
        }
    }

    const handleDemoLogin = async () => {
        setLoading(true)
        setError('')

        try {
            const response = await fetch(buildApiUrl('/api/auth/demo-login'), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                cache: 'no-store'
            })

            const data = await response.json()

            if (!response.ok) {
                throw new Error(data.detail || 'Có lỗi xảy ra')
            }

            // Demo login thành công
            const userData = data.user || {
                id: data.user_id,
                username: data.username,
                email: data.email,
                full_name: data.full_name || 'Demo User',
                is_admin: data.user?.is_admin || false
            }

            login(userData, data.access_token)

            // Không redirect ở đây nữa, để useEffect xử lý
        } catch (error) {
            setError(error.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-gray-700 flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-md">
                <div className="pb-3">
                    <div className="flex justify-center">
                        <img
                            src="/logo_company.png"
                            alt="LYDINC Logo"
                            className="h-26 w-auto"
                        />
                    </div>
                </div>
                <div className="text-center mb-8">
                    <h1 className="font-heading text-2xl font-bold text-gray-800 mb-2">
                        Classroom Management
                    </h1>
                </div>

                {error && (
                    <div className={`mb-6 p-4 rounded-lg ${error.includes('thành công')
                        ? 'bg-green-100 text-green-800 border border-green-200'
                        : 'bg-red-100 text-red-800 border border-red-200'
                        }`}>
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-6">
                    {!isLogin && (
                        <>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Họ và tên
                                </label>
                                <input
                                    type="text"
                                    name="full_name"
                                    value={formData.full_name}
                                    onChange={handleChange}
                                    required={!isLogin}
                                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition duration-200"
                                    placeholder="Nhập họ và tên"
                                />
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Email
                                </label>
                                <input
                                    type="email"
                                    name="email"
                                    value={formData.email}
                                    onChange={handleChange}
                                    required={!isLogin}
                                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition duration-200"
                                    placeholder="Nhập email"
                                />
                            </div>
                        </>
                    )}

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Tên đăng nhập
                        </label>
                        <input
                            type="text"
                            name="username"
                            value={formData.username}
                            onChange={handleChange}
                            required
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition duration-200 text-black"
                            placeholder="Nhập tên đăng nhập"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Mật khẩu
                        </label>
                        <input
                            type="password"
                            name="password"
                            value={formData.password}
                            onChange={handleChange}
                            required
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition duration-200 text-black"
                            placeholder="Nhập mật khẩu"
                        />
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-[#B39858] hover:cursor-pointer text-white py-3 px-4 rounded-lg font-semibold transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {loading ? 'Đang xử lý...' : 'Đăng Nhập Admin'}
                    </button>
                </form>

                {isLogin && (
                    <>
                        <div className="mt-6">
                            <button
                                onClick={handleDemoLogin}
                                disabled={loading}
                                className="w-full bg-black hover:cursor-pointer text-white py-3 px-4 rounded-lg font-semibold transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {loading ? 'Đang xử lý...' : 'Đăng Nhập Khách'}
                            </button>
                        </div>
                    </>
                )}
            </div>
        </div>
    )
}