// src/app/reports/page.js
'use client'
import { useEffect, useState } from 'react'
import { buildApiUrl } from '../config/api'

export default function ReportsPage() {
    const [reports, setReports] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [generating, setGenerating] = useState(false)

    useEffect(() => {
        fetchReports()
    }, [])

    const fetchReports = async () => {
        try {
            setError(null)
            // Mock reports data
            const mockReports = [
                {
                    id: 1,
                    name: 'Daily Attendance Report',
                    date: new Date().toISOString().split('T')[0],
                    type: 'PDF',
                    size: '245 KB',
                    icon: '📊'
                },
                {
                    id: 2,
                    name: 'Weekly Performance Summary',
                    date: '2024-01-10',
                    type: 'Excel',
                    size: '189 KB',
                    icon: '📈'
                },
                {
                    id: 3,
                    name: 'Emotion Analytics Report',
                    date: '2024-01-05',
                    type: 'PDF',
                    size: '312 KB',
                    icon: '😊'
                },
                {
                    id: 4,
                    name: 'Monthly Engagement Overview',
                    date: '2024-01-01',
                    type: 'PDF',
                    size: '428 KB',
                    icon: '🎯'
                },
            ]
            setReports(mockReports)
        } catch (error) {
            console.error('Error fetching reports:', error)
            setError('Failed to load reports')
        } finally {
            setLoading(false)
        }
    }

    const generateReport = async () => {
        try {
            setGenerating(true)
            const response = await fetch(buildApiUrl('/api/reports/export'))

            if (response.ok) {
                // Download logic here
                const blob = await response.blob()
                const url = window.URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.style.display = 'none'
                a.href = url

                // Lấy tên file từ header Content-Disposition hoặc tạo mới
                const contentDisposition = response.headers.get('content-disposition')
                let filename = `attendance_report_${new Date().toISOString().split('T')[0]}.xlsx` // Đổi thành .xlsx

                if (contentDisposition) {
                    const filenameMatch = contentDisposition.match(/filename="(.+)"/)
                    if (filenameMatch && filenameMatch[1]) {
                        filename = filenameMatch[1]
                    }
                }

                a.download = filename
                document.body.appendChild(a)
                a.click()
                window.URL.revokeObjectURL(url)

                // Add new report to list
                const newReport = {
                    id: reports.length + 1,
                    name: `Attendance Export - ${new Date().toLocaleDateString()}`,
                    date: new Date().toISOString().split('T')[0],
                    type: 'Excel', // Đổi từ 'CSV' thành 'Excel'
                    size: '~200 KB',
                    icon: '📊' // Có thể đổi icon cho Excel
                }
                setReports(prev => [newReport, ...prev])

                alert('Report generated and downloaded successfully!')
            } else {
                throw new Error('Failed to generate report')
            }
        } catch (error) {
            console.error('Error generating report:', error)
            alert('Failed to generate report. Please try again.')
        } finally {
            setGenerating(false)
        }
    }

    const deleteReport = (reportId) => {
        if (window.confirm('Are you sure you want to delete this report?')) {
            setReports(prev => prev.filter(report => report.id !== reportId))
        }
    }

    const getTypeColor = (type) => {
        const colors = {
            PDF: { bg: 'bg-red-100', text: 'text-red-800', border: 'border-red-200' },
            Excel: { bg: 'bg-green-100', text: 'text-green-800', border: 'border-green-200' },
            CSV: { bg: 'bg-blue-100', text: 'text-blue-800', border: 'border-blue-200' }
        }
        return colors[type] || colors.PDF
    }

    if (loading) {
        return (
            <div className="reports-page flex justify-center items-center min-h-screen">
                <div className="loading-spinner"></div>
                <span className="ml-3 text-lg text-white">Loading Reports...</span>
            </div>
        )
    }

    if (error) {
        return (
            <div className="reports-page flex justify-center items-center min-h-screen">
                <div className="bg-white rounded-2xl shadow-xl p-8 max-w-md w-full mx-4">
                    <div className="text-red-500 text-center mb-4">
                        <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                        </svg>
                    </div>
                    <h3 className="text-xl font-bold text-gray-800 text-center mb-2">Connection Error</h3>
                    <p className="text-gray-600 text-center mb-6">{error}</p>
                    <button
                        onClick={fetchReports}
                        className="w-full bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700 transition duration-200"
                    >
                        Try Again
                    </button>
                </div>
            </div>
        )
    }

    return (
        <div className="reports-page bg-[#B39858] p-6 min-h-screen">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8">
                    <h1 className="text-4xl font-bold text-white mb-3">Reports & Analytics</h1>
                    <p className="text-blue-100 text-lg">Generate and manage comprehensive classroom reports</p>
                </div>

                {/* Action Card */}
                <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
                    <div className="flex flex-col lg:flex-row items-center justify-between">
                        <div className="text-center lg:text-left mb-6 lg:mb-0">
                            <h2 className="text-2xl font-bold text-gray-800 mb-2">Generate Daily Report</h2>
                            <p className="text-gray-600">Create comprehensive reports for attendance, engagement, and analytics</p>
                        </div>
                        <button
                            onClick={generateReport}
                            disabled={generating}
                            className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-4 rounded-xl font-semibold hover:from-blue-700 hover:to-purple-700 transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
                        >
                            {generating ? (
                                <>
                                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Generating...
                                </>
                            ) : (
                                <>
                                    <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                                    </svg>
                                    Generate Report
                                </>
                            )}
                        </button>
                    </div>
                </div>

                {/* Reports List */}
                <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
                    <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-6">
                        <h2 className="text-2xl font-bold text-white flex items-center">
                            <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            Available Reports
                        </h2>
                    </div>

                    <div className="p-6">
                        {reports.length === 0 ? (
                            <div className="text-center py-12">
                                <svg className="w-24 h-24 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                                <h3 className="text-xl font-semibold text-gray-800 mb-2">No Reports Available</h3>
                                <p className="text-gray-600 mb-6">Generate your first report to get started</p>
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                {reports.map((report) => {
                                    const typeColor = getTypeColor(report.type)
                                    return (
                                        <div key={report.id} className="border border-gray-200 rounded-xl p-6 hover:shadow-lg transition duration-200">
                                            <div className="flex items-start justify-between mb-4">
                                                <div className="flex items-center">
                                                    <span className="text-2xl mr-3">{report.icon}</span>
                                                    <div>
                                                        <h3 className="font-semibold text-gray-800 text-lg">{report.name}</h3>
                                                        <p className="text-gray-600 text-sm">Created on {report.date}</p>
                                                    </div>
                                                </div>
                                                <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${typeColor.bg} ${typeColor.text} ${typeColor.border}`}>
                                                    {report.type}
                                                </span>
                                            </div>

                                            <div className="flex items-center justify-between">
                                                <span className="text-gray-600 text-sm">{report.size}</span>
                                                <div className="flex space-x-2">
                                                    <button className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition duration-200 flex items-center text-sm">
                                                        <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                                                        </svg>
                                                        Download
                                                    </button>
                                                    <button
                                                        onClick={() => deleteReport(report.id)}
                                                        className="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition duration-200 flex items-center text-sm"
                                                    >
                                                        <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                                        </svg>
                                                        Delete
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    )
                                })}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}