// src/app/attendance/page.js
'use client'
import { useEffect, useState } from 'react'
import { buildApiUrl } from '../config/api'

export default function AttendancePage() {
    const [attendanceData, setAttendanceData] = useState([])
    const [presentStudents, setPresentStudents] = useState([])
    const [absentStudents, setAbsentStudents] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [summary, setSummary] = useState({
        total_students: 0,
        present_count: 0,
        absent_count: 0,
        attendance_rate: 0
    })

    useEffect(() => {
        fetchAttendanceData()
    }, [])

    // Hàm để loại bỏ học sinh trùng lặp dựa trên tên
    const removeDuplicateStudentsByName = (students) => {
        if (!Array.isArray(students)) return [];

        const seen = new Set();
        const uniqueStudents = [];

        for (const student of students) {
            // Chuẩn hóa tên: lowercase và trim
            const studentName = student.student || student.student_name || 'Unknown Student';
            const normalizedName = studentName.trim().toLowerCase();

            if (!normalizedName) continue;

            // Chỉ thêm vào mảng nếu chưa thấy tên này
            if (!seen.has(normalizedName)) {
                seen.add(normalizedName);
                uniqueStudents.push({
                    ...student,
                    normalizedName: normalizedName,
                    displayName: studentName
                });
            }
        }

        return uniqueStudents;
    }

    // Hàm để lọc bỏ học sinh đã có mặt khỏi danh sách vắng (dựa trên tên)
    const filterAbsentWithoutPresent = (absentList, presentList) => {
        if (!Array.isArray(absentList) || !Array.isArray(presentList)) return absentList;

        // Tạo set các tên học sinh đã có mặt
        const presentNames = new Set();
        presentList.forEach(student => {
            const studentName = student.student || student.student_name;
            if (studentName) {
                presentNames.add(studentName.trim().toLowerCase());
            }
        });

        // Lọc danh sách vắng, chỉ giữ những học sinh không có tên trong danh sách có mặt
        return absentList.filter(student => {
            const studentName = student.student || student.student_name;
            if (!studentName) return true;

            const normalizedName = studentName.trim().toLowerCase();
            return !presentNames.has(normalizedName);
        });
    }

    // Hàm so sánh tên gần đúng (để xử lý các trường hợp viết hoa/viết thường, khoảng trắng)
    const normalizeName = (name) => {
        if (!name) return '';
        return name
            .trim()
            .toLowerCase()
            .replace(/\s+/g, ' ') // Chuẩn hóa khoảng trắng
            .normalize('NFD') // Chuẩn hóa Unicode
            .replace(/[\u0300-\u036f]/g, ''); // Xóa dấu
    }

    const fetchAttendanceData = async () => {
        try {
            setError(null)
            setLoading(true)

            console.log('🔄 Fetching attendance summary...')

            // Sử dụng endpoint summary để lấy dữ liệu tổng hợp
            const response = await fetch(buildApiUrl('/api/attendance/summary'))

            console.log('📡 Response status:', response.status)

            if (!response.ok) {
                throw new Error(`API Error: ${response.status} ${response.statusText}`)
            }

            const data = await response.json()
            console.log('📊 Summary API data:', data)

            // Xử lý dữ liệu từ API summary
            if (data.summary) {
                let presentList = [];
                let absentList = [];

                // Xử lý học sinh có mặt
                if (data.present_students && Array.isArray(data.present_students)) {
                    presentList = data.present_students.map(student => {
                        const studentName = student.student_name || student.student || 'Unknown Student';
                        return {
                            student: studentName,
                            time_in: student.check_in_time ? formatTime(student.check_in_time) : '07:30 AM',
                            status: 'present',
                            student_id: student.student_id,
                            class_name: student.class_name,
                            timestamp: student.timestamp,
                            normalizedName: normalizeName(studentName)
                        };
                    });
                }

                // Xử lý học sinh vắng
                if (data.absent_students && Array.isArray(data.absent_students)) {
                    absentList = data.absent_students.map(student => {
                        const studentName = student.student_name || student.student || 'Unknown Student';
                        return {
                            student: studentName,
                            time_in: '-',
                            status: 'absent',
                            student_id: student.student_id,
                            class_name: student.class_name,
                            normalizedName: normalizeName(studentName)
                        };
                    });
                }

                // Loại bỏ trùng lặp trong mỗi danh sách (dựa trên tên)
                presentList = removeDuplicateStudentsByName(presentList);
                absentList = removeDuplicateStudentsByName(absentList);

                console.log('📊 Before filtering - Present:', presentList.length, 'Absent:', absentList.length);

                // Lọc bỏ học sinh đã có mặt khỏi danh sách vắng (dựa trên tên)
                absentList = filterAbsentWithoutPresent(absentList, presentList);

                console.log('📊 After filtering - Present:', presentList.length, 'Absent:', absentList.length);

                // Sắp xếp present theo thời gian sớm nhất
                presentList.sort((a, b) => {
                    if (!a.timestamp || !b.timestamp) return 0;
                    return new Date(a.timestamp) - new Date(b.timestamp);
                });

                // Cập nhật state
                setPresentStudents(presentList);
                setAbsentStudents(absentList);
                setAttendanceData([...presentList, ...absentStudents]);

                // Tính toán thống kê từ dữ liệu thực tế
                const presentCount = presentList.length;
                const absentCount = absentList.length;
                const totalStudents = presentCount + absentCount;
                const attendanceRate = totalStudents > 0 ? Math.round((presentCount / totalStudents) * 100) : 0;

                setSummary({
                    total_students: totalStudents,
                    present_count: presentCount,
                    absent_count: absentCount,
                    attendance_rate: attendanceRate
                });

                // Log để debug
                console.log('✅ Present students (unique by name):', presentList.map(s => s.student));
                console.log('❌ Absent students (filtered by name):', absentList.map(s => s.student));
            } else {
                throw new Error('Invalid data structure from API');
            }

        } catch (error) {
            console.error('❌ Error fetching attendance summary:', error)
            setError(error.message || 'Failed to load attendance data')
            useFallbackData()
        } finally {
            setLoading(false)
        }
    }

    const formatTime = (timestamp) => {
        try {
            if (!timestamp) return '07:30 AM';

            const date = new Date(timestamp);
            if (isNaN(date.getTime())) return '07:30 AM';

            return date.toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                hour12: true
            });
        } catch (error) {
            return '07:30 AM';
        }
    }

    const useFallbackData = () => {
        // Fallback data khi API không hoạt động
        const fallbackPresent = [
            { student: 'Nguyễn Văn A', time_in: '07:30 AM', status: 'present', student_id: 'ST001' },
            { student: 'Trần Thị B', time_in: '07:35 AM', status: 'present', student_id: 'ST002' },
            { student: 'Lê Văn C', time_in: '08:00 AM', status: 'present', student_id: 'ST003' },
            { student: 'Hoàng Văn E', time_in: '07:40 AM', status: 'present', student_id: 'ST005' },
            { student: 'Nguyễn Văn G', time_in: '07:55 AM', status: 'present', student_id: 'ST007' },
        ]

        const fallbackAbsent = [
            { student: 'Phạm Thị D', time_in: '-', status: 'absent', student_id: 'ST004' },
            { student: 'Đỗ Thị F', time_in: '-', status: 'absent', student_id: 'ST006' },
            { student: 'Trần Thị H', time_in: '-', status: 'absent', student_id: 'ST008' },
            // Thêm học sinh trùng tên với present để test
            { student: 'Nguyễn Văn A', time_in: '-', status: 'absent', student_id: 'ST009' },
            { student: 'trần thị b', time_in: '-', status: 'absent', student_id: 'ST010' }, // lowercase để test
            { student: 'Lê  Văn  C', time_in: '-', status: 'absent', student_id: 'ST011' }, // thêm khoảng trắng để test
        ]

        // Loại bỏ trùng lặp trong mỗi danh sách (dựa trên tên)
        const uniquePresent = removeDuplicateStudentsByName(fallbackPresent);
        let uniqueAbsent = removeDuplicateStudentsByName(fallbackAbsent);

        // Lọc bỏ học sinh đã có mặt khỏi danh sách vắng (dựa trên tên)
        uniqueAbsent = filterAbsentWithoutPresent(uniqueAbsent, uniquePresent);

        setPresentStudents(uniquePresent)
        setAbsentStudents(uniqueAbsent)

        const totalStudents = uniquePresent.length + uniqueAbsent.length;
        const attendanceRate = totalStudents > 0 ? Math.round((uniquePresent.length / totalStudents) * 100) : 0;

        setSummary({
            total_students: totalStudents,
            present_count: uniquePresent.length,
            absent_count: uniqueAbsent.length,
            attendance_rate: attendanceRate
        })

        console.log('🔄 Using fallback data with name-based filtering');
        console.log('✅ Present:', uniquePresent.map(s => s.student));
        console.log('❌ Absent (after filtering):', uniqueAbsent.map(s => s.student));
    }

    // Safe calculation of statistics
    const presentCount = presentStudents.length
    const absentCount = absentStudents.length
    const totalStudents = presentCount + absentCount
    const presentPercentage = totalStudents > 0 ? Math.round((presentCount / totalStudents) * 100) : 0

    // Safe initials generation
    const getInitials = (studentName) => {
        if (!studentName || typeof studentName !== 'string') {
            return '??'
        }

        const parts = studentName.trim().split(' ')
        if (parts.length === 0) return '??'

        if (parts.length === 1) {
            return parts[0].charAt(0).toUpperCase()
        }

        return (parts[0].charAt(0) + parts[parts.length - 1].charAt(0)).toUpperCase()
    }

    // Format date for display
    const getFormattedDate = () => {
        return new Date().toLocaleDateString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        })
    }

    if (loading) {
        return (
            <div className="attendance-page flex justify-center items-center min-h-screen bg-black">
                <div className="text-center">
                    <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-4"></div>
                    <span className="text-lg text-white block">Loading Attendance...</span>
                    <p className="text-gray-400 mt-2">Fetching data from server...</p>
                </div>
            </div>
        )
    }

    if (error && presentStudents.length === 0 && absentStudents.length === 0) {
        return (
            <div className="attendance-page flex justify-center items-center min-h-screen bg-black">
                <div className="bg-gray-900 rounded-2xl shadow-xl p-8 max-w-md w-full mx-4 border border-gray-800">
                    <div className="text-red-500 text-center mb-4">
                        <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                        </svg>
                    </div>
                    <h3 className="text-xl font-bold text-white text-center mb-2">Connection Error</h3>
                    <p className="text-gray-400 text-center mb-6">{error}</p>
                    <div className="space-y-3">
                        <button
                            onClick={fetchAttendanceData}
                            className="w-full bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700 transition duration-200"
                        >
                            Try Again
                        </button>
                        <button
                            onClick={useFallbackData}
                            className="w-full bg-gray-700 text-white py-3 rounded-xl font-semibold hover:bg-gray-600 transition duration-200"
                        >
                            Use Demo Data
                        </button>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="attendance-page bg-[#B39858] p-6 min-h-screen">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8">
                    <h1 className="text-4xl font-bold text-white mb-3">Attendance Management</h1>
                    <p className="text-blue-100 text-lg">Track and manage student attendance in real-time</p>
                </div>

                {/* Two Tables Section - Present and Absent */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Present Students Table */}
                    <div className="bg-gray-900 rounded-2xl shadow-xl overflow-hidden">
                        <div className="bg-gradient-to-r from-green-600 to-emerald-600 p-6">
                            <div className="flex justify-between items-center">
                                <h2 className="text-2xl font-bold text-white flex items-center">
                                    <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                    Present Students
                                </h2>
                                <div className="flex items-center space-x-2">
                                    <span className="bg-green-800 text-green-100 px-3 py-1 rounded-full text-sm font-semibold">
                                        {presentCount} students
                                    </span>
                                </div>
                            </div>
                        </div>

                        <div className="p-6">
                            {presentStudents.length === 0 ? (
                                <div className="text-center py-12">
                                    <div className="w-16 h-16 bg-green-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                                        <svg className="w-8 h-8 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                    </div>
                                    <h3 className="text-xl font-semibold text-gray-300 mb-2">No Students Present</h3>
                                    <p className="text-gray-500">No attendance records found for today.</p>
                                </div>
                            ) : (
                                <div className="overflow-hidden rounded-xl border border-gray-700">
                                    <table className="w-full">
                                        <thead className="bg-gray-800">
                                            <tr>
                                                <th className="text-left p-4 text-gray-300 font-semibold">Student</th>
                                                <th className="text-left p-4 text-gray-300 font-semibold">Check-in Time</th>
                                                <th className="text-center p-4 text-gray-300 font-semibold">Status</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-gray-800">
                                            {presentStudents.map((record, index) => (
                                                <tr key={`present-${index}-${record.normalizedName}`} className="hover:bg-gray-800/50 transition duration-150">
                                                    <td className="p-4">
                                                        <div className="flex items-center">
                                                            <div className="w-10 h-10 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full flex items-center justify-center text-white font-semibold text-sm mr-3">
                                                                {getInitials(record.student)}
                                                            </div>
                                                            <div>
                                                                <span className="text-gray-200 font-medium block">{record.student || 'Unknown Student'}</span>
                                                                {record.student_id && (
                                                                    <span className="text-gray-500 text-xs">ID: {record.student_id}</span>
                                                                )}
                                                            </div>
                                                        </div>
                                                    </td>
                                                    <td className="p-4">
                                                        <span className="text-lg font-semibold text-gray-300">
                                                            {record.time_in || '-'}
                                                        </span>
                                                    </td>
                                                    <td className="p-4 text-center">
                                                        <span className="inline-flex items-center px-4 py-2 rounded-full text-sm font-semibold bg-green-900/30 text-green-400 border border-green-800/50">
                                                            <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                                                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                                            </svg>
                                                            Present
                                                        </span>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Absent Students Table */}
                    <div className="bg-gray-900 rounded-2xl shadow-xl overflow-hidden">
                        <div className="bg-gradient-to-r from-red-600 to-orange-600 p-6">
                            <div className="flex justify-between items-center">
                                <h2 className="text-2xl font-bold text-white flex items-center">
                                    <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                    Absent Students
                                </h2>
                                <span className="bg-red-800 text-red-100 px-3 py-1 rounded-full text-sm font-semibold">
                                    {absentCount} students
                                </span>
                            </div>
                        </div>

                        <div className="p-6">
                            {absentStudents.length === 0 ? (
                                <div className="text-center py-12">
                                    <div className="w-16 h-16 bg-green-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                                        <svg className="w-8 h-8 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                    </div>
                                    <h3 className="text-xl font-semibold text-gray-300 mb-2">Perfect Attendance!</h3>
                                    <p className="text-gray-500">All students are present today.</p>
                                </div>
                            ) : (
                                <div className="overflow-hidden rounded-xl border border-gray-700">
                                    <table className="w-full">
                                        <thead className="bg-gray-800">
                                            <tr>
                                                <th className="text-left p-4 text-gray-300 font-semibold">Student</th>
                                                <th className="text-left p-4 text-gray-300 font-semibold">Class</th>
                                                <th className="text-center p-4 text-gray-300 font-semibold">Status</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-gray-800">
                                            {absentStudents.map((record, index) => (
                                                <tr key={`absent-${index}-${record.normalizedName}`} className="hover:bg-gray-800/50 transition duration-150">
                                                    <td className="p-4">
                                                        <div className="flex items-center">
                                                            <div className="w-10 h-10 bg-gradient-to-r from-red-500 to-orange-500 rounded-full flex items-center justify-center text-white font-semibold text-sm mr-3">
                                                                {getInitials(record.student)}
                                                            </div>
                                                            <div>
                                                                <span className="text-gray-200 font-medium block">{record.student || 'Unknown Student'}</span>
                                                                {record.student_id && (
                                                                    <span className="text-gray-500 text-xs">ID: {record.student_id}</span>
                                                                )}
                                                            </div>
                                                        </div>
                                                    </td>
                                                    <td className="p-4">
                                                        <span className="text-gray-400">
                                                            {record.class_name || 'N/A'}
                                                        </span>
                                                    </td>
                                                    <td className="p-4 text-center">
                                                        <span className="inline-flex items-center px-4 py-2 rounded-full text-sm font-semibold bg-red-900/30 text-red-400 border border-red-800/50">
                                                            <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                                            </svg>
                                                            Absent
                                                        </span>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}