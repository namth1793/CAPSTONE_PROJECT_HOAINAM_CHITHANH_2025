/* eslint-disable @next/next/no-page-custom-font */
// app/layout.js
import { ClientLayout } from './client-layout';
import './globals.css';

export const metadata = {
  title: 'Classroom Analytics',
  description: 'AI-powered classroom monitoring system',
}

export default function RootLayout({ children }) {
  return (
    <html lang="vi">
      <head>
        <link
          rel="preconnect"
          href="https://fonts.googleapis.com"
        />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-gray-900 text-gray-100">
        <ClientLayout>{children}</ClientLayout>
        {/* Tạm thời thêm DebugAuth để kiểm tra */}
      </body>
    </html >
  )
}