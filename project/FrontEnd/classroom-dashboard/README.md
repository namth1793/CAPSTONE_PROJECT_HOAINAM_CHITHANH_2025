This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://github.com/vercel/next.js/tree/canary/packages/create-next-app).

## Getting Started

First, run the development server:

```bash

cd classroom-dashboard
npm install
npm run dev

```

## Frontend Folder Structure

The frontend is built using **Next.js (App Router)** and follows a feature-based
directory organization to support modular UI development and scalable classroom
analytics dashboards.

```text
FrontEnd/classroom-dashboard/
├── public/                    # Static assets (icons, images, public files)
├── src/
│   └── app/                   # Next.js App Router root
│       ├── analytics/         # Classroom analytics & engagement visualization
│       ├── attendance/        # Attendance management UI
│       ├── components/        # Reusable UI components
│       ├── config/            # Frontend configuration (API endpoints, constants)
│       ├── context/           # Global state management (React Context)
│       ├── feedback/          # Student feedback (text & voice) interfaces
│       ├── live-class/        # Real-time classroom monitoring views
│       ├── login/             # Authentication & login pages
│       ├── reports/           # Exportable reports and summaries
│       ├── user-dashboard/    # User-specific dashboard views
│       ├── welcome/           # Landing and introduction pages
│       ├── client-layout.js   # Client-side layout wrapper
│       ├── favicon.ico        # Application favicon
│       ├── globals.css        # Global CSS styles
│       ├── layout.js          # Root layout component
│       └── page.js            # Main entry page
├── .gitignore                 # Git ignore rules
├── eslint.config.mjs          # ESLint configuration
├── jsconfig.json              # JavaScript path alias configuration
├── next.config.mjs            # Next.js configuration
├── package.json               # Project dependencies and scripts
├── package-lock.json          # Dependency lock file
├── postcss.config.mjs         # PostCSS configuration
└── README.md                  # Frontend documentation
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.js`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
