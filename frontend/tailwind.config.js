/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                background: '#0F172A',
                surface: '#1E293B',
                primary: '#22C55E',
                darker: '#020617',
                light: '#F8FAFC',
            },
            fontFamily: {
                sans: ['"Fira Sans"', 'sans-serif'],
                mono: ['"Fira Code"', 'monospace'],
            },
        },
    },
    plugins: [],
}
