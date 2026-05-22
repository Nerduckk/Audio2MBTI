/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        mistral: {
          bg: '#fffaeb',       // Nền Warm Ivory
          surface: '#fff0c2',  // Nền thẻ Cream
          black: '#1f1f1f',    // Chữ Mistral Black
          yellow: '#ffd900',   // Bright Yellow
          amber: '#ffa110',    // Sunshine Amber
          orange: '#fa520f',   // Mistral Orange
        }
      },
      boxShadow: {
        'golden': '-8px 16px 39px rgba(127, 99, 21, 0.12)', 
      },
      fontFamily: {
        sans: ['Arial', 'Helvetica', 'sans-serif'], 
      }
    },
  },
  plugins: [],
}