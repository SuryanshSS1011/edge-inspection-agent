import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Pages serves the project site under /<repo>/, so assets must resolve there.
// Locally (dev / preview) base "/" is fine.
export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/edge-inspection-agent/' : '/',
  plugins: [react()],
}))
