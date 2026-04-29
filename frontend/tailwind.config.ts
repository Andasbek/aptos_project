import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#17202a",
        clinical: "#0f766e",
        signal: "#b45309",
      },
      boxShadow: {
        soft: "0 16px 42px rgba(23, 32, 42, 0.08)",
      },
    },
  },
  plugins: [],
};

export default config;

