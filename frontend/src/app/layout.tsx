import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "APTOS DR Classifier",
  description: "Research prototype for diabetic retinopathy image classification.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

