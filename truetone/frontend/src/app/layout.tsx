import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "TrueTone | Voice-Clone Detection",
  description: "AI-Powered Real-Time Voice-Clone Detection",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-tt-bg text-tt-text`}>
        {children}
      </body>
    </html>
  );
}
