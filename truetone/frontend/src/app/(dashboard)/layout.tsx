"use client";

import Link from "next/link";
import { Shield, Database, Settings, Smartphone, Activity, Mic } from "lucide-react";
import { usePathname } from "next/navigation";
import { useDashboardStore } from "@/lib/ws-client";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const isConnected = useDashboardStore(state => state.isConnected);
  
  const navLinkClass = (path: string) => `flex items-center gap-3 px-3 py-2 rounded-md transition-colors ${
    pathname?.startsWith(path) 
      ? 'bg-tt-success-bg text-tt-navy border-l-[3px] border-tt-teal' 
      : 'text-tt-text-secondary hover:text-tt-navy hover:bg-tt-bg border-l-[3px] border-transparent'
  }`;
  
  const navIconClass = (path: string) => `w-5 h-5 ${
    pathname?.startsWith(path) ? 'text-tt-teal' : 'text-tt-text-muted'
  }`;

  return (
    <div className="flex h-screen bg-tt-bg text-tt-text w-full font-sans">
      {/* Sidebar */}
      <aside className="w-64 border-r border-tt-border bg-tt-surface flex flex-col shrink-0">
        <div className="p-6 border-b border-tt-border flex items-center gap-2">
          <Shield className="w-6 h-6 text-tt-teal" />
          <div>
            <h1 className="text-[20px] font-bold tracking-tight text-tt-navy">TrueTone</h1>
            <p className="text-[12px] text-tt-text-muted uppercase tracking-wider font-semibold">Shield Console</p>
          </div>
        </div>
        
        <nav className="flex-1 py-6 px-4 flex flex-col gap-1.5">
          <Link href="/dashboard" className={navLinkClass("/dashboard")}>
            <Shield className={navIconClass("/dashboard")} />
            <span className="font-medium text-[14px]">Dashboard</span>
          </Link>
          <Link href="/audit" className={navLinkClass("/audit")}>
            <Database className={navIconClass("/audit")} />
            <span className="font-medium text-[14px]">Audit Trail</span>
          </Link>
          <Link href="/admin" className={navLinkClass("/admin")}>
            <Settings className={navIconClass("/admin")} />
            <span className="font-medium text-[14px]">Admin</span>
          </Link>
          <div className="my-4 border-t border-tt-border"></div>
          <Link href="/test-bench" className={navLinkClass("/test-bench")}>
            <Activity className={navIconClass("/test-bench")} />
            <span className="font-medium text-[14px]">Test Bench</span>
          </Link>
          <Link href="/live-mic" className={navLinkClass("/live-mic")}>
            <Mic className={navIconClass("/live-mic")} />
            <span className="font-medium text-[14px]">Live Mic Test</span>
          </Link>
          <Link href="/individual" className="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-tt-bg text-tt-text-secondary hover:text-tt-navy transition-colors mt-8 border border-tt-border">
            <Smartphone className="w-5 h-5 text-tt-text-muted" />
            <span className="font-medium text-[14px]">Tier 2 View (Demo)</span>
          </Link>
        </nav>
        
        <div className="p-5 border-t border-tt-border">
          <div className="flex items-center gap-3 px-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-tt-teal animate-pulse-subtle' : 'bg-tt-text-muted'}`}></div>
            <span className="text-[13px] font-semibold tracking-wide uppercase text-tt-text-secondary">
              {isConnected ? 'System Online' : 'Connecting...'}
            </span>
          </div>
        </div>
      </aside>
      
      {/* Main Content */}
      <main className="flex-1 overflow-y-auto bg-tt-bg w-full">
        {children}
      </main>
    </div>
  );
}
