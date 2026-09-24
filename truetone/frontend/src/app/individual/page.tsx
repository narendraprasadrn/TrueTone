"use client";

import { useDashboardSocket, useDashboardStore } from "@/lib/ws-client";
import { useState, useEffect } from "react";
import { Phone, Mic, Volume2, Video, PhoneOff, AlertCircle, Radio } from "lucide-react";

export default function IndividualShieldPage() {
  useDashboardSocket();
  const activeAlerts = useDashboardStore(state => state.activeAlerts);
  
  // Filter for tier 2 alerts only
  const tier2Alerts = activeAlerts.filter(a => a.tier === "tier2");
  const latestAlert = tier2Alerts.length > 0 ? tier2Alerts[tier2Alerts.length - 1] : null;

  const [demoRunning, setDemoRunning] = useState(false);
  const [callDuration, setCallDuration] = useState(0);

  // Simple call timer
  useEffect(() => {
    const timer = setInterval(() => {
      setCallDuration(prev => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const startDemo = async () => {
    setDemoRunning(true);
    try {
      await fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000") + "/demo/start", { 
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "tier2" })
      });
    } catch (e) {
      console.error(e);
    }
    setTimeout(() => setDemoRunning(false), 5000);
  };

  return (
    <div className="min-h-screen bg-tt-bg flex items-center justify-center p-4 font-sans relative overflow-hidden text-tt-text">
      
      {/* Background Decor */}
      <div className="absolute top-0 left-0 w-full h-full flex flex-col items-center justify-center pointer-events-none opacity-20">
        <h1 className="text-[120px] font-black text-tt-border-strong tracking-tighter">TIER 2</h1>
        <p className="text-[32px] font-bold text-tt-text-muted mt-2">Individual Shield</p>
      </div>

      {/* Control Panel (Outside phone) */}
      <div className="absolute top-8 right-8 flex flex-col items-end gap-4 z-10">
        <div className="bg-tt-surface backdrop-blur border border-tt-border p-5 rounded-xl shadow-sm flex flex-col gap-3">
          <h2 className="text-[13px] font-semibold text-tt-text-secondary uppercase tracking-wider">Demo Controls</h2>
          <button 
            onClick={startDemo}
            disabled={demoRunning}
            className="bg-tt-teal hover:bg-[#0d9488] disabled:opacity-50 text-white px-5 py-2.5 rounded-lg font-medium text-sm transition-colors shadow-sm"
          >
            {demoRunning ? "Simulating Incoming Call..." : "Trigger Incoming AI Call"}
          </button>
        </div>
      </div>

      {/* Phone Mockup Frame */}
      <div className="relative w-[360px] h-[760px] bg-white rounded-[3rem] shadow-[0_20px_50px_rgba(11,31,58,0.1)] overflow-hidden border-[12px] border-[#CBD5E1] z-20">
        
        {/* Notch */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-32 h-7 bg-[#CBD5E1] rounded-b-3xl z-50"></div>
        
        {/* Status Bar */}
        <div className="absolute top-2 w-full px-6 flex justify-between items-center text-[11px] font-medium text-tt-text-secondary z-40">
          <span>12:00</span>
          <div className="flex gap-1.5 items-center">
            <span className="w-4 h-3 rounded-sm bg-tt-text-muted relative overflow-hidden">
              <span className="absolute bottom-0 right-0 w-full h-[60%] bg-tt-navy"></span>
            </span>
          </div>
        </div>

        {/* Call Screen UI */}
        <div className="w-full h-full flex flex-col pt-24 px-6 bg-white">
          
          <div className="text-center">
            <h2 className="text-3xl font-medium text-tt-navy mt-4">Unknown Caller</h2>
            <p className="text-tt-text-secondary mt-2 font-mono text-lg">{formatTime(callDuration)}</p>
          </div>

          <div className="mt-16 flex-1">
            {/* Contact Avatar Placeholder */}
            <div className="w-32 h-32 mx-auto rounded-full bg-tt-surface-muted border border-tt-border flex items-center justify-center">
              <Phone className="w-12 h-12 text-tt-text-muted" />
            </div>
          </div>

          {/* In-Call Audio Announcement Overlay (Subtle, secondary channel) */}
          {latestAlert && (
            <div className="mb-8 w-full bg-white border border-tt-amber rounded-lg p-3 flex items-center gap-3 animate-in slide-in-from-bottom-4 duration-500 shadow-[0_4px_20px_rgba(245,158,11,0.15)]">
              <div className="bg-tt-warning-bg p-2 rounded-full flex-shrink-0">
                <Radio className="w-5 h-5 text-tt-amber animate-pulse" />
              </div>
              <div className="text-sm font-medium text-tt-navy">
                <span className="text-tt-amber block text-[11px] font-bold uppercase tracking-wider mb-0.5">Audio Announcement</span>
                "Warning: Potential AI voice detected."
              </div>
            </div>
          )}

          {/* Call Controls */}
          <div className="grid grid-cols-3 gap-6 mb-12">
            {[
              { icon: Mic, label: "mute" },
              { icon: KeypadIcon, label: "keypad" },
              { icon: Volume2, label: "speaker" },
              { icon: PlusIcon, label: "add call" },
              { icon: Video, label: "FaceTime" },
              { icon: UserIcon, label: "contacts" },
            ].map((btn, i) => (
              <div key={i} className="flex flex-col items-center gap-2">
                <button className="w-16 h-16 rounded-full bg-tt-surface-muted flex items-center justify-center text-tt-text-secondary hover:bg-tt-border transition-colors border border-tt-border">
                  <btn.icon className="w-6 h-6" />
                </button>
                <span className="text-xs text-tt-text-secondary capitalize">{btn.label}</span>
              </div>
            ))}
          </div>

          {/* End Call */}
          <div className="flex justify-center mb-12">
            <button className="w-16 h-16 rounded-full bg-[#EF4444] flex items-center justify-center shadow-lg hover:bg-[#DC2626] transition-colors">
              <PhoneOff className="w-8 h-8 text-white" />
            </button>
          </div>

        </div>

        {/* Flash SMS Overlay (Class 0) */}
        {latestAlert && (
          <div className="absolute inset-0 bg-tt-navy/40 backdrop-blur-sm z-50 flex items-center justify-center animate-in fade-in duration-200">
            {/* The shake animation simulates phone vibration */}
            <div className="w-[85%] bg-white rounded-2xl border border-tt-border overflow-hidden shadow-2xl animate-[shake_0.4s_ease-in-out]">
              <div className="bg-tt-amber p-4 text-center">
                <h3 className="text-white font-bold tracking-widest uppercase flex items-center justify-center gap-2 text-[14px]">
                  <AlertCircle className="w-5 h-5" /> Flash SMS
                </h3>
              </div>
              <div className="p-6 text-center">
                <p className="text-tt-navy text-[15px] leading-relaxed font-medium">
                  <span className="text-tt-amber font-bold">⚠ TrueTone Alert:</span><br/>This call may involve an AI-cloned voice. 
                </p>
                <p className="text-tt-text-secondary text-[13px] mt-4">
                  Verify caller identity before sharing sensitive information.
                </p>
              </div>
              <div className="border-t border-tt-border">
                <button 
                  onClick={() => {
                    // Alert acknowledge endpoint was removed, just close locally in a real app
                  }}
                  className="w-full py-4 text-tt-teal font-semibold text-center hover:bg-tt-surface-muted transition-colors text-[15px]"
                >
                  Dismiss
                </button>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

// Dummy icons for buttons
function PlusIcon(props: any) {
  return <svg {...props} fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>;
}
function UserIcon(props: any) {
  return <svg {...props} fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>;
}
function KeypadIcon(props: any) {
  return <svg {...props} fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/></svg>;
}
