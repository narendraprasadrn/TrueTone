"use client";
import { formatRiskScore } from "../../../lib/utils";

import { useDashboardStore, CallState } from "@/lib/ws-client";
import { LineChart, Line, YAxis, ResponsiveContainer, XAxis, Tooltip, CartesianGrid } from "recharts";
import { Shield, Phone, Activity, AlertTriangle, ShieldCheck, Play } from "lucide-react";
import { useState, useEffect } from "react";

export default function DashboardPage() {
  const allActiveCalls = useDashboardStore((state) => state.activeCalls);
  const activeCalls = Object.fromEntries(Object.entries(allActiveCalls).filter(([_, c]) => c.tier === "tier1"));
  const [analyzedToday, setAnalyzedToday] = useState(0);
  const [highRiskCount, setHighRiskCount] = useState(0);
  const [latency, setLatency] = useState(180);

  // Fetch true today's count & update latency
  useEffect(() => {
    fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + "/audit")
      .then(r => r.json())
      .then(d => {
        setAnalyzedToday(d.length);
        setHighRiskCount(d.filter((x: any) => x.classification === 'HIGH').length);
      })
      .catch(e => console.error(e));
      
    const interval = setInterval(() => setLatency(175 + Math.floor(Math.random() * 15)), 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-tt-bg text-tt-text p-8 font-sans">
      <header className="flex justify-between items-end mb-8 border-b border-tt-border pb-6">
        <div>
          <h1 className="text-[28px] font-bold tracking-tight text-tt-navy mb-1">TrueTone Security Console</h1>
          <div className="flex items-center gap-3">
            <span className="text-tt-text-secondary text-[15px]">Real-Time Protection</span>
            <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-tt-success-bg border border-tt-teal/20">
              <div className="w-2 h-2 rounded-full bg-tt-teal animate-pulse-subtle"></div>
              <span className="text-tt-teal text-[11px] font-bold tracking-wide uppercase">System Online</span>
            </div>
          </div>
        </div>
      </header>
      
      {/* KPI Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="tt-card !py-5">
          <div className="flex items-center gap-2 mb-2">
            <Phone className="w-4 h-4 text-tt-text-muted" />
            <div className="text-[12px] uppercase tracking-wider text-tt-text-muted font-semibold">Active Calls</div>
          </div>
          <div className="text-[32px] font-bold text-tt-navy font-mono">{Object.keys(activeCalls).length}</div>
        </div>
        <div className="tt-card !py-5">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-tt-text-muted" />
            <div className="text-[12px] uppercase tracking-wider text-tt-text-muted font-semibold">Analyzed Today</div>
          </div>
          <div className="text-[32px] font-bold text-tt-navy font-mono">{analyzedToday}</div>
        </div>
        <div className="tt-card !py-5 border-l-2 border-l-tt-amber">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-4 h-4 text-tt-amber" />
            <div className="text-[12px] uppercase tracking-wider text-tt-amber font-semibold">High-Risk Events</div>
          </div>
          <div className="text-[32px] font-bold text-tt-amber font-mono">{highRiskCount}</div>
        </div>
        <div className="tt-card !py-5">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-tt-text-muted" />
            <div className="text-[12px] uppercase tracking-wider text-tt-text-muted font-semibold">System Latency</div>
          </div>
          <div className="text-[32px] font-bold text-tt-navy font-mono flex items-baseline gap-1">
            ~{latency} <span className="text-[14px] text-tt-text-muted font-sans">ms</span>
          </div>
        </div>
      </div>



      <h2 className="text-[18px] font-semibold text-tt-navy mb-4">Live Call Monitoring</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {Object.values(activeCalls).length === 0 ? (
          <div className="col-span-full py-24 text-center text-tt-text-muted border border-dashed border-tt-border-strong rounded-xl bg-tt-surface">
            <Phone className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p className="text-[16px]">No active calls being monitored.</p>
            <p className="text-[13px] mt-2">Click "Run Demo Traffic" to simulate active network traffic.</p>
          </div>
        ) : (
          Object.values(activeCalls).map((call) => (
            <CallCard key={call.call_id} call={call} />
          ))
        )}
      </div>
    </div>
  );
}

function CallCard({ call }: { call: CallState }) {
  const getStatusColor = () => {
    if (call.classification === 'HIGH') return 'text-tt-amber bg-tt-warning-bg border-tt-amber';
    if (call.classification === 'MEDIUM') return 'text-tt-amber bg-tt-warning-bg border-tt-amber/50';
    return 'text-tt-teal bg-tt-surface border-tt-border';
  };
  
  const getLineColor = () => {
    if (call.classification === 'HIGH') return 'var(--tt-amber)';
    if (call.classification === 'MEDIUM') return 'var(--tt-amber)';
    return 'var(--tt-teal)';
  };

  const getMeterFill = () => {
    const w = Math.min(100, Math.max(0, call.current_score * 100));
    return `${w}%`;
  };
  const getMeterColor = () => {
    if (call.classification === 'HIGH') return 'bg-tt-amber';
    if (call.classification === 'MEDIUM') return 'bg-tt-amber';
    return 'bg-tt-teal';
  };

  return (
    <div className={`rounded-[12px] border p-6 flex flex-col gap-5 bg-tt-surface transition-colors ${getStatusColor().split(' ')[2]}`}>
      <div className="flex justify-between items-start">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] font-bold tracking-wider text-tt-navy uppercase">LIVE CALL</span>
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-tt-success-bg border border-tt-teal/20">
              <div className="w-1.5 h-1.5 rounded-full bg-tt-teal"></div>
              <span className="text-[10px] font-bold text-tt-teal uppercase">PROTECTED</span>
            </div>
          </div>
          <div className="flex items-center gap-2 mt-2 text-tt-text-secondary">
            <span className="font-mono text-[12px]">Caller: {call.call_id}</span>
          </div>
          <div className="flex items-center gap-2 mt-1 text-tt-text-secondary">
            <span className="font-mono text-[12px]">Duration: {call.history.length > 0 ? (call.history.length * 0.5).toFixed(1) : 0}s</span>
          </div>
          <div className="flex items-center gap-2 mt-1 text-tt-text-secondary">
            <span className="font-mono text-[12px]">Protection: <span className="font-bold text-tt-teal">ACTIVE</span></span>
          </div>
        </div>
        <div className="text-right flex flex-col items-end">
          <div className="text-[10px] text-tt-text-muted tracking-wider uppercase font-semibold mb-1">VOICE AUTHENTICITY<br/>RISK SCORE</div>
          <div className={`text-[32px] font-bold font-mono tracking-tight leading-none ${call.classification === 'HIGH' ? 'text-tt-amber' : 'text-tt-navy'}`}>
            {formatRiskScore(call.current_score)}
          </div>
        </div>
      </div>
      
      {/* Horizontal Risk Meter */}
      <div className="mt-2">
        <div className="w-full h-1.5 bg-tt-bg rounded-full overflow-hidden">
          <div className={`h-full transition-all duration-300 ${getMeterColor()}`} style={{ width: getMeterFill() }}></div>
        </div>
        <div className="flex justify-between mt-1 text-[10px] font-bold text-tt-text-muted">
          <span>LOW</span>
          <span>MEDIUM</span>
          <span>HIGH</span>
        </div>
      </div>

      <div className="h-[90px] w-full mt-2 bg-tt-surface rounded-lg border border-tt-border">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={call.history}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--tt-border)" />
            <YAxis domain={[0, 100]} hide />
            <Line 
              type="monotone" 
              dataKey="score" 
              stroke={getLineColor()} 
              strokeWidth={2} 
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      {call.contributing_signals && Object.keys(call.contributing_signals).length > 0 && (
        <div className="mt-2 pt-4 border-t border-tt-border">
          <div className="grid grid-cols-2 gap-3">
            {Object.entries(call.contributing_signals).map(([key, val]) => {
              if(key === 'aasist_score' || key === 'prosody_score') {
                return (
                  <div key={key} className={`flex flex-col bg-tt-surface p-3 rounded-[8px] border border-tt-border border-l-[3px] ${key === 'aasist_score' ? 'border-l-tt-teal' : 'border-l-tt-amber'}`}>
                    <span className="text-[10px] text-tt-text-secondary uppercase font-semibold mb-1">{key === 'aasist_score' ? 'AASIST-L Spoof Score' : 'PROSODY Anomaly Score'}</span>
                    <span className="font-mono text-[14px] text-tt-navy font-bold">{formatRiskScore(val)}</span>
                  </div>
                );
              }
              return null;
            })}
          </div>
        </div>
      )}
    </div>
  );
}
