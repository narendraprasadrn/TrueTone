"use client";

import { useState, useEffect } from "react";
import { Settings, Plus, Activity, RefreshCw, Cpu, Database, Bell, Lock, ActivitySquare, Server, AlertTriangle } from "lucide-react";

interface Endpoint {
  id: string;
  name: string;
  url: string;
  status: string;
}

export default function AdminPage() {
  const [endpoints, setEndpoints] = useState<Endpoint[]>([]);
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");

  const fetchEndpoints = async () => {
    try {
      const res = await fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000") + "/admin/siprec");
      const data = await res.json();
      setEndpoints(data);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchEndpoints();
  }, []);

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !url) return;
    try {
      await fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000") + "/admin/siprec", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, url })
      });
      setName("");
      setUrl("");
      fetchEndpoints();
    } catch (e) {
      console.error(e);
    }
  };

  const checkHealth = async (id: string) => {
    try {
      await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000"}/admin/siprec/${id}/health`, { method: "POST" });
      fetchEndpoints();
    } catch (e) {
      console.error(e);
    }
  };

  const renderStatusDot = (status: string) => {
    if (status === 'Loaded' || status === 'Active' || status === 'healthy') {
      return <div className="w-2 h-2 rounded-full bg-tt-teal"></div>;
    }
    if (status === 'Optional') {
      return <div className="w-2 h-2 rounded-full border border-tt-text-muted"></div>;
    }
    return <div className="w-2 h-2 rounded-full bg-tt-amber"></div>;
  };

  return (
    <div className="min-h-screen bg-tt-bg text-tt-text p-8 font-sans">
      <header className="flex items-center gap-3 mb-8 border-b border-tt-border pb-4">
        <Settings className="w-8 h-8 text-tt-teal" />
        <h1 className="text-[28px] font-bold tracking-tight text-tt-navy">Admin Console <span className="text-tt-text-muted font-normal text-[20px]">| Configurations</span></h1>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        
        {/* Model Status */}
        <div className="tt-card !py-5 flex flex-col gap-4">
          <div className="flex items-center gap-2 mb-2">
            <Cpu className="w-5 h-5 text-tt-teal" />
            <h2 className="text-[16px] font-bold text-tt-navy">Model Status</h2>
          </div>
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="text-[14px] text-tt-text-secondary font-medium">AASIST-L</span>
              <div className="flex items-center gap-2">
                {renderStatusDot('Loaded')}
                <span className="text-[13px] font-mono text-tt-navy">Loaded</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[14px] text-tt-text-secondary font-medium">Prosody Model</span>
              <div className="flex items-center gap-2">
                {renderStatusDot('Loaded')}
                <span className="text-[13px] font-mono text-tt-navy">Loaded</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[14px] text-tt-text-secondary font-medium">ECAPA-TDNN</span>
              <div className="flex items-center gap-2">
                {renderStatusDot('Optional')}
                <span className="text-[13px] font-mono text-tt-text-muted">Optional</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[14px] text-tt-text-secondary font-medium">Risk Engine</span>
              <div className="flex items-center gap-2">
                {renderStatusDot('Active')}
                <span className="text-[13px] font-mono text-tt-navy">Active</span>
              </div>
            </div>
          </div>
        </div>

        {/* Risk Thresholds */}
        <div className="tt-card !py-5 flex flex-col gap-4">
          <div className="flex items-center gap-2 mb-2">
            <ActivitySquare className="w-5 h-5 text-tt-amber" />
            <h2 className="text-[16px] font-bold text-tt-navy">Risk Thresholds</h2>
          </div>
          <p className="text-[13px] text-tt-text-muted">Threshold parameters for automated classification.</p>
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="text-[14px] text-tt-text-secondary font-medium">High Risk (Upper Bound)</span>
              <span className="font-mono text-[14px] text-tt-navy bg-tt-surface-muted px-2 py-0.5 rounded border border-tt-border">0.85</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[14px] text-tt-text-secondary font-medium">Medium Risk (Lower Bound)</span>
              <span className="font-mono text-[14px] text-tt-navy bg-tt-surface-muted px-2 py-0.5 rounded border border-tt-border">0.50</span>
            </div>
          </div>
        </div>

        {/* Alert Configuration */}
        <div className="tt-card !py-5 flex flex-col gap-4">
          <div className="flex items-center gap-2 mb-2">
            <Bell className="w-5 h-5 text-tt-teal" />
            <h2 className="text-[16px] font-bold text-tt-navy">Alert Configuration</h2>
          </div>
          <p className="text-[13px] text-tt-text-muted">Notification routing for flagged activities.</p>
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="text-[14px] text-tt-text-secondary font-medium">SOC Webhook</span>
              <span className="text-[13px] text-tt-teal font-medium">Enabled</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[14px] text-tt-text-secondary font-medium">Auto-Block Threshold</span>
              <span className="text-[13px] text-tt-text-muted">Disabled</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        
        {/* System Health / SIPREC Registration */}
        <div className="tt-card !py-5">
          <div className="flex items-center gap-2 mb-4">
            <Server className="w-5 h-5 text-tt-navy" />
            <h2 className="text-[16px] font-bold text-tt-navy">SIPREC System Integration</h2>
          </div>
          <p className="text-[13px] text-tt-text-muted mb-4">Register your session border controller or SIP recording endpoints here.</p>
          
          <form onSubmit={handleRegister} className="flex flex-col gap-4 mb-6 p-4 bg-tt-bg rounded-lg border border-tt-border">
            <div>
              <label className="block text-[12px] font-semibold text-tt-text-secondary uppercase tracking-wider mb-1.5">Endpoint Name</label>
              <input 
                value={name} onChange={e => setName(e.target.value)}
                placeholder="e.g. Acme Corp SBC"
                className="w-full bg-white border border-tt-border-strong rounded px-3 py-2 text-sm outline-none focus:border-tt-teal focus:ring-1 focus:ring-tt-teal transition-colors text-tt-navy"
              />
            </div>
            <div>
              <label className="block text-[12px] font-semibold text-tt-text-secondary uppercase tracking-wider mb-1.5">SIP URL / IP</label>
              <input 
                value={url} onChange={e => setUrl(e.target.value)}
                placeholder="sip:recording@10.0.0.5:5060"
                className="w-full bg-white border border-tt-border-strong rounded px-3 py-2 text-sm outline-none focus:border-tt-teal focus:ring-1 focus:ring-tt-teal transition-colors text-tt-navy"
              />
            </div>
            <button 
              type="submit"
              className="bg-tt-teal hover:bg-[#0d9488] text-white px-4 py-2 rounded-lg font-medium text-[13px] transition-colors mt-2 self-start flex items-center gap-2 shadow-sm"
            >
              <Plus className="w-4 h-4" /> Register Endpoint
            </button>
          </form>

          <h3 className="text-[14px] font-semibold text-tt-navy mb-3 flex items-center gap-2">
            <Activity className="w-4 h-4 text-tt-teal" /> Active Endpoints
          </h3>
          
          <div className="flex flex-col gap-3">
            {endpoints.length === 0 ? (
              <div className="text-center py-6 text-tt-text-muted border border-dashed border-tt-border-strong rounded-lg bg-tt-bg">
                No SIPREC endpoints registered.
              </div>
            ) : (
              endpoints.map(ep => (
                <div key={ep.id} className="bg-tt-bg border border-tt-border rounded-lg p-4 flex items-center justify-between">
                  <div>
                    <div className="font-medium text-tt-navy text-[14px]">{ep.name}</div>
                    <div className="text-[13px] font-mono text-tt-text-secondary mt-0.5">{ep.url}</div>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className={`text-[11px] px-2.5 py-1 rounded-sm font-bold tracking-wider uppercase border flex items-center gap-1.5 ${
                      ep.status === 'healthy' ? 'bg-tt-success-bg text-[#0F766E] border-transparent' :
                      'bg-tt-surface-muted text-tt-text-secondary border-tt-border'
                    }`}>
                      {renderStatusDot(ep.status)}
                      {ep.status}
                    </span>
                    <button 
                      onClick={() => checkHealth(ep.id)}
                      className="p-1.5 text-tt-text-muted hover:text-tt-navy hover:bg-tt-surface-muted rounded transition-colors"
                      title="Check Health"
                    >
                      <RefreshCw className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Privacy & Compliance */}
        <div className="tt-card !py-5 flex flex-col gap-4">
          <div className="flex items-center gap-2 mb-2">
            <Lock className="w-5 h-5 text-tt-navy" />
            <h2 className="text-[16px] font-bold text-tt-navy">Privacy & Compliance</h2>
          </div>
          <p className="text-[13px] text-tt-text-muted">Manage data retention and auditing policies.</p>
          <div className="flex flex-col gap-4 mt-2">
            <div className="flex items-start gap-3 p-3 bg-tt-bg rounded-lg border border-tt-border">
              <Database className="w-5 h-5 text-tt-text-secondary mt-0.5" />
              <div>
                <h4 className="text-[14px] font-bold text-tt-navy">Data Retention</h4>
                <p className="text-[13px] text-tt-text-secondary mt-1">Audit logs are retained for 90 days. Raw audio is never stored by default.</p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-3 bg-tt-bg rounded-lg border border-tt-border">
              <AlertTriangle className="w-5 h-5 text-tt-amber mt-0.5" />
              <div>
                <h4 className="text-[14px] font-bold text-tt-navy">Compliance Mode</h4>
                <p className="text-[13px] text-tt-text-secondary mt-1">Strict anonymization is active. Caller IDs are masked in Tier 2 views.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
