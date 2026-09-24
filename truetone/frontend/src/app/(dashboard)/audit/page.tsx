"use client";
import { formatRiskScore } from "../../../lib/utils";

import React, { useState, useEffect } from "react";
import { format } from "date-fns";
import { Search, Filter, Database, ChevronDown, ChevronRight, Activity } from "lucide-react";

interface AuditEntry {
  id: number;
  call_id: string;
  timestamp: string;
  window_score: number;
  call_level_score: number;
  classification: string;
  model_versions: string;
  contributing_signals: string;
  outcome: string;
  tier: string;
}

export default function AuditPage() {
  const [logs, setLogs] = useState<AuditEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  
  // Filters
  const [filterClass, setFilterClass] = useState("all");
  const [filterTier, setFilterTier] = useState("all");
  const [filterOutcome, setFilterOutcome] = useState("all");

  const fetchLogs = async () => {
    try {
      setLoading(true);
      const res = await fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000") + "/audit");
      const data = await res.json();
      setLogs(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
  }, []);

  const filteredLogs = logs.filter(log => {
    if (filterClass !== "all" && log.classification !== filterClass) return false;
    if (filterTier !== "all" && log.tier !== filterTier) return false;
    if (filterOutcome !== "all" && log.outcome !== filterOutcome) return false;
    return true;
  });

  const getStatusColor = (classification: string) => {
    if (classification === 'HIGH') return 'text-[#B45309] bg-[#FFF7ED] border-[#F59E0B]';
    if (classification === 'MEDIUM') return 'text-[#B45309] bg-[#FFFBEB] border-transparent';
    return 'text-[#0F766E] bg-[#ECFDF5] border-transparent';
  };

  return (
    <div className="min-h-screen bg-tt-bg text-tt-text p-8 font-sans">
      <header className="flex justify-between items-center mb-8 border-b border-tt-border pb-4">
        <div className="flex items-center gap-3">
          <Database className="w-8 h-8 text-tt-teal" />
          <h1 className="text-[28px] font-bold tracking-tight text-tt-navy">Audit Trail <span className="text-tt-text-muted font-normal text-[20px]">| Immutable Log</span></h1>
        </div>
        <button onClick={fetchLogs} className="bg-white hover:bg-tt-bg text-tt-navy px-4 py-2 rounded-md font-medium text-[13px] transition-colors border border-tt-border shadow-sm">
          Refresh Data
        </button>
      </header>

      {/* Filters */}
      <div className="flex gap-4 mb-6 p-4 bg-tt-surface rounded-xl border border-tt-border items-center shadow-sm">
        <Filter className="w-5 h-5 text-tt-text-muted" />
        <select 
          value={filterClass} onChange={e => setFilterClass(e.target.value)}
          className="bg-tt-bg border border-tt-border rounded px-3 py-1.5 text-sm outline-none focus:border-tt-teal text-tt-text"
        >
          <option value="all">All Classifications</option>
          <option value="LOW">Low</option>
          <option value="MEDIUM">Medium</option>
          <option value="HIGH">High</option>
        </select>
        
        <select 
          value={filterTier} onChange={e => setFilterTier(e.target.value)}
          className="bg-tt-bg border border-tt-border rounded px-3 py-1.5 text-sm outline-none focus:border-tt-teal text-tt-text"
        >
          <option value="all">All Tiers</option>
          <option value="tier1">Tier 1</option>
          <option value="tier2">Tier 2</option>
        </select>

        <select 
          value={filterOutcome} onChange={e => setFilterOutcome(e.target.value)}
          className="bg-tt-bg border border-tt-border rounded px-3 py-1.5 text-sm outline-none focus:border-tt-teal text-tt-text"
        >
          <option value="all">All Outcomes</option>
          <option value="logged">Logged</option>
          <option value="alerted">Alerted</option>
        </select>
      </div>

      {/* Table */}
      <div className="bg-tt-surface border border-tt-border rounded-xl overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse min-w-[800px]">
            <thead>
              <tr className="bg-tt-bg border-b border-tt-border text-sm font-medium text-tt-text-muted">
                <th className="p-4 w-12"></th>
                <th className="p-4">Timestamp (UTC)</th>
                <th className="p-4">Call ID</th>
                <th className="p-4">Classification</th>
                <th className="p-4">Risk Score</th>
                <th className="p-4">Tier</th>
                <th className="p-4">Outcome</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              {loading ? (
                <tr><td colSpan={7} className="p-8 text-center text-tt-text-secondary">Loading audit logs...</td></tr>
              ) : filteredLogs.length === 0 ? (
                <tr><td colSpan={7} className="p-8 text-center text-tt-text-secondary">No logs match the current filters.</td></tr>
              ) : (
                filteredLogs.map(log => (
                  <React.Fragment key={log.id}>
                    <tr 
                      className="border-b border-tt-border hover:bg-tt-bg cursor-pointer transition-colors"
                      onClick={() => setExpandedId(expandedId === log.id ? null : log.id)}
                    >
                      <td className="p-4 text-tt-text-secondary">
                        {expandedId === log.id ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                      </td>
                      <td className="p-4 font-mono text-xs text-tt-text-secondary">{format(new Date(log.timestamp), "yyyy-MM-dd HH:mm:ss.SSS")}</td>
                      <td className="p-4 font-mono text-tt-navy">{log.call_id}</td>
                      <td className="p-4">
                        <span className={`text-xs px-2 py-0.5 rounded-sm font-semibold border ${getStatusColor(log.classification)}`}>
                          {log.classification.toUpperCase()}
                        </span>
                      </td>
                      <td className="p-4 font-mono font-semibold text-tt-navy">{formatRiskScore(log.call_level_score)}</td>
                      <td className="p-4 text-tt-text-muted uppercase text-xs font-semibold">{log.tier}</td>
                      <td className="p-4">
                        <span className={`text-xs px-2 py-1 rounded font-medium ${
                          log.outcome === 'alerted' ? 'bg-tt-warning-bg text-[#B45309]' :
                          'text-tt-text-secondary bg-tt-surface-muted'
                        }`}>
                          {log.outcome}
                        </span>
                      </td>
                    </tr>
                    
                    {/* Expanded Row */}
                    {expandedId === log.id && (
                      <tr className="bg-tt-bg border-b border-tt-border shadow-inner">
                        <td colSpan={7} className="p-6">
                          <div className="grid grid-cols-2 gap-6">
                            <div className="bg-tt-surface rounded-lg p-4 border border-tt-border">
                              <h4 className="text-[11px] text-tt-text-muted uppercase font-bold tracking-wider mb-3">Contributing Signals</h4>
                              <pre className="text-tt-navy font-mono text-[11px] whitespace-pre-wrap">
                                {JSON.stringify(JSON.parse(log.contributing_signals), null, 2)}
                              </pre>
                            </div>
                            <div className="bg-tt-surface rounded-lg p-4 border border-tt-border">
                              <h4 className="text-[11px] text-tt-text-muted uppercase font-bold tracking-wider mb-3">Model Versions</h4>
                              <pre className="text-tt-navy font-mono text-[11px] whitespace-pre-wrap">
                                {JSON.stringify(JSON.parse(log.model_versions), null, 2)}
                              </pre>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
