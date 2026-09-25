"use client";
import { formatRiskScore, aggregateWindowScores } from "../../../lib/utils";

import { useState } from "react";
import { Upload, Activity, ShieldCheck, AlertTriangle, FileAudio, Settings } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

type WindowResult = {
  window_index: number;
  start_s: number;
  end_s: number;
  aasist_score: number;
  prosody_score: number;
  speaker_score: number | null;
  fused_score: number;
  classification: "LOW" | "MEDIUM" | "HIGH";
};

type AnalysisResult = {
  filename: string;
  duration_seconds: number;
  windows: WindowResult[];
  overall_score: number;
  overall_classification: "LOW" | "MEDIUM" | "HIGH";
};

export default function TestBench() {
  const [file, setFile] = useState<File | null>(null);
  const [enrolledId, setEnrolledId] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleUpload = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append("file", file);
    if (enrolledId) formData.append("enrolled_identity_id", enrolledId);

    try {
      const url = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
      const res = await fetch(`${url}/test/analyze-audio`, {
        method: "POST",
        body: formData
      });
      
      if (!res.ok) {
        const text = await res.text();
        let errMsg = "Analysis failed";
        try {
          const d = JSON.parse(text);
          errMsg = d.detail || "Analysis failed";
        } catch {
          errMsg = text.substring(0, 50) + (text.length > 50 ? "..." : "");
        }
        throw new Error(errMsg);
      }
      
      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (cls: string) => {
    if (cls === 'HIGH') return 'text-[#B45309]';
    if (cls === 'MEDIUM') return 'text-[#B45309]';
    return 'text-tt-teal';
  };

  const getRiskBg = (cls: string) => {
    if (cls === 'HIGH') return 'bg-tt-warning-bg border-tt-amber';
    if (cls === 'MEDIUM') return 'bg-tt-warning-bg border-transparent';
    return 'bg-tt-success-bg border-transparent';
  };

  const consolidated = result ? aggregateWindowScores(result.windows) : null;

  


  return (
    <div className="p-8 max-w-7xl mx-auto font-sans bg-tt-bg min-h-screen text-tt-text">
      <div className="mb-8">
        <h1 className="text-[28px] font-bold text-tt-navy flex items-center gap-3">
          <Activity className="w-8 h-8 text-tt-teal" />
          Diagnostic Test Bench
        </h1>
        <p className="text-tt-text-secondary mt-2 text-[15px]">Analyze recorded audio against TrueTone detection models.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Input Panel */}
        <div className="tt-card flex flex-col gap-6 h-fit bg-tt-surface border-tt-border">
          <h2 className="text-[18px] font-semibold text-tt-navy flex items-center gap-2">
            <Upload className="w-5 h-5 text-tt-teal" />
            INPUT AUDIO
          </h2>
          
          <div 
            className="border border-dashed border-tt-border-strong rounded-xl p-8 flex flex-col items-center justify-center bg-tt-bg hover:border-tt-teal transition-colors cursor-pointer group"
            onClick={() => document.getElementById('file-upload')?.click()}
          >
            <input 
              id="file-upload"
              type="file" 
              accept="audio/*" 
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            {file ? (
              <div className="text-center">
                <FileAudio className="w-12 h-12 text-tt-teal mx-auto mb-3" />
                <p className="text-tt-navy font-medium truncate max-w-[200px]">{file.name}</p>
                <p className="text-tt-text-muted text-[13px] mt-1">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                {/* Normally we'd calculate duration via AudioContext, mocked here if no result yet */}
              </div>
            ) : (
              <div className="text-center group-hover:text-tt-teal transition-colors">
                <Upload className="w-12 h-12 text-tt-border-strong mx-auto mb-3 group-hover:text-tt-teal transition-colors" />
                <p className="text-tt-navy font-medium">Click to upload audio</p>
                <p className="text-tt-text-muted text-[13px] mt-1">WAV, MP3, OGG up to 20MB</p>
              </div>
            )}
          </div>

          <div>
            <label className="block text-[13px] font-semibold text-tt-text-secondary uppercase mb-1">Enrolled Identity ID</label>
            <input 
              type="text"
              value={enrolledId}
              onChange={e => setEnrolledId(e.target.value)}
              placeholder="Optional"
              className="w-full bg-white border border-tt-border-strong rounded px-3 py-2.5 text-[14px] outline-none focus:border-tt-teal text-tt-navy"
            />
          </div>

          <button 
            onClick={handleUpload}
            disabled={!file || loading}
            className="w-full bg-tt-teal hover:bg-[#0d9488] disabled:opacity-50 text-white py-3 rounded-lg font-medium text-[15px] transition-all shadow-sm"
          >
            {loading ? "Analyzing..." : "Run Analysis"}
          </button>
          
          {error && (
            <div className="p-3 bg-[#FEF2F2] border border-[#FCA5A5] text-[#B91C1C] rounded-lg text-[13px]">
              Error: {error}
            </div>
          )}
        </div>

        {/* Results Panel */}
        <div className="md:col-span-2 flex flex-col gap-6">
          {result ? (
            <>
              {/* Header Result */}
              <div className={`rounded-xl border p-8 relative overflow-hidden bg-tt-surface ${getRiskBg(consolidated?.classification || 'LOW')}`}>
                <div className="absolute inset-0 bg-gradient-to-br from-transparent to-black/5 opacity-0"></div>
                <h2 className="text-[14px] uppercase tracking-wider font-bold text-tt-text-secondary relative z-10">
                  VOICE AUTHENTICITY RESULT
                </h2>
                
                <h3 className={`text-[42px] font-bold tracking-tight mt-2 relative z-10 flex items-center gap-3 ${getRiskColor(consolidated?.classification || 'LOW')}`}>
                  {consolidated?.classification === 'HIGH' && <AlertTriangle className="w-10 h-10" />}
                  {consolidated?.count === 0 
                    ? 'NO RESULT'
                    : consolidated?.classification === 'HIGH' 
                    ? 'HIGH RISK' 
                    : consolidated?.classification === 'MEDIUM'
                      ? 'MEDIUM RISK'
                      : 'LOW RISK'
                  }
                </h3>
                {consolidated?.count! > 0 && (
                  <p className="text-[12px] uppercase text-tt-text-muted mt-2 relative z-10 font-semibold tracking-wider">
                    CONSOLIDATED ACROSS {consolidated?.count} WINDOW{consolidated?.count !== 1 && 'S'}
                  </p>
                )}
              </div>
              
              {/* Metrics (3 Column) */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                 <div className="tt-card !py-6 flex flex-col border-l-[4px] border-l-tt-teal bg-tt-surface">
                    <span className="text-[12px] uppercase tracking-wider text-tt-text-secondary font-semibold mb-2">AASIST-L</span>
                    <span className="text-[11px] uppercase text-tt-text-muted font-bold mb-2">SPOOF SCORE</span>
                    <span className="text-[32px] font-bold text-tt-navy font-mono leading-none">{consolidated?.count === 0 ? 'N/A' : formatRiskScore(consolidated?.aasist)}</span>
                 </div>
                 <div className="tt-card !py-6 flex flex-col border-l-[4px] border-l-tt-amber bg-tt-surface">
                    <span className="text-[12px] uppercase tracking-wider text-tt-text-secondary font-semibold mb-2">PROSODY</span>
                    <span className="text-[11px] uppercase text-tt-text-muted font-bold mb-2">ANOMALY SCORE</span>
                    <span className="text-[32px] font-bold text-tt-navy font-mono leading-none">{consolidated?.count === 0 ? 'N/A' : formatRiskScore(consolidated?.prosody)}</span>
                 </div>
                 <div className="tt-card !py-6 flex flex-col border-l-[4px] border-l-tt-navy bg-tt-surface">
                    <span className="text-[12px] uppercase tracking-wider text-tt-text-secondary font-semibold mb-2">FUSED RISK</span>
                    <span className="text-[11px] uppercase text-tt-text-muted font-bold mb-2">OVERALL</span>
                    <span className="text-[32px] font-bold text-tt-navy font-mono leading-none">{consolidated?.count === 0 ? 'N/A' : formatRiskScore(consolidated?.fused)}</span>
                 </div>
              </div>

              {/* Chart */}
              <div className="tt-card bg-tt-surface">
                <h3 className="text-[16px] font-semibold text-tt-navy mb-6 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-tt-teal" />
                  Timeline Analysis
                </h3>
                <div className="h-[280px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={result.windows} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--tt-border)" vertical={false} />
                      <XAxis 
                        dataKey="window_index" 
                        stroke="var(--tt-text-muted)" 
                        tickFormatter={(val) => `#${val + 1}`} 
                        fontSize={12}
                        tickMargin={10}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis stroke="var(--tt-text-muted)" fontSize={12} axisLine={false} tickLine={false} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#ffffff', borderColor: 'var(--tt-border)', borderRadius: '8px', color: 'var(--tt-navy)' }}
                        itemStyle={{ color: 'var(--tt-navy)' }}
                        labelFormatter={(label) => `Window #${Number(label) + 1}`}
                      />
                      <Legend iconType="circle" wrapperStyle={{ fontSize: '13px', color: 'var(--tt-text-secondary)' }} />
                      <Bar dataKey="aasist_score" name="AASIST-L" fill="var(--tt-teal)" radius={[2, 2, 0, 0]} barSize={20} />
                      <Bar dataKey="prosody_score" name="Prosody" fill="var(--tt-amber)" radius={[2, 2, 0, 0]} barSize={20} />
                      <Bar dataKey="fused_score" name="Fused Risk" fill="var(--tt-navy)" radius={[2, 2, 0, 0]} barSize={20} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Raw Data Table */}
              <div className="tt-card overflow-hidden !p-0 bg-tt-surface">
                <div className="p-6 border-b border-tt-border">
                  <h3 className="text-[16px] font-semibold text-tt-navy">Window Analysis</h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="bg-tt-bg text-[11px] uppercase tracking-wider text-tt-text-muted">
                        <th className="p-4 font-semibold">WINDOW</th>
                        <th className="p-4 font-semibold">TIME (S)</th>
                        <th className="p-4 font-semibold text-right">AASIST-L</th>
                        <th className="p-4 font-semibold text-right">PROSODY</th>
                        {result.windows[0]?.speaker_score !== null && (
                          <th className="p-4 font-semibold text-right">IDENTITY</th>
                        )}
                        <th className="p-4 font-semibold text-right">FUSED RISK</th>
                        <th className="p-4 font-semibold text-center">CLASS</th>
                      </tr>
                    </thead>
                    <tbody className="text-[13px] divide-y divide-tt-border">
                      {result.windows.map((w, i) => (
                        <tr key={i} className="hover:bg-tt-bg transition-colors bg-white">
                          <td className="p-4 text-tt-text-muted">#{w.window_index + 1}</td>
                          <td className="p-4 text-tt-text-secondary font-mono">{w.start_s.toFixed(1)} - {w.end_s.toFixed(1)}</td>
                          <td className="p-4 text-tt-navy font-mono text-right">{formatRiskScore(w.aasist_score)}</td>
                          <td className="p-4 text-tt-navy font-mono text-right">{formatRiskScore(w.prosody_score)}</td>
                          {w.speaker_score !== null && (
                            <td className="p-4 text-tt-navy font-mono text-right">{formatRiskScore(w.speaker_score)}</td>
                          )}
                          <td className="p-4 font-mono font-bold text-right text-tt-navy">{formatRiskScore(w.fused_score)}</td>
                          <td className="p-4 text-center">
                            <span className={`text-[10px] uppercase px-2.5 py-1 rounded-[4px] font-bold border ${
                              w.classification === 'HIGH' ? 'bg-[#FFF7ED] text-[#B45309] border-[#F59E0B]' :
                              w.classification === 'MEDIUM' ? 'bg-[#FFFBEB] text-[#B45309] border-transparent' :
                              'bg-[#ECFDF5] text-[#0F766E] border-transparent'
                            }`}>
                              {w.classification}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          ) : (
            <div className="tt-card flex flex-col items-center justify-center text-center h-full min-h-[500px] bg-tt-surface border-tt-border border-dashed">
              <ShieldCheck className="w-16 h-16 text-tt-border-strong mb-6" />
              <h3 className="text-[20px] font-medium text-tt-text-secondary">Ready for Analysis</h3>
              <p className="text-tt-text-muted mt-2 max-w-sm text-[14px]">
                Upload an audio file on the left and run analysis to inspect the pipeline's detection metrics.
              </p>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
