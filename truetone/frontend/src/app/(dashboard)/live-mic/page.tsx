"use client";
import { formatRiskScore } from "../../../lib/utils";

import { useState, useRef, useEffect } from "react";
import { Mic, Square, Activity, ShieldCheck, AlertTriangle, Radio } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

type WSResult = {
  window_index: number;
  timestamp: number;
  aasist_score: number;
  prosody_score: number;
  speaker_score: number | null;
  fused_score: number;
  classification: "LOW" | "MEDIUM" | "HIGH";
};

export default function LiveMicTest() {
  const [isRecording, setIsRecording] = useState(false);
  const [results, setResults] = useState<WSResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [duration, setDuration] = useState(0);
  
  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const startRecording = async () => {
    setError(null);
    setResults([]);
    setDuration(0);
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        } 
      });
      streamRef.current = stream;

      const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:5000";
      const ws = new WebSocket(`${wsUrl}/ws/live-mic`);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsRecording(true);
        timerRef.current = setInterval(() => setDuration(d => d + 1), 1000);

        const ctx = new window.AudioContext({ sampleRate: 16000 });
        audioCtxRef.current = ctx;
        
        const source = ctx.createMediaStreamSource(stream);
        const processor = ctx.createScriptProcessor(4096, 1, 1);
        processorRef.current = processor;

        processor.onaudioprocess = (e) => {
          if (ws.readyState === WebSocket.OPEN) {
            const inputData = e.inputBuffer.getChannelData(0);
            ws.send(inputData.buffer);
          }
        };

        source.connect(processor);
        processor.connect(ctx.destination);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.error) {
            if (data.error === "No speech detected") {
              setResults(prev => {
                const next = [...prev, {
                  window_index: prev.length > 0 ? prev[prev.length - 1].window_index + 1 : 1,
                  timestamp: Date.now(),
                  aasist_score: 0,
                  prosody_score: 0,
                  speaker_score: null,
                  fused_score: 0,
                  classification: "no_speech"
                }];
                if (next.length > 20) return next.slice(next.length - 20);
                return next;
              });
              return;
            }
            setError(data.error);
            stopRecording();
            return;
          }
          setResults(prev => {
            const next = [...prev, data];
            if (next.length > 20) return next.slice(next.length - 20);
            return next;
          });
        } catch (e) {
          console.error("Failed to parse WS message", e);
        }
      };

      ws.onerror = () => {
        setError("WebSocket connection error");
        stopRecording();
      };
      
      ws.onclose = () => {
        if (isRecording) stopRecording();
      };

    } catch (err: any) {
      setError(err.message || "Microphone access denied");
    }
  };

  const stopRecording = () => {
    setIsRecording(false);
    if (timerRef.current) clearInterval(timerRef.current);
    
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, []);

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

  const latestResult = results.length > 0 ? results[results.length - 1] : null;

  return (
    <div className="p-8 max-w-7xl mx-auto font-sans bg-tt-bg min-h-screen text-tt-text">
      <div className="mb-8">
        <h1 className="text-[28px] font-bold text-tt-navy flex items-center gap-3">
          <Mic className="w-8 h-8 text-tt-teal" />
          Live Mic Test
        </h1>
        <p className="text-tt-text-secondary mt-2 text-[15px]">Test the TrueTone pipeline against live microphone audio in real-time.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Control Panel */}
        <div className="flex flex-col gap-6 h-fit">
          <div className="tt-card bg-tt-surface border-tt-border">
            <h2 className="text-[18px] font-semibold text-tt-navy flex items-center gap-2 mb-6 uppercase tracking-wider text-[12px]">
              <Radio className="w-4 h-4 text-tt-teal" />
              Capture Device
            </h2>
            
            <div className="flex flex-col items-center">
              {!isRecording ? (
                <button 
                  onClick={startRecording}
                  className="w-full flex items-center justify-center gap-2 bg-tt-teal hover:bg-[#0d9488] text-white py-3 px-6 rounded-lg font-medium transition-colors shadow-sm"
                >
                  <Mic className="w-5 h-5" />
                  Start Recording
                </button>
              ) : (
                <button 
                  onClick={stopRecording}
                  className="w-full flex items-center justify-center gap-2 bg-white border border-tt-border-strong hover:bg-tt-surface-muted text-[#B91C1C] py-3 px-6 rounded-lg font-medium transition-colors shadow-sm"
                >
                  <Square className="w-5 h-5" />
                  Stop Recording
                </button>
              )}
            </div>
            
            {isRecording && (
              <div className="mt-6 flex flex-col items-center justify-center gap-2">
                <div className="flex items-center gap-2 w-full h-8 px-8 justify-center overflow-hidden">
                  <div className="w-1 bg-tt-teal animate-wave" style={{animationDelay: '0ms'}}></div>
                  <div className="w-1 bg-tt-teal animate-wave" style={{animationDelay: '150ms'}}></div>
                  <div className="w-1 bg-tt-teal animate-wave" style={{animationDelay: '300ms'}}></div>
                  <div className="w-1 bg-tt-teal animate-wave" style={{animationDelay: '450ms'}}></div>
                  <div className="w-1 bg-tt-teal animate-wave" style={{animationDelay: '600ms'}}></div>
                  <div className="w-1 bg-tt-teal animate-wave" style={{animationDelay: '750ms'}}></div>
                </div>
                <span className="text-tt-text-secondary font-mono text-sm">
                  {Math.floor(duration / 60).toString().padStart(2, '0')}:{(duration % 60).toString().padStart(2, '0')}
                </span>
              </div>
            )}

            {error && (
              <div className="mt-6 p-3 bg-[#FEF2F2] border border-[#FCA5A5] rounded text-[#B91C1C] text-sm flex items-start gap-2">
                <AlertTriangle className="w-5 h-5 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>
          
          {/* Signal Cards */}
          {latestResult && (
            <div className="tt-card flex flex-col gap-4 bg-tt-surface border-tt-border">
              <h3 className="text-[12px] font-semibold text-tt-text-secondary uppercase tracking-wider">Raw Signals</h3>
              
              <div className="flex justify-between items-center p-3 rounded bg-tt-bg border border-tt-border">
                <span className="text-[14px] text-tt-navy font-medium">AASIST-L</span>
                <span className="text-tt-navy font-mono font-bold">{formatRiskScore(latestResult.aasist_score)}</span>
              </div>
              
              <div className="flex justify-between items-center p-3 rounded bg-tt-bg border border-tt-border">
                <span className="text-[14px] text-tt-navy font-medium">Prosody</span>
                <span className="text-tt-navy font-mono font-bold">{formatRiskScore(latestResult.prosody_score)}</span>
              </div>
              
              {latestResult.speaker_score !== null && (
                <div className="flex justify-between items-center p-3 rounded bg-tt-bg border border-tt-border">
                  <span className="text-[14px] text-tt-navy font-medium">Identity</span>
                  <span className="text-tt-navy font-mono font-bold">{formatRiskScore(latestResult.speaker_score)}</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Results Panel */}
        <div className="md:col-span-2 flex flex-col gap-6">
          {latestResult ? (
            <>
              {/* Verdict Banner */}
              <div className={`tt-card border flex flex-col items-center justify-center py-10 shadow-sm relative overflow-hidden transition-all duration-500 ${latestResult.classification === 'no_speech' || latestResult.classification === 'LOW' ? 'bg-tt-success-bg border-transparent' : getRiskBg(latestResult.classification)}`}>
                
                {latestResult.classification === 'HIGH' ? (
                  <AlertTriangle className="w-14 h-14 text-tt-amber mb-4 relative z-10" />
                ) : latestResult.classification === 'MEDIUM' ? (
                  <AlertTriangle className="w-14 h-14 text-tt-amber opacity-80 mb-4 relative z-10" />
                ) : (
                  <ShieldCheck className="w-14 h-14 text-tt-teal mb-4 relative z-10" />
                )}
                
                <h2 className={`text-[12px] uppercase tracking-[0.2em] font-bold mb-2 relative z-10 text-tt-text-secondary`}>
                  LIVE VOICE ANALYSIS
                </h2>
                
                <h3 className={`text-[36px] font-black tracking-tight relative z-10 ${latestResult.classification === 'no_speech' || latestResult.classification === 'LOW' ? 'text-tt-teal' : getRiskColor(latestResult.classification)}`}>
                  {latestResult.classification === 'HIGH' 
                    ? 'HIGH RISK' 
                    : latestResult.classification === 'MEDIUM'
                      ? 'ELEVATED RISK'
                      : 'LOW RISK'
                  }
                </h3>
                
                <p className="text-tt-text-secondary mt-4 text-[15px] relative z-10 flex items-center gap-2">
                  Fused Risk Score: <span className="font-mono font-bold text-tt-navy bg-white border border-tt-border px-2 py-1 rounded">{formatRiskScore(latestResult.fused_score)}</span>
                </p>
              </div>

              {/* Chart */}
              <div className="tt-card flex-1 min-h-[350px] flex flex-col bg-tt-surface border-tt-border">
                <h3 className="text-[16px] font-semibold text-tt-navy mb-6 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-tt-teal" />
                  Live Trajectory (Last 20 windows)
                </h3>
                <div className="flex-1 w-full relative">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={results} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--tt-border)" vertical={false} />
                      <XAxis 
                        dataKey="window_index" 
                        stroke="var(--tt-text-muted)" 
                        tickFormatter={(val) => `#${val}`} 
                        fontSize={12}
                        tickMargin={10}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis stroke="var(--tt-text-muted)" fontSize={12} domain={[0, 1]} axisLine={false} tickLine={false} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#ffffff', borderColor: 'var(--tt-border)', borderRadius: '8px', color: 'var(--tt-navy)' }}
                        itemStyle={{ color: 'var(--tt-navy)' }}
                        labelFormatter={(label) => `Window: #${label}`}
                      />
                      <Legend iconType="circle" wrapperStyle={{ fontSize: '13px', color: 'var(--tt-text-secondary)' }} />
                      <Line type="monotone" dataKey="aasist_score" name="AASIST-L" stroke="var(--tt-teal)" strokeWidth={2} dot={false} isAnimationActive={false} />
                      <Line type="monotone" dataKey="prosody_score" name="Prosody" stroke="var(--tt-amber)" strokeWidth={2} dot={false} isAnimationActive={false} />
                      <Line type="monotone" dataKey="fused_score" name="Fused Risk" stroke="var(--tt-navy)" strokeWidth={3} dot={{ r: 4, fill: 'var(--tt-navy)', strokeWidth: 0 }} activeDot={{ r: 6 }} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          ) : (
            <div className="tt-card flex flex-col items-center justify-center text-center h-full min-h-[500px] bg-tt-surface border-tt-border border-dashed">
              <Radio className="w-16 h-16 text-tt-border-strong mb-6" />
              <h3 className="text-[20px] font-medium text-tt-text-secondary">
                {isRecording ? "Listening for Audio..." : "Ready to Analyze"}
              </h3>
              <p className="text-tt-text-muted mt-2 max-w-sm text-[14px]">
                {isRecording 
                  ? "Speak into your microphone. Scores will appear once the first analysis window is processed."
                  : "Click Start Recording to begin processing real-time microphone audio."}
              </p>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
