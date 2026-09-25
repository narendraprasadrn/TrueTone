import { create } from 'zustand';
import { useEffect, useRef } from 'react';

export type CallClassification = 'LOW' | 'MEDIUM' | 'HIGH';

export interface ScoreUpdate {
  call_id: string;
  window_ts: string;
  score: number;
  call_score: number;
  classification: CallClassification;
  contributing_signals: Record<string, number>;
  tier: string;
}

export interface CallState {
  call_id: string;
  tier: string;
  classification: CallClassification;
  current_score: number;
  history: { time: string; score: number }[];
  contributing_signals: Record<string, number>;
  last_updated: number;
}

export interface AlertEvent {
  call_id: string;
  window_ts: string;
  score: number;
  tier: string;
}

interface DashboardStore {
  isConnected: boolean;
  activeCalls: Record<string, CallState>;
  activeAlerts: AlertEvent[];
  setConnected: (status: boolean) => void;
  addScoreUpdate: (update: ScoreUpdate) => void;
  triggerAlert: (alert: AlertEvent) => void;
  cleanupStaleCalls: () => void;
}

export const useDashboardStore = create<DashboardStore>((set) => ({
  isConnected: false,
  activeCalls: {},
  activeAlerts: [],
  
  setConnected: (status) => set({ isConnected: status }),
  
  addScoreUpdate: (update) => set((state) => {
    const existing = state.activeCalls[update.call_id] || {
      call_id: update.call_id,
      tier: update.tier,
      history: [],
    };
    
    // Keep last 30 data points for the sparkline
    const newHistory = [...existing.history, { 
      time: new Date(update.window_ts).toLocaleTimeString(), 
      score: update.call_score 
    }].slice(-30);
    
    return {
      activeCalls: {
        ...state.activeCalls,
        [update.call_id]: {
          ...existing,
          classification: update.classification,
          current_score: update.call_score,
          contributing_signals: update.contributing_signals,
          history: newHistory,
          last_updated: Date.now(),
        }
      }
    };
  }),
  
  triggerAlert: (alert) => set((state) => {
    // Only add if not already in active alerts
    if (state.activeAlerts.find(a => a.call_id === alert.call_id)) {
      return state;
    }
    return {
      activeAlerts: [...state.activeAlerts, alert]
    };
  }),
  
  cleanupStaleCalls: () => set((state) => {
    const now = Date.now();
    const active = { ...state.activeCalls };
    let changed = false;
    
    // Drop calls not updated in 10 seconds
    for (const [id, call] of Object.entries(active)) {
      if (now - call.last_updated > 10000) {
        delete active[id];
        changed = true;
      }
    }
    
    if (changed) {
      return { activeCalls: active };
    }
    return state;
  })
}));

export function useDashboardSocket(url: string = (process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000') + '/ws/dashboard') {
  const store = useDashboardStore();
  const ws = useRef<WebSocket | null>(null);
  
  useEffect(() => {
    let reconnectTimeout: NodeJS.Timeout;
    
    const connect = () => {
      ws.current = new WebSocket(url);
      
      ws.current.onopen = () => store.setConnected(true);

      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.event === 'score_update') {
          store.addScoreUpdate(data);
        } else if (data.event === 'alert_triggered') {
          store.triggerAlert({
            call_id: data.call_id,
            window_ts: data.window_ts,
            score: data.call_score,
            tier: data.tier
          });
        }
      };
      
      ws.current.onclose = () => {
        store.setConnected(false);
        console.log('WS disconnected. Reconnecting in 3s...');
        reconnectTimeout = setTimeout(connect, 3000);
      };
    };
    
    connect();
    
    // Cleanup interval
    const cleanupInterval = setInterval(() => {
      store.cleanupStaleCalls();
    }, 5000);
    
    return () => {
      clearTimeout(reconnectTimeout);
      clearInterval(cleanupInterval);
      if (ws.current) {
        ws.current.onclose = null;
        ws.current.close();
      }
    };
  }, [url]);
}
