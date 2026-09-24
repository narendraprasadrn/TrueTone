import { aggregateWindowScores, formatRiskScore } from './utils';

describe('aggregateWindowScores', () => {
  it('should calculate mean correctly for multiple windows', () => {
    const windows = [
      { aasist_score: 0.1, prosody_score: 0.2, fused_score: 0.3 },
      { aasist_score: 0.9, prosody_score: 0.8, fused_score: 0.7 }
    ];
    const res = aggregateWindowScores(windows);
    expect(res.aasist).toBeCloseTo(0.5);
    expect(res.prosody).toBeCloseTo(0.5);
    expect(res.fused).toBeCloseTo(0.5);
    expect(res.classification).toBe('MEDIUM');
  });

  it('should ignore null/invalid scores safely', () => {
    const windows = [
      { aasist_score: 0.1, prosody_score: null, fused_score: 0.3 },
      { aasist_score: null, prosody_score: 0.8, fused_score: 0.7 }
    ];
    const res = aggregateWindowScores(windows);
    expect(res.aasist).toBeCloseTo(0.1);
    expect(res.prosody).toBeCloseTo(0.8);
    expect(res.fused).toBeCloseTo(0.5);
  });

  it('should return null for empty/invalid arrays', () => {
    const res = aggregateWindowScores([]);
    expect(res.aasist).toBeNull();
    expect(res.fused).toBeNull();
    expect(res.classification).toBe('LOW');
  });
  
  it('prevents last-window bug by returning arithmetic mean of all windows', () => {
    const windows = [
      { aasist_score: 0.72, prosody_score: 0.00, fused_score: 0.54 },
      { aasist_score: 0.77, prosody_score: 0.00, fused_score: 0.57 },
      { aasist_score: 0.01, prosody_score: 0.25, fused_score: 0.23 },
      { aasist_score: 0.00, prosody_score: 0.01, fused_score: 0.15 }
    ];
    const res = aggregateWindowScores(windows);
    expect(res.aasist).toBeCloseTo(0.375);
    expect(res.prosody).toBeCloseTo(0.065);
    expect(res.fused).toBeCloseTo(0.3725);
    expect(res.classification).toBe('LOW');
  });
});
