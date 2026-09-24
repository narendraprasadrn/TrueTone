export function formatRiskScore(score: number | null | undefined): string {
  if (score === null || score === undefined) return 'N/A';
  return `${(score * 100).toFixed(0)}%`;
}

export function formatScoreDecimal(score: number | null | undefined): string {
  if (score === null || score === undefined) return 'N/A';
  return score.toFixed(2);
}

export function aggregateWindowScores(windows: any[]) {
  if (!windows || windows.length === 0) {
    return { aasist: null, prosody: null, fused: null, count: 0, classification: 'LOW' };
  }
  
  let aasistSum = 0, aasistCount = 0;
  let prosodySum = 0, prosodyCount = 0;
  let fusedSum = 0, fusedCount = 0;

  for (const w of windows) {
    if (typeof w.aasist_score === 'number' && !isNaN(w.aasist_score)) {
      aasistSum += w.aasist_score;
      aasistCount++;
    }
    if (typeof w.prosody_score === 'number' && !isNaN(w.prosody_score)) {
      prosodySum += w.prosody_score;
      prosodyCount++;
    }
    if (typeof w.fused_score === 'number' && !isNaN(w.fused_score)) {
      fusedSum += w.fused_score;
      fusedCount++;
    }
  }

  const aasist = aasistCount > 0 ? aasistSum / aasistCount : null;
  const prosody = prosodyCount > 0 ? prosodySum / prosodyCount : null;
  const fused = fusedCount > 0 ? fusedSum / fusedCount : null;

  let classification = 'LOW';
  if (fused !== null) {
    if (fused > 0.69) classification = 'HIGH';
    else if (fused > 0.39) classification = 'MEDIUM';
    else classification = 'LOW';
  }

  return { aasist, prosody, fused, count: windows.length, classification };
}
