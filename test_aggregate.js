const { aggregateWindowScores } = require('./truetone/frontend/src/lib/utils.ts');
// Note: Can't easily require TS directly in node without ts-node. Let's just copy the function.

function aggregateWindowScores2(windows) {
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

const windows = [
  { aasist_score: 0.72, prosody_score: 0.00, fused_score: 0.54 },
  { aasist_score: 0.77, prosody_score: 0.00, fused_score: 0.57 },
  { aasist_score: 0.01, prosody_score: 0.25, fused_score: 0.23 },
  { aasist_score: 0.00, prosody_score: 0.01, fused_score: 0.15 }
];

console.log(aggregateWindowScores2(windows));
