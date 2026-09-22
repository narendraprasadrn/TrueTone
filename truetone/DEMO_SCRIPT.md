# TrueTone Demo Script

**Target Time:** 3:30  
**Target Audience:** Hackathon Judges / Technical Evaluators  
**Goal:** Prove the pipeline works end-to-end, validate the low-latency claim, and showcase both Tier 1 and Tier 2 UX designs.

---

### [0:00 - 0:30] Problem Framing & Introduction
**Action:** *Have `http://localhost:5000/dashboard` open, displaying empty call grid.*

**Speaker Track:**  
"Voice cloning scams cost victims billions, and current detection methods either require sending sensitive recordings to third-party clouds or rely on post-call analysis. **TrueTone** is an edge-ready, real-time detection pipeline that intercepts deepfakes *during* the live call in under 500ms. We do this by fusing acoustic artifacts, behavioral prosody, and speaker verification into a single continuous risk engine."

### [0:30 - 1:00] Tier 1 Dashboard (Enterprise/SBC)
**Action:** *Click **"Run Demo"** (ensure it's set to Both or Tier 1). Two call cards will appear.*

**Speaker Track:**  
"Here is our Tier 1 Organization Shield, designed for call centers monitoring SIPREC streams. As you can see, we are feeding live rolling audio windows into our detector models. 
*Point to the Genuine call.* 'This is normal speech—our engine sees low risk.' 
*Point to the Cloned call.* 'But here, our AASIST-L acoustic model detects synthetic signatures, causing the rolling average to spike instantly.'"

### [1:00 - 2:00] Alert Firing & Escalation
**Action:** *Wait for the Cloned Call to trigger the yellow/red alert banner. Do NOT click Acknowledge immediately.*

**Speaker Track:**  
"Once the threshold is crossed, it fires an alert. You can see exactly *why* it fired by looking at the contributing signals breakdown—this isn't a black box. 
If an operator ignores this, our auto-escalation timers kick in. *Wait for the pulse animation to appear.* There, it just escalated to a Tier 2 supervisor webhook. I'll acknowledge it now to clear the queue."
*(Click Acknowledge)*

### [2:00 - 2:30] Tier 2 Individual Shield (Phone Mockup)
**Action:** *Navigate to `http://localhost:5000/individual`. Click **"Trigger Incoming AI Call"**.*

**Speaker Track:**  
"But what about ordinary consumers? We don't want them looking at dashboards. For our Tier 2 Individual Shield, we simulate an incoming deepfake call. 
*(Wait for the screen to shake and the Flash SMS to appear).*
When the risk engine flags the audio, we trigger a Class 0 Flash SMS over the telecom network. It vibrates the phone and forces the message to render *over* the call screen instantly, alongside a subtle in-call audio warning, giving the victim immediate actionable intelligence without installing a third-party app."

### [2:30 - 3:00] Immutable Audit Log
**Action:** *Navigate to `http://localhost:5000/audit`.*

**Speaker Track:**  
"Because we operate in high-liability environments, explainability is mandatory. Our backend maintains an append-only, GET-only SQLite audit log. 
*(Click a row to expand).*
Every single alert permanently logs the exact floating-point outputs of all 3 models and the model versions used at that exact timestamp, proving compliance."

### [3:00 - 3:30] Closing / Architecture Recap
**Speaker Track:**  
"To be fully transparent: the AASIST-L and ECAPA-TDNN models running under the hood are real, pretrained neural networks, and the risk fusion engine is entirely functional. We simulated the telecom SIP trunking and the final webhook dispatches to avoid API rate limits today. TrueTone scales protection from the enterprise edge all the way to the end consumer. Thank you."

---
### ⚠️ Fragility Fallback Plan
* If the live demo crashes or the WebSocket drops unexpectedly, hit the **"Refresh Data"** button on the Audit Log. The backend writes to SQLite synchronously before broadcasting WS events. You can prove the pipeline worked by showing the newly generated timestamps in the DB, even if the frontend missed the visual render.
