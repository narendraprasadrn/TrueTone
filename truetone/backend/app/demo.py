import asyncio
import os
import uuid
import soundfile as sf
from fastapi import APIRouter, BackgroundTasks

from app.simulator.simulator import CallSimulator
from app.preprocessing.pipeline import preprocess_for_detection
from app.risk_engine.context_stubs import evaluate_context_stubs
from app.pipeline import CallPipeline
from app.risk_engine.fusion import RiskEngine
from app.detection.aasist import AasistDetector
from app.detection.prosody import ProsodyDetector
from app.detection.speaker_verification import SpeakerVerifier

router = APIRouter()

demo_is_running = False

async def run_demo_calls(demo_mode: str = "tier1"):
    global demo_is_running
    try:
        demo_is_running = True
        
        # Initialize pipeline components (in production these would be singletons/DI)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        aasist = AasistDetector(os.path.join(base_dir, "third_party/aasist/models/weights/AASIST-L.pth"))
        prosody = ProsodyDetector(os.path.join(base_dir, "app/detection/models/prosody_lgbm.txt"))
        sv = SpeakerVerifier(savedir=os.path.join(base_dir, "pretrained_models/ecapa-tdnn"))
        re = RiskEngine(os.path.join(base_dir, "app/risk_engine/config.yaml"))
        pipeline = CallPipeline(re)
        
        # Enroll user
        sample_dir = os.path.join(os.path.dirname(base_dir), "sample_audio")
        gen_path = os.path.join(sample_dir, "genuine_sample.wav")
        cln_path = os.path.join(sample_dir, "cloned_sample.wav")
        
        if os.path.exists(gen_path):
            gen_audio, sr = sf.read(gen_path)
            sv.enroll("user_demo", gen_audio[:int(sr * 5.0)])
            
        sim = CallSimulator(window_duration_sec=2.0, step_duration_sec=1.0)
        
        async def process_call(filepath, call_id, label, is_first, transcript, tier):
            if not os.path.exists(filepath):
                return
            async for frame in sim.stream_audio_file(
                filepath=filepath,
                call_id=call_id,
                direction="inbound",
                label=label,
                first_time_caller=is_first,
                transcript_stub=transcript
            ):
                audio = frame["audio_data"]
                sr = frame["sr"]
                
                try:
                    processed = preprocess_for_detection(audio, sr, debug=False)
                    aasist_res = aasist.predict_aasist(processed, 16000)
                    a_score = aasist_res["spoof_score"]
                    p_res = prosody.score(processed, 16000)
                    p_score = p_res["spoof_score"]
                    s_score = sv.score(processed, "user_demo") if tier == "tier1" else None
                    
                    ctx_flags = evaluate_context_stubs(
                        call_id=frame["call_id"],
                        first_time_caller=frame["first_time_caller"],
                        transcript_stub=frame["transcript_stub"]
                    )
                    
                    win_res = pipeline.risk_engine.score_window(frame["call_id"], a_score, p_score, s_score, ctx_flags)
                    await pipeline.process_window(frame["call_id"], win_res, tier, {"aasist": "v1", "prosody": "v1"})
                except Exception as e:
                    print(f"Error processing frame for {call_id}: {e}")
                    # Skip to next frame/call gracefully
                    pass
                
        # Determine which calls to spawn
        tasks = []
        if demo_mode in ["tier1", "both"]:
            tasks.append(process_call(gen_path, f"call_gen_{uuid.uuid4().hex[:6]}", "genuine", False, "", "tier1"))
            tasks.append(process_call(cln_path, f"call_cln_{uuid.uuid4().hex[:6]}", "cloned", True, "please transfer the funds", "tier1"))
            
        if demo_mode in ["tier2", "both"]:
            # For tier2, we use no speaker enrollment, so s_score is None.
            tasks.append(process_call(cln_path, f"call_t2_{uuid.uuid4().hex[:6]}", "cloned", True, "verify your account", "tier2"))
            
        await asyncio.gather(*tasks)
        
    finally:
        demo_is_running = False

from pydantic import BaseModel
class DemoRequest(BaseModel):
    mode: str = "tier1"

@router.post("/demo/start")
async def start_demo(req: DemoRequest, background_tasks: BackgroundTasks):
    global demo_is_running
    if demo_is_running:
        return {"status": "error", "message": "Demo is already running"}
        
    background_tasks.add_task(run_demo_calls, req.mode)
    return {"status": "ok", "message": f"Demo started in mode {req.mode}"}

@router.get("/demo/status")
async def demo_status():
    global demo_is_running
    return {"running": demo_is_running}
