import argparse
import csv
import json
import time
from pathlib import Path
from app.common import ROOT,utc_now,write_json,sha256
from app.config import load_config
from app.runtime import Runtime


def benchmark(seconds,cameras,channels=4,microbatch=1,exercise_zones=False):
    cfg=load_config(cameras);cfg["cameras"]=cfg["cameras"][:channels]
    cfg["runtime"]["microbatch"]=microbatch
    if any(not c.get("enabled") for c in cfg["cameras"]):raise ValueError("enable_every_benchmark_camera")
    run=ROOT/"outputs/benchmarks"/("run-"+utc_now().replace(":","").replace(".","-"));run.mkdir(parents=True)
    inputs=[]
    for c in cfg["cameras"]:
        inputs.append({"camera_id":c["id"],"type":c["source_type"],"sha256":sha256(c["path"]) if c["source_type"]=="file" else None,"network_route":"not verified" if c["source_type"]=="rtsp" else "local file"})
    write_json(run/"inputs_manifest.json",{"inputs":inputs})
    runtime=Runtime(cfg,run/"runtime")
    if exercise_zones:
        if not all(c.get("demo") for c in cfg["cameras"]):raise ValueError("generated_test_zones_require_DEMO_sources")
        runtime.zones={c["id"]:[{"id":"demo-zone","name":"DEMO test zone","points":[[.01,.01],[.99,.01],[.99,.99],[.01,.99]],"margin":0,"enter_s":.2,"exit_s":.3}] for c in cfg["cameras"]}
    runtime.start()
    samples=[]
    try:
        deadline=time.monotonic()+300
        while runtime.loading and time.monotonic()<deadline:time.sleep(.2)
        if not runtime.detector.ready:raise RuntimeError(runtime.detector.error)
        write_json(run/"run_manifest.json",{"duration_s":seconds,"channels":channels,"config":cfg,"zones":runtime.zones,"config_revision":runtime.config_revision,"backend":runtime.detector.status(),"hardware":runtime.system.value,"browser_connected":False,"physical_camera_latency":False,
            "source_code_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/"app").glob('*.py'))}})
        start=time.monotonic()
        with open(run/"performance.csv","w",newline="",encoding="utf-8") as stream:
            fields=["elapsed_s","camera_id","capture_fps","inference_fps","preview_fps","preview_mode","p95_decode_to_encoded_ms","overlay_age_p95_ms","observation_age_s","drops","queue","process_ram_gb","vram_gb"]
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            while time.monotonic()-start<seconds:
                snapshot=runtime.snapshot();samples.append(snapshot)
                for c in snapshot["cameras"]:
                    writer.writerow({"elapsed_s":round(time.monotonic()-start,2),"camera_id":c["id"],"capture_fps":c["capture_decode_fps"],"inference_fps":c["inference_fps"],"preview_fps":c["preview_fps"],
                        "preview_mode":"live_display" if runtime.live_preview else "analysed_frame","overlay_age_p95_ms":(c.get("overlay_age_ms") or {}).get("p95"),
                        "p95_decode_to_encoded_ms":(c["decode_ready_to_encoded_ms"] or {}).get("p95"),"observation_age_s":c["last_observation_age_s"],"drops":c["frame_drops"],"queue":c["queue_size"],
                        "process_ram_gb":snapshot["system"].get("process_ram_gb"),"vram_gb":(snapshot["system"].get("gpu") or {}).get("vram_used_gb")})
                stream.flush();time.sleep(1)
        write_json(run/"summary.json",{"samples":len(samples),"final":samples[-1] if samples else {},"acceptance":False,"reason":"browser display, real camera event accuracy, and target laptop not verified"})
    finally:runtime.close()
    print(str(run))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seconds",type=int,default=1800);p.add_argument("--cameras",required=True);p.add_argument("--channels",type=int,choices=[1,2,4],default=4);p.add_argument("--microbatch",type=int,choices=[1,2,4],default=1);p.add_argument("--exercise-zones",action='store_true');a=p.parse_args();benchmark(a.seconds,a.cameras,a.channels,a.microbatch,a.exercise_zones)
