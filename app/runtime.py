from __future__ import annotations
import threading
import time
import queue
import os
from dataclasses import asdict,dataclass
from pathlib import Path
import cv2
import numpy as np
from app.capture import Capture,LatestSlot
from app.common import ROOT,read_json,write_json,revision,utc_now
from app.events import EventEngine
from app.geometry import overlap_fraction
from app.inference import Detector
from app.monitoring import Metrics,SystemMonitor
from app.ppe import PPE
from app.rules import Rules
from app.storage import Store
from app.tracking import TrackerBank
from app.types import Frame,RuleObservation


def draw_labeled_box(image,box,label,color,below=False):
    """Draw one display box; keep its label visible at frame boundaries."""
    h,w=image.shape[:2]
    x1,y1,x2,y2=(int(v) for v in box)
    x1,x2=max(0,min(x1,w-1)),max(0,min(x2,w-1))
    y1,y2=max(0,min(y1,h-1)),max(0,min(y2,h-1))
    if x2<=x1 or y2<=y1:return
    cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
    font=cv2.FONT_HERSHEY_SIMPLEX;font_scale=.55;padding=5
    text_width,text_height=cv2.getTextSize(label,font,font_scale,1)[0]
    label_width=min(w,text_width+2*padding);label_height=text_height+2*padding+3
    left=max(0,min(x1,w-label_width))
    top=y2+2 if below else y1-label_height-2
    if top+label_height>h:top=y1-label_height-2
    top=max(0,top)
    cv2.rectangle(image,(left,top),(left+label_width-1,min(h-1,top+label_height)),color,-1)
    ink=(255,255,255) if color==(0,0,255) else (15,20,25)
    cv2.putText(image,label,(left+padding,top+padding+text_height),font,font_scale,ink,1,cv2.LINE_AA)


def near_person(box,person):
    """True when most of a PPE box lies on a person box grown by 10% per side."""
    x1,y1,x2,y2=person;dx=.1*(x2-x1);dy=.1*(y2-y1)
    return overlap_fraction(box,(x1-dx,y1-dy,x2+dx,y2+dy))>.5


@dataclass(frozen=True)
class AnalysisSnapshot:
    session_id:str
    frame_id:int
    observed_at:float
    source_size:tuple[int,int]
    tracks:list
    ppe:dict


def overlay_for(display,analysis,ttl_s,*,now=None):
    """Choose the boxes for a displayed frame that was not itself analysed.

    The analysis must precede the display, match its session and source size,
    and remain fresh both relative to that frame and to the wall clock.
    The stamp says which frame the boxes belong to; they are not new detections.
    Otherwise the picture is shown bare with a warning, so boxes that stopped
    updating never look current. Returns (tracks,ppe,note,warning,age_s).
    """
    if analysis is not None:
        current=time.monotonic() if now is None else now
        age=display.received_monotonic-analysis.observed_at
        if (analysis.session_id==display.session_id and analysis.frame_id<=display.frame_id
            and analysis.source_size==(display.source_width,display.source_height)
            and 0<=age<ttl_s and 0<=current-analysis.observed_at<ttl_s):
            return analysis.tracks,analysis.ppe,f"boxes: frame {analysis.frame_id}, {age*1000:.0f} ms earlier",None,age
    return [],{},None,"NO CURRENT ANALYSIS",None


def render(frame,tracks,ppe,zones,width,model_ready,note=None,warning=None,fast=False):
    source_h,source_w=frame.image.shape[:2]
    scale=min(1.,width/source_w)
    # Ultralytics turns OpenCV threading off, so area resampling of a 2560x1440
    # frame takes about 13 ms here against about 2 ms for bilinear. `fast` is for
    # the live picture only; evidence keeps the area filter.
    method=cv2.INTER_LINEAR if fast else cv2.INTER_AREA
    image=cv2.resize(frame.image,(int(source_w*scale),int(source_h*scale)),interpolation=method) if scale<1 else frame.image.copy()
    h,w=image.shape[:2]
    for zone in zones:
        points=np.asarray([[x*w,y*h] for x,y in zone["points"]],np.int32)
        cv2.polylines(image,[points],True,(80,180,255),2)
    colors={"person":(235,195,60),"helmet":(80,220,160),"safety_vest":(90,220,240),"forklift":(70,130,250)}
    people=[t.xyxy for t in tracks if t.observed and t.class_name=="person"]
    for obj in tracks:
        if not obj.observed or obj.class_name not in colors:continue
        # A helmet or vest matters only on a worker. One that lies mostly outside
        # every observed person (a screen, a bag) is not drawn; this changes the
        # picture only, never the detections or the PPE state.
        if obj.class_name in {"helmet","safety_vest"} and not any(near_person(obj.xyxy,p) for p in people):continue
        x1,y1,x2,y2=(int(v*scale) for v in obj.xyxy);color=colors[obj.class_name]
        text=f"{obj.class_name} {obj.confidence:.2f}"+(f" #{obj.track_id}" if obj.track_id is not None else "")
        states=ppe.get(obj.key,{}) if obj.class_name=="person" and model_ready else {}
        if obj.class_name=="person":
            draw_labeled_box(image,(x1,y1,x2,y2),text.replace("person","PERSON",1),color,below=True)
            for kind,label in (("helmet","NO - HARD HAT"),("safety_vest","NO-VEST")):
                result=states.get(kind,{})
                if result.get("state")=="missing" and result.get('region_xyxy') is not None:
                    draw_labeled_box(image,[v*scale for v in result['region_xyxy']],label,(0,0,255),below=kind=="safety_vest")
            continue
        cv2.rectangle(image,(x1,y1),(x2,y2),color,2);cv2.putText(image,text,(x1,max(16,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    stamp=f"{frame.camera_id} / {frame.session_id[:8]} / frame {frame.frame_id}"+(f" / {note}" if note else "")
    cv2.putText(image,stamp,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    if not model_ready:cv2.putText(image,"MODEL NOT READY - NO SAFETY ANALYSIS",(12,52),cv2.FONT_HERSHEY_SIMPLEX,.5,(50,150,255),1)
    elif warning:cv2.putText(image,warning,(12,52),cv2.FONT_HERSHEY_SIMPLEX,.5,(50,150,255),1)
    ok,jpeg=cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,80])
    if not ok:raise RuntimeError("jpeg_encode_failed")
    return jpeg.tobytes()


class Runtime:
    def __init__(self,config,output=None):
        self.config=config;self.lock=threading.RLock();self.stop_event=threading.Event()
        self.cameras={c["id"]:Capture(c) for c in config["cameras"]}
        self.metrics={cid:Metrics() for cid in self.cameras};self.latest={};self.render_slots={cid:LatestSlot() for cid in self.cameras}
        self.detector=Detector(config["runtime"]);self.tracker=TrackerBank(config["runtime"]["observation_ttl_s"],config['runtime'].get('confidence',.25),config['runtime'].get('tracking_low_confidence',.1))
        rules=config["rules"]
        self.ppe=PPE(rules["ppe_confirm_s"],rules["ppe_clear_s"],rules["max_observation_gap_s"],rules["min_person_height"])
        self.rules=Rules(rules);self.events=EventEngine(rules["max_observation_gap_s"],rules["cooldown_s"])
        self.health_events=EventEngine(max_gap_s=3,cooldown_s=5)
        self.store=Store(output or ROOT/"outputs/runtime",config["runtime"]["storage"])
        self.system=SystemMonitor();self.zones={cid:read_json(ROOT/"configs/zones"/(cid+".json"),[]) for cid in self.cameras}
        self.thread=threading.Thread(target=self.run,name="gpu-scheduler",daemon=True)
        self.renderer=threading.Thread(target=self.render_loop,name="preview-encoder",daemon=True)
        self.viewer=threading.Thread(target=self.view_loop,name="live-preview",daemon=True)
        self.error=None;self.loading=True
        self.evidence_render=queue.Queue(maxsize=16)
        self.evidence_requested=set()
        self.scheduler_cursor=0
        self.started=False;self.ownership=None
        self.rule_revision=None
        # live_preview shows every decoded frame with the newest analysis drawn on
        # it, so the picture no longer waits for inference. One result per camera.
        self.live_preview=bool(config["runtime"].get("live_preview",False));self.analysis={};self.placeholders={};self.preview_expires={}

    @property
    def config_revision(self):
        with self.lock:return revision({"base":self.config["revision"],"zones":self.zones})

    def start(self):
        if self.started:return
        directory=ROOT/".runtime";directory.mkdir(exist_ok=True)
        self.ownership=open(directory/"pipeline.lock","a+b");self.ownership.write(b"1");self.ownership.flush();self.ownership.seek(0)
        try:
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(self.ownership.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.ownership.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.ownership.close();self.ownership=None
            raise RuntimeError("another_process_owns_this_project_GPU_and_cameras_use_one_worker")
        self.store.thread.start();self.system.thread.start();self.renderer.start();self.thread.start()
        if self.live_preview:self.viewer.start()
        self.started=True
        for capture in self.cameras.values():
            if capture.config.get("enabled"):capture.start()

    def close(self):
        self.stop_event.set()
        for capture in list(self.cameras.values()):capture.stop()
        self.thread.join(timeout=10);self.renderer.join(timeout=5)
        if self.viewer.is_alive():self.viewer.join(timeout=5)
        self.system.stop.set();self.system.thread.join(timeout=3);self.store.close()
        if self.ownership:self.ownership.close();self.ownership=None
        self.started=False

    def add_camera(self,config):
        with self.lock:
            if config["id"] in self.cameras:raise ValueError("camera_id_exists")
            if len(self.cameras)>=4:raise ValueError("four_camera_limit")
            cid=config["id"];self.cameras[cid]=Capture(config);self.metrics[cid]=Metrics();self.render_slots[cid]=LatestSlot();self.zones[cid]=[]
            import yaml
            path=ROOT/"configs/cameras.local.yaml"
            path.write_text(yaml.safe_dump({"cameras":[c.config for c in self.cameras.values()]},allow_unicode=True),encoding="utf-8")
            self.config["revision"]=revision([c.config for c in self.cameras.values()])
        if config.get("enabled"):self.cameras[cid].start()

    def set_zones(self,camera_id,zones):
        with self.lock:
            write_json(ROOT/"configs/zones"/(camera_id+".json"),zones);self.zones[camera_id]=zones
        return self.config_revision

    def camera_status(self):
        output=[]
        for cid,capture in list(self.cameras.items()):
            metrics=self.metrics[cid].snapshot();age=metrics["last_observation_age_s"]
            status=capture.status
            if status=="connected" and capture.last_received is not None and time.monotonic()-capture.last_received>2:status="stale"
            output.append({"id":cid,"name":capture.config["name"],"mode":capture.mode,"status":status,"error":capture.error,
                "configured":bool(capture.source()),"session_id":capture.session_id,"zones":len(self.zones[cid]),
                "analysis_live":self.detector.ready and status=="connected" and age is not None and age<self.config["runtime"]["observation_ttl_s"],
                **capture.metrics(),**metrics})
        return output

    def snapshot(self):
        return {"time":utc_now(),"config_revision":self.config_revision,"cameras":self.camera_status(),"model":self.detector.status(),
            "system":self.system.value,"storage":{"error":self.store.error,"queued":self.store.jobs.qsize(),"dropped":self.store.dropped,
            "failed_jobs":self.store.failed,"compressed_buffer_bytes":self.store.buffer_bytes},"loading":self.loading,"error":self.error}

    def run(self):
        try:self.detector.load()
        except Exception as error:
            self.detector.error=type(error).__name__;self.detector.degraded=True;self.error="model_load_failed"
        self.loading=False;last_health=0
        while not self.stop_event.is_set():
            worked=False;batch=[];camera_items=list(self.cameras.items())
            order=camera_items[self.scheduler_cursor:]+camera_items[:self.scheduler_cursor]
            for cid,capture in order:
                if self.stop_event.is_set():break
                frame=capture.slot.take()
                if frame is None:continue
                if frame.session_id!=capture.session_id:continue
                if time.monotonic()-frame.received_monotonic>2:continue
                batch.append((cid,capture,frame))
                self.scheduler_cursor=(camera_items.index((cid,capture))+1)%max(1,len(camera_items))
                if len(batch)>=self.config["runtime"].get("microbatch",1):break
            start=time.perf_counter()
            try:
                predictions=self.detector.predict_many([f for _,_,f in batch]) if batch and self.detector.ready else [[] for _ in batch]
            except Exception as error:
                self.error="inference_"+type(error).__name__
                if self.detector.device.startswith("cuda"):
                    try:self.detector.fallback_cpu()
                    except Exception:self.detector.ready=False;self.detector.error="cpu_fallback_failed"
                else:self.detector.ready=False;self.detector.error="inference_failed"
                predictions=[[] for _ in batch]
                for cid,_,_ in batch:
                    for event in self.events.suspend_camera(cid):self.store.save_event(event)
                batch=[]
            for (cid,capture,frame),detections in zip(batch,predictions):
                worked=True;tracks=[];ppe={};changes=[]
                with self.lock:
                    current_zones=list(self.zones[cid]);current_revision=self.config_revision
                try:
                    if self.detector.ready:
                        tracks=self.tracker.update(frame,detections)
                        ppe=self.ppe.update(frame,tracks,self.detector.classes|self.detector.auxiliary_classes)
                        if self.rule_revision!=current_revision:
                            self.rules=Rules(self.config["rules"]);self.rule_revision=current_revision
                        observations=self.rules.update(frame,tracks,ppe,current_zones,capture.config)
                        for observation in observations:observation.key+=":"+current_revision
                        changes=self.events.update(frame,observations,";".join(self.detector.hashes),current_revision)
                        values=[item["state"] for person in ppe.values() for item in person.values()]
                        self.metrics[cid].ppe_unknown=sum(s=="unknown" for s in values)/len(values) if values else 0.
                        self.metrics[cid].record(frame,(time.perf_counter()-start)*1000)
                    for event in changes:
                        event["source_mode"]=capture.mode
                        self.store.save_event(event)
                        if event["status"]=="active" and event["id"] not in self.evidence_requested:
                            try:
                                self.evidence_render.put_nowait((frame,tracks,ppe,current_zones,[event]))
                                self.evidence_requested.add(event["id"])
                            except queue.Full:self.store.error="evidence_render_queue_full"
                    if len(self.evidence_requested)>10000:self.evidence_requested={e["id"] for e in self.events.states.values()}
                    self.render_slots[cid].put((frame,tracks,ppe,current_zones,[]))
                    self.analysis[cid]=AnalysisSnapshot(frame.session_id,frame.frame_id,frame.received_monotonic,
                        (frame.source_width,frame.source_height),tracks,ppe)
                except Exception as error:
                    self.error=type(error).__name__
                    for event in self.events.suspend_camera(cid):self.store.save_event(event)
            now=time.monotonic()
            if now-last_health>1:
                self.check_health();last_health=now
            if not worked:self.stop_event.wait(.003)

    def check_health(self):
        for cid,capture in list(self.cameras.items()):
            status=capture.status;offline=status in {"offline","stale"} or (status=="connected" and capture.last_received is not None and time.monotonic()-capture.last_received>2)
            if status!="connected" or offline:
                for event in self.events.suspend_camera(cid):self.store.save_event(event)
            frame=Frame(cid,"system",0,1,1,None,utc_now(),time.monotonic(),np.zeros((1,1,3),np.uint8),source_mode=capture.mode)
            observations=[RuleObservation(cid+":offline","camera_offline",offline if capture.source() and status!="stopped" else None,
                basis={"connection":status},confirm_s=1.,clear_s=1.),
                RuleObservation(cid+":degraded","inference_degraded",self.detector.degraded or not self.detector.ready or self.error is not None,
                basis=self.detector.status(),confirm_s=0,clear_s=1.)]
            for event in self.health_events.update(frame,observations,";".join(self.detector.hashes),self.config_revision):
                event["source_mode"]=capture.mode;self.store.save_event(event)

    def render_loop(self):
        last={}
        while not self.stop_event.is_set():
            worked=False
            try:
                frame,tracks,ppe,zones,changes=self.evidence_render.get_nowait()
                jpeg=render(frame,tracks,ppe,zones,self.config["runtime"]["preview_width"],self.detector.ready)
                for event in changes:self.store.save_event(event,frame,jpeg)
                self.evidence_render.task_done();worked=True
            except queue.Empty:pass
            except Exception as error:self.store.error="evidence_render_"+type(error).__name__
            for cid,slot in list(self.render_slots.items()):
                now=time.monotonic()
                if now<last.get(cid,0):continue
                item=slot.take()
                if item is None:continue
                worked=True;frame,tracks,ppe,zones,changes=item
                if frame.session_id!=self.cameras[cid].session_id:continue
                try:
                    jpeg=render(frame,tracks,ppe,zones,self.config["runtime"]["preview_width"],self.detector.ready)
                    last[cid]=max(now,last.get(cid,now))+1/self.config["runtime"]["preview_fps"]
                    if not self.live_preview:
                        self.latest[cid]=(frame,jpeg);self.metrics[cid].preview.append(time.monotonic())
                        self.metrics[cid].latency.append((time.monotonic()-frame.received_monotonic)*1000)
                    # Evidence clips are always built from analysed frames carrying
                    # their own boxes, whatever the live picture shows.
                    self.store.add_frame(frame,jpeg)
                    for event in changes:self.store.save_event(event,frame,jpeg)
                except Exception as error:self.error="render_or_evidence_"+type(error).__name__
            if not worked:self.stop_event.wait(.005)

    def view_loop(self):
        """Live picture: each decoded frame with the newest analysis drawn on it.

        Runs apart from render_loop so a burst of evidence rendering cannot stall
        the picture. It only reads analysis results and never feeds evidence.
        """
        shown={}
        while not self.stop_event.is_set():
            worked=False
            for cid,capture in list(self.cameras.items()):
                now=time.monotonic()
                if now<shown.get(cid,0):continue
                frame=capture.view.take()
                if frame is None or frame.session_id!=capture.session_id:continue
                worked=True
                try:
                    with self.lock:zones=list(self.zones[cid])
                    analysis=self.analysis.get(cid);ttl=self.config["runtime"]["observation_ttl_s"]
                    tracks,ppe,note,warning,age=overlay_for(frame,analysis,ttl)
                    jpeg=render(frame,tracks,ppe,zones,self.config["runtime"]["preview_width"],self.detector.ready,note,warning,fast=True)
                    self.preview_expires[cid]=min(frame.received_monotonic+ttl,analysis.observed_at+ttl) if age is not None else frame.received_monotonic+ttl
                    self.latest[cid]=(frame,jpeg);self.metrics[cid].preview.append(time.monotonic())
                    # Pace from the previous slot, not from `now`: adding the interval
                    # to the moment of each take let poll and render time accumulate
                    # until a source frame was skipped (25 fps in, about 20 shown).
                    shown[cid]=max(now,shown.get(cid,now)+1/self.config["runtime"]["preview_fps"])
                    self.metrics[cid].latency.append((time.monotonic()-frame.received_monotonic)*1000)
                    if age is not None:self.metrics[cid].overlay_age.append(age*1000)
                except Exception as error:self.error="live_preview_"+type(error).__name__
            if not worked:self.stop_event.wait(.004)

    def preview(self,camera_id):
        item=self.latest.get(camera_id);capture=self.cameras[camera_id]
        if item:
            frame,jpeg=item
            now=time.monotonic()
            if (frame.session_id==capture.session_id and capture.status=="connected"
                and now-frame.received_monotonic<self.config["runtime"]["observation_ttl_s"]
                and (not self.live_preview or now<self.preview_expires.get(camera_id,float('inf')))):
                return frame.key,jpeg
        # The placeholder changes once a second; viewers poll far more often, so
        # build it once per key instead of on every poll. One entry per camera.
        key=(camera_id,capture.status,int(time.monotonic()));cached=self.placeholders.get(camera_id)
        if cached and cached[0]==key:return cached
        image=np.zeros((360,640,3),np.uint8);image[:]=(25,21,17)
        cv2.putText(image,camera_id+" / "+capture.status.upper(),(30,170),cv2.FONT_HERSHEY_SIMPLEX,.8,(160,175,190),1,cv2.LINE_AA)
        cv2.putText(image,"NO CURRENT OBSERVATION",(30,210),cv2.FONT_HERSHEY_SIMPLEX,.5,(100,120,140),1)
        self.placeholders[camera_id]=(key,cv2.imencode(".jpg",image)[1].tobytes())
        return self.placeholders[camera_id]
