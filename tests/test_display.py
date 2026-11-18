import threading
import time
import av
import numpy as np
import pytest
from app.capture import Capture
from app.runtime import AnalysisSnapshot,near_person,overlay_for,render
from app.types import Detection


def test_display_hides_ppe_boxes_that_have_no_wearer(frame_factory):
    frame=frame_factory(width=640,height=480);person=Detection('person',(100.,50.,200.,400.),.9)
    worn=Detection('helmet',(120.,50.,180.,100.),.9);stray=Detection('safety_vest',(400.,200.,500.,300.),.9)
    assert near_person(worn.xyxy,person.xyxy) and not near_person(stray.xyxy,person.xyxy)
    def picture(*tracks):return render(frame,list(tracks),{},[],640,True)
    assert picture(person,stray)==picture(person)
    assert picture(person,worn)!=picture(person)
    # The detections themselves are untouched; only the picture changes.
    assert stray.observed and stray.xyxy==(400.,200.,500.,300.)


def test_overlay_draws_only_fresh_analysis_from_the_same_session(frame_factory):
    shown=frame_factory(10.,frame_id=8);person=Detection('person',(100.,50.,200.,400.),.9,1);state={'k':{}}
    def snapshot(session='session1',frame_id=5,t=9.88,size=(640,480)):
        return AnalysisSnapshot(session,frame_id,t,size,[person],state)
    tracks,ppe,note,warning,age=overlay_for(shown,snapshot(),1.,now=10.)
    assert tracks==[person] and ppe is state and warning is None
    assert 'frame 5' in note and '120 ms' in note and age==pytest.approx(.12)
    # Never label future boxes as "0 ms earlier", or reuse original pixels on
    # a different source resolution, even if the capture session is unchanged.
    for stale in (None,snapshot(session='session2'),snapshot(t=8.9),snapshot(frame_id=9,t=10.02),snapshot(size=(1280,720))):
        assert overlay_for(shown,stale,1.,now=10.)==([],{},None,'NO CURRENT ANALYSIS',None)
    # Display/analysis times can be close while both waited in the pipeline.
    assert overlay_for(shown,snapshot(),1.,now=10.9)==([],{},None,'NO CURRENT ANALYSIS',None)


def test_stamp_note_and_warning_are_visible_and_never_mask_model_state(frame_factory):
    frame=frame_factory(width=640,height=480)
    plain=render(frame,[],{},[],640,True)
    assert render(frame,[],{},[],640,True,note='boxes: frame 5, 120 ms earlier')!=plain
    assert render(frame,[],{},[],640,True,warning='NO CURRENT ANALYSIS')!=plain
    assert render(frame,[],{},[],640,False,warning='NO CURRENT ANALYSIS')==render(frame,[],{},[],640,False)


def test_capture_offers_each_frame_to_analysis_and_display(tmp_path):
    path=tmp_path/'sample.mp4'
    with av.open(str(path),'w') as out:
        stream=out.add_stream('libx264',rate=25);stream.width=160;stream.height=120;stream.pix_fmt='yuv420p'
        for i in range(25):
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.full((120,160,3),i*8,np.uint8),format='bgr24')):out.mux(packet)
        for packet in stream.encode():out.mux(packet)
    capture=Capture(dict(id='file',source_type='file',path=str(path),loop=True));capture.start()
    try:
        seen={'slot':None,'view':None};end=time.monotonic()+4
        while time.monotonic()<end and None in seen.values():
            for name in seen:seen[name]=seen[name] or getattr(capture,name).take()
            time.sleep(.02)
        assert seen['slot'] is not None and seen['view'] is not None
        assert seen['view'].image.shape==(120,160,3) and seen['view'].session_id==capture.session_id
    finally:capture.stop()
    assert capture.view.take() is None and capture.slot.take() is None


def run_loops(runtime,until):
    threads=[threading.Thread(target=loop,daemon=True) for loop in (runtime.render_loop,runtime.view_loop)]
    for thread in threads:thread.start()
    end=time.monotonic()+2
    while time.monotonic()<end and not until():time.sleep(.01)
    runtime.stop_event.set()
    for thread in threads:thread.join(timeout=2)


def make_runtime(tmp_path,live):
    from app.config import load_config
    from app.runtime import Runtime
    config=load_config();config['runtime']={**config['runtime'],'live_preview':live,'preview_fps':50}
    runtime=Runtime(config,tmp_path);runtime.detector.ready=True
    camera=next(iter(runtime.cameras));capture=runtime.cameras[camera]
    capture.session_id='session1';capture.status='connected'
    return runtime,camera,capture


def test_live_preview_shows_unanalysed_frames_and_keeps_evidence_on_analysed_ones(tmp_path,frame_factory):
    runtime,camera,capture=make_runtime(tmp_path,True);now=time.monotonic()
    analysed=frame_factory(now-.1,camera=camera,frame_id=5);shown=frame_factory(now,camera=camera,frame_id=8)
    person=Detection('person',(100.,50.,200.,400.),.9,1)
    runtime.analysis[camera]=AnalysisSnapshot(analysed.session_id,5,analysed.received_monotonic,(640,480),[person],{})
    runtime.render_slots[camera].put((analysed,[person],{},[],[]));capture.view.put(shown)
    run_loops(runtime,lambda:camera in runtime.latest and runtime.store.buffers)
    assert runtime.latest[camera][0].frame_id==8 and runtime.preview(camera)[0]==shown.key
    assert runtime.metrics[camera].snapshot()['overlay_age_ms']['p50']==pytest.approx(100,abs=1)
    # The pre-event clip buffer received the analysed frame only.
    assert [stamp for stamp,_ in runtime.store.buffers[(camera,'session1')]]==[analysed.received_monotonic]


def test_default_preview_still_shows_only_analysed_frames(tmp_path,frame_factory):
    runtime,camera,capture=make_runtime(tmp_path,False);now=time.monotonic()
    analysed=frame_factory(now-.1,camera=camera,frame_id=5);capture.view.put(frame_factory(now,camera=camera,frame_id=8))
    runtime.render_slots[camera].put((analysed,[],{},[],[]))
    thread=threading.Thread(target=runtime.render_loop,daemon=True);thread.start();end=time.monotonic()+2
    while time.monotonic()<end and camera not in runtime.latest:time.sleep(.01)
    runtime.stop_event.set();thread.join(timeout=2)
    assert runtime.latest[camera][0].frame_id==5
    assert runtime.metrics[camera].snapshot()['overlay_age_ms'] is None
    # Without live_preview the display thread is never started.
    assert not runtime.live_preview and not runtime.viewer.is_alive()


def test_cached_preview_expires_when_its_analysis_expires(tmp_path,frame_factory):
    runtime,camera,capture=make_runtime(tmp_path,True);now=time.monotonic()
    shown=frame_factory(now-.1,camera=camera,frame_id=8)
    runtime.latest[camera]=(shown,b'cached-prediction-picture')
    runtime.preview_expires[camera]=now-.01
    key,jpeg=runtime.preview(camera)
    assert key!=shown.key and jpeg!=b'cached-prediction-picture'


def test_decoder_errors_do_not_print_stream_credentials():
    import subprocess,sys
    code="""
import av
from app.capture import quiet_decoder_logs
quiet_decoder_logs()
assert av.logging.get_level() is None
try:
    av.open('rtsp://sample-user:sample-secret@127.0.0.1:1/unavailable',timeout=(.2,.2))
except av.error.FFmpegError:
    pass
else:
    raise AssertionError('Expected an unavailable local stream')
"""
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,timeout=5)
    assert result.returncode==0 and result.stdout==result.stderr==b''
