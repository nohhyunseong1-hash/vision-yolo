"""Sequential, bounded local training queue. Never registers/deploys candidates."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import yaml

from app.common import ROOT, read_json, sha256, utc_now, write_json

SOURCE_FILES=('scripts/training_campaign.py','scripts/train_campaign_job.py','scripts/prepare_training_campaign.py')


def select_candidate(candidates):
    """Validation-only architecture selection, followed by separate full training.

    Within one percentage point AP50-95 and three points minimum-class recall
    of the leading candidate, prefer smaller measured validation inference cost.
    These local timings are not target laptop throughput acceptance.
    """
    if not candidates:raise ValueError('no_candidates')
    for row in candidates:
        if row.get('status')!='completed' or not row.get('class_recall'):
            raise ValueError('incomplete_candidate')
        numbers=[row['best_validation_metrics']['metrics/mAP50-95(B)'],*row['class_recall'],row['validation_speed_ms']['inference']]
        if not all(math.isfinite(x) for x in numbers):raise ValueError('nonfinite_candidate')
    best=max(candidates,key=lambda r:r['best_validation_metrics']['metrics/mAP50-95(B)'])
    eligible=[r for r in candidates if r['best_validation_metrics']['metrics/mAP50-95(B)']>=best['best_validation_metrics']['metrics/mAP50-95(B)']-.01
              and min(r['class_recall'])>=min(best['class_recall'])-.03]
    return min(eligible,key=lambda r:r['validation_speed_ms']['inference'])


def initialize(campaign):
    folder=ROOT/'outputs/training'/campaign
    if folder.exists():raise FileExistsError(folder)
    dataset=ROOT/'data/derived'/campaign;audit=read_json(dataset/'audit.json')
    if audit['user_video_used_for_training']:raise ValueError('user_video_forbidden')
    folder.mkdir(parents=True);(folder/'jobs').mkdir();(folder/'logs').mkdir()
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    sources={name:sha256(ROOT/name) for name in SOURCE_FILES}
    data_specs={}
    for task in ('person','head','vest'):
        task_folder=dataset/task
        # Cap a source component in the selection validation only. Preserve full
        # manifests/lists for later complete evaluation, including locked tests.
        rows=read_json(task_folder/'validation.json')['records'];groups={}
        for row in rows:groups.setdefault(row['group'],[]).append(row)
        selected=[]
        for key,items in sorted(groups.items()):selected.extend(sorted(items,key=lambda r:r['image_sha256'])[:180])
        (task_folder/'selection-validation.txt').write_text(''.join(str(ROOT/r['image'])+'\n' for r in selected),encoding='utf-8')
        spec=yaml.safe_load((task_folder/'data.yaml').read_text(encoding='utf-8'));spec['val']='selection-validation.txt'
        (task_folder/'selection.yaml').write_text(yaml.safe_dump(spec,allow_unicode=True),encoding='utf-8')
        data_specs[task]={'full_validation_images':len(rows),'selection_validation_images':len(selected),
            'manifests':{split:{'path':path,'sha256':sha256(ROOT/path)} for split,path in audit['manifest_paths'][task].items()}}
    jobs=[]
    combinations=[('head','s',640,12),('vest','s',640,12),('person','s',1280,4)]
    for task,size,imgsz,batch in combinations:
        jobs.append(dict(id=f'smoke-{task}-{size}-{imgsz}',phase='smoke',task=task,size=size,imgsz=imgsz,batch=batch,epochs=3,patience=3,seed=101,freeze_epochs=0))
    for task,combos in [('head',[('n',640,16),('s',640,12)]),('vest',[('n',640,16),('s',640,12)]),('person',[('n',960,12),('s',960,8),('n',1280,6),('s',1280,4)])]:
        for size,imgsz,batch in combos:
            jobs.append(dict(id=f'pilot-{task}-{size}-{imgsz}',phase='pilot',task=task,size=size,imgsz=imgsz,batch=batch,epochs=15,patience=15,seed=101,freeze_epochs=3))
    config={'campaign':campaign,'created_at':utc_now(),'source_commit':head,'source_hashes':sources,'dataset_audit_sha256':sha256(dataset/'audit.json'),
        'dataset_specs':data_specs,'jobs':jobs,'main_epochs':150,'main_patience':25,'main_seeds':[101,202,303],
        'budget_hours':72,'automatic_main_training':True,'automatic_deployment':False,'user_video_used_for_training':False,
        'selection_scope':'Balanced existing labeled development validation, not independent field performance. Full test/video/latency acceptance follows training.'}
    write_json(folder/'config.json',config)
    write_json(folder/'status.json',{'status':'prepared','updated_at':utc_now(),'queued_jobs':len(jobs),'automatic_main_training':True})
    print(json.dumps({'campaign':campaign,'queued_jobs':len(jobs),'validation':data_specs},ensure_ascii=True))


def acquire_pipeline_lock():
    folder=ROOT/'.runtime';folder.mkdir(exist_ok=True)
    handle=open(folder/'pipeline.lock','a+b');handle.write(b'1');handle.flush();handle.seek(0)
    if os.name=='nt':
        import msvcrt
        msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    return handle


def materialize_job(config,job,folder,deadline):
    data=ROOT/'data/derived'/config['campaign']/job['task']/('smoke.yaml' if job['phase']=='smoke' else 'selection.yaml')
    data_yaml=yaml.safe_load(data.read_text(encoding='utf-8'))
    initial=ROOT/f'models/yolo26{job["size"]}.pt';source=read_json(initial.with_suffix('.pt.source.json'))
    if sha256(initial)!=source['sha256']:raise ValueError('unverified_initial_checkpoint')
    return {**job,'data':str(data.relative_to(ROOT)),'data_sha256':sha256(data),
        'list_sha256':{s:sha256(Path(data_yaml['path'])/data_yaml[k]) for s,k in (('train','train'),('validation','val'))},
        'manifests':config['dataset_specs'][job['task']]['manifests'],'initial':str(initial.relative_to(ROOT)),
        'initial_sha256':source['sha256'],'initial_url':source['url'],'output':str((folder/'runs'/job['id']).relative_to(ROOT)),
        'source_commit':config['source_commit'],'source_hashes':config['source_hashes'],'dataset_audit_sha256':config['dataset_audit_sha256'],
        'stop_file':str(folder/'STOP'),'deadline_timestamp':deadline,'memory_limit_gib':13.5}


def execute(campaign,resume=False):
    folder=ROOT/'outputs/training'/campaign;config=read_json(folder/'config.json')
    for name,digest in config['source_hashes'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('campaign_source_changed:'+name)
    if sha256(ROOT/'data/derived'/campaign/'audit.json')!=config['dataset_audit_sha256']:
        raise ValueError('campaign_audit_changed')
    lock=acquire_pipeline_lock()
    state=read_json(folder/'status.json',{})
    if state.get('status') not in ('prepared',) and not resume:raise ValueError('explicit_resume_required')
    if state.get('config_sha256') and state['config_sha256']!=sha256(folder/'config.json'):
        raise ValueError('campaign_config_changed')
    started=state.get('started_timestamp',time.time());deadline=started+config['budget_hours']*3600
    state.update(status='running',started_timestamp=started,started_at=state.get('started_at',utc_now()),
        runner_pid=os.getpid(),runner_create_time=psutil.Process().create_time(),updated_at=utc_now(),completed_jobs=state.get('completed_jobs',[]),
        user_video_used_for_training=False,automatic_deployment=False,config_sha256=sha256(folder/'config.json'))
    jobs=list(config['jobs']);child=None
    def save():state['updated_at']=utc_now();write_json(folder/'status.json',state)
    try:
        cursor=0
        while True:
            if cursor==len(jobs):
                if any(j['phase']=='main' for j in jobs):
                    state.update(status='training_completed_pending_evaluation',active_job=None,child_pid=None);save();break
                selections=[]
                for task in ('person','head','vest'):
                    candidates=[read_json(folder/'runs'/j['id']/'provenance.json') for j in jobs if j['phase']=='pilot' and j['task']==task]
                    winner=select_candidate(candidates);template=next(j for j in jobs if j['id']==winner['job_id'])
                    selections.append({'task':task,'winner':winner['job_id'],'validation_metrics':winner['best_validation_metrics'],
                        'class_recall':winner['class_recall'],'validation_speed_ms':winner['validation_speed_ms'],
                        'deployment_approved':False,'field_acceptance':False})
                    for seed in config['main_seeds']:
                        jobs.append({**template,'id':f'main-{task}-{template["size"]}-{template["imgsz"]}-seed{seed}',
                            'phase':'main','seed':seed,'epochs':config['main_epochs'],'patience':config['main_patience']})
                write_json(folder/'selection.json',{'selected_at':utc_now(),'selections':selections,'policy':'validation AP50-95 with 0.01 AP / 0.03 minimum-class recall equivalence window, then lowest measured validation inference time; fresh official initialization for each seed','deployment_approved':False})
            job=jobs[cursor];cursor+=1
            if job['id'] in state['completed_jobs']:continue
            if (folder/'STOP').exists() or time.time()>=deadline:
                state.update(status='stopped',reason='stop_file_or_budget',active_job=None,child_pid=None);save();break
            job_path=folder/'jobs'/(job['id']+'.json');job_output=folder/'runs'/job['id']
            spec=materialize_job(config,job,folder,deadline)
            if job_path.exists():
                if read_json(job_path)!=spec:raise ValueError('existing_job_spec_changed')
            else:write_json(job_path,spec)
            command=[sys.executable,'-u','-m','scripts.train_campaign_job','--job',str(job_path)]
            existing=read_json(job_output/'provenance.json',{})
            if existing.get('status')=='completed':
                state['completed_jobs'].append(job['id']);save();continue
            if job_output.exists():
                if not resume:raise ValueError('existing_job_requires_resume')
                command.append('--resume')
            state.update(active_job=job['id'],phase=job['phase'],queued_remaining=len(jobs)-cursor,child_pid=None);save()
            with (folder/'logs'/(job['id']+'.log')).open('a',encoding='utf-8') as log:
                env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONUTF8':'1','YOLO_CONFIG_DIR':str(ROOT/'.cache/ultralytics'),'YOLO_AUTOINSTALL':'False','YOLO_OFFLINE':'True'}
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                state['child_pid']=child.pid;save()
                while child.poll() is None:
                    time.sleep(5)
                    progress=read_json(job_output/'provenance.json',{})
                    state['progress']={k:progress.get(k) for k in ('status','epoch','completed_epoch','epochs','elapsed_seconds','actual_batch','peak_allocated_gib','peak_reserved_gib','validation_metrics','stop_reason')}
                    save()
                code=child.returncode;child=None
            result=read_json(job_output/'provenance.json',{})
            if code or result.get('status')!='completed':
                state.update(status='stopped' if result.get('status')=='stopped' else 'failed',exit_code=code,
                    reason=result.get('error',result.get('stop_reason','child_failed_before_provenance')),child_pid=None);save();break
            state['completed_jobs'].append(job['id']);state['child_pid']=None;save()
    except BaseException as error:
        state.update(status='failed',reason=type(error).__name__+': '+str(error));save()
        if child is not None and child.poll() is None:
            # Signal the child to checkpoint/stop; do not kill unrelated processes.
            (folder/'STOP').write_text('runner exception',encoding='utf-8')
            child.wait()
        raise
    finally:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['init','run','status','stop']);parser.add_argument('--campaign',required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();folder=ROOT/'outputs/training'/args.campaign
    if args.action=='init':initialize(args.campaign)
    elif args.action=='run':execute(args.campaign,args.resume)
    elif args.action=='stop':(folder/'STOP').write_text('user requested graceful training stop',encoding='utf-8');print('Stop requested; current job will checkpoint at its next safe boundary.')
    else:print(json.dumps(read_json(folder/'status.json',{}),ensure_ascii=False,indent=2))
