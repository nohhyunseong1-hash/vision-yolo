"""One isolated CUDA training job with immutable inputs and live provenance."""
import argparse
import csv
import importlib.metadata
import math
import os
import shutil
import time
from pathlib import Path

from app.common import ROOT, read_json, sha256, utc_now, write_json


def verify_dataset(spec):
    import yaml
    data = ROOT / spec['data']
    if sha256(data) != spec['data_sha256']:
        raise ValueError('dataset_yaml_changed')
    config = yaml.safe_load(data.read_text(encoding='utf-8'))
    if 'download' in config:
        raise ValueError('dataset_download_forbidden')
    all_rows = {}
    for split, expected in spec['manifests'].items():
        path = ROOT / expected['path']
        if sha256(path) != expected['sha256']:
            raise ValueError('manifest_changed:' + split)
        rows = read_json(path)['records']
        all_rows[split] = rows
        for row in rows:
            if row.get('user_video_used') or row.get('user_video_used_for_training'):
                raise ValueError('user_video_training_forbidden')
            for key, digest in (('image','image_sha256'),('label','derived_label_sha256')):
                actual = (ROOT / row[key]).resolve()
                if not actual.is_relative_to((ROOT/'data/derived').resolve()):
                    raise ValueError('non_project_derived_input')
                if sha256(actual) != row[digest]:
                    raise ValueError('dataset_file_changed:' + row[key])
    for first, second in (('train','validation'),('train','test'),('validation','test')):
        for key in ('image_sha256','group'):
            if {r[key] for r in all_rows[first]} & {r[key] for r in all_rows[second]}:
                raise ValueError('split_leakage:' + key)
    for split, yaml_key in (('train','train'),('validation','val')):
        list_path = Path(config['path']) / config[yaml_key]
        if sha256(list_path) != spec['list_sha256'][split]:
            raise ValueError('training_list_changed')
        allowed = {(ROOT/r['image']).resolve() for r in all_rows[split]}
        selected = [Path(line).resolve() for line in list_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not selected or len(selected) != len(set(selected)) or not set(selected) <= allowed:
            raise ValueError('training_list_outside_approved_split')
    return config


def numeric_metrics(values):
    result = {}
    for key, value in values.items():
        value = float(value)
        if not math.isfinite(value):
            raise FloatingPointError('nonfinite_metric:' + key)
        result[key] = value
    return result


def recorded_epoch(path):
    # Ultralytics invokes on_fit_epoch_end again during final validation after
    # incrementing trainer.epoch. The saved CSV is the completed-epoch source.
    with Path(path).open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    return int(float(rows[-1]['epoch'])) if rows else 0


def run(job_path, resume=False):
    os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'.cache/ultralytics')
    os.environ['YOLO_AUTOINSTALL'] = 'False'
    os.environ['YOLO_OFFLINE'] = 'True'
    os.environ['WANDB_MODE'] = 'disabled'
    import torch
    from ultralytics import YOLO
    from ultralytics import settings
    settings.update({'sync':False,'wandb':False,'mlflow':False,'clearml':False,'comet':False,'dvc':False})
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('cuda_required_no_cpu_fallback')
    spec=read_json(job_path);verify_dataset(spec)
    initial=ROOT/spec['initial']
    if sha256(initial)!=spec['initial_sha256']:
        raise ValueError('initial_weight_changed')
    output=ROOT/spec['output'];provenance_path=output/'provenance.json'
    job_hash=sha256(job_path)
    if output.exists():
        prior=read_json(provenance_path,{})
        if not resume or prior.get('job_sha256')!=job_hash or prior.get('status')=='completed':
            raise FileExistsError('existing_or_incompatible_job:' + str(output))
        if not any((output/'weights'/name).exists() for name in ('resume.pt','last.pt')):
            raise FileNotFoundError('no_checkpoint_to_resume')
    else:
        if resume:raise FileNotFoundError('resume_run_missing')
        output.mkdir(parents=True)
    params=dict(data=str(ROOT/spec['data']),epochs=spec['epochs'],imgsz=spec['imgsz'],batch=spec['batch'],device=0,
        workers=0,nbs=64,seed=spec['seed'],deterministic=True,optimizer='AdamW',lr0=spec.get('lr0',.0003),lrf=.05,
        warmup_bias_lr=spec.get('lr0',.0003),warmup_epochs=min(3,spec['epochs']-1),weight_decay=.0005,
        cos_lr=True,freeze=0,nms=False,patience=spec['patience'],project=str(output.parent),name=output.name,
        exist_ok=True,cache=False,plots=True,amp=True,mosaic=.15,mixup=0.,scale=.3,translate=.08,degrees=5.,
        fliplr=.5,close_mosaic=min(10,max(0,spec['epochs']//4)),max_det=300,save=True,save_period=10,verbose=False)
    started=time.monotonic();last_heartbeat=[0.];freeze_state=[None];stop_reason=[None]
    provenance=dict(job_id=spec['id'],job_sha256=job_hash,status='initializing',started_at=utc_now(),resume=resume,
        source_commit=spec['source_commit'],source_hashes=spec['source_hashes'],parameters=params,
        initial_weights_source=spec['initial_url'],initial_weights_sha256=spec['initial_sha256'],
        dataset_audit_sha256=spec['dataset_audit_sha256'],manifests=spec['manifests'],smoke=spec['phase']=='smoke',
        packages={name:importlib.metadata.version(name) for name in ('torch','torchvision','ultralytics','numpy')},
        gpu=torch.cuda.get_device_name(0),actual_device='cuda:0',user_video_used_for_training=False,
        field_acceptance=False,automatic_deployment=False,checkpoint_selection='fixed development validation only')
    write_json(provenance_path,provenance)
    def heartbeat(trainer,force=False):
        now=time.monotonic()
        if not force and now-last_heartbeat[0]<10:return
        last_heartbeat[0]=now
        epoch=recorded_epoch(output/'results.csv') if force else min(trainer.epoch+1,spec['epochs'])
        update=dict(status='running',updated_at=utc_now(),epoch=epoch,epochs=spec['epochs'],
            elapsed_seconds=round(now-started,2),actual_batch=trainer.batch_size,
            gradient_accumulation=trainer.accumulate,
            peak_allocated_gib=round(torch.cuda.max_memory_allocated()/2**30,3),
            peak_reserved_gib=round(torch.cuda.max_memory_reserved()/2**30,3),
            trainable_parameters=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
            early_layers_frozen=freeze_state[0],stop_reason=stop_reason[0])
        if trainer.tloss is not None:
            update['losses']=numeric_metrics(trainer.tloss)
        provenance.update(update);write_json(provenance_path,provenance)
    def batch_start(trainer):
        freeze=trainer.epoch<spec.get('freeze_epochs',3)
        if freeze_state[0]!=freeze:
            # The optimizer was built with every learnable parameter. Temporarily
            # freezing here preserves their optimizer groups for later unfreeze.
            for layer in trainer.model.model[:6]:
                for parameter in layer.parameters():parameter.requires_grad_(not freeze)
            freeze_state[0]=freeze
        if freeze:
            for layer in trainer.model.model[:6]:layer.eval()
    def batch_end(trainer):
        if trainer.loss is not None and not torch.isfinite(trainer.loss.detach()).all():
            raise FloatingPointError('nonfinite_training_loss')
        if Path(spec['stop_file']).exists():stop_reason[0]='requested_stop'
        if time.time()>=spec['deadline_timestamp']:stop_reason[0]='campaign_72_hour_budget'
        if torch.cuda.max_memory_allocated()/2**30>spec.get('memory_limit_gib',13.5):
            stop_reason[0]='memory_budget_exceeded'
        if stop_reason[0]:trainer.stop=True
        heartbeat(trainer)
    def epoch_end(trainer):
        provenance['validation_metrics']=numeric_metrics(trainer.metrics)
        provenance['completed_epoch']=recorded_epoch(output/'results.csv')
        heartbeat(trainer,True)
        print('CAMPAIGN_EPOCH',spec['id'],provenance['completed_epoch'],provenance['validation_metrics'],flush=True)
    def checkpoint(trainer):
        if stop_reason[0] and trainer.last.exists():
            # final_eval strips optimizer state from last.pt. Keep a resumable
            # checkpoint before that happens on an explicitly stopped job.
            shutil.copyfile(trainer.last,output/'weights/resume.pt')
    resume_path=output/'weights/resume.pt'
    if not resume_path.exists():resume_path=output/'weights/last.pt'
    model=YOLO(str(resume_path) if resume else str(initial))
    model.add_callback('on_train_batch_start',batch_start)
    model.add_callback('on_train_batch_end',batch_end)
    model.add_callback('on_fit_epoch_end',epoch_end)
    model.add_callback('on_model_save',checkpoint)
    try:
        metrics=model.train(resume=True) if resume else model.train(**params)
        best=output/'weights/best.pt'
        if not best.exists():raise RuntimeError('missing_best_checkpoint')
        provenance.update(status='stopped' if stop_reason[0] else 'completed',finished_at=utc_now(),
            epoch=recorded_epoch(output/'results.csv'),completed_epoch=recorded_epoch(output/'results.csv'),
            elapsed_seconds=round(time.monotonic()-started,2),stop_reason=stop_reason[0],
            checkpoint=str(best.relative_to(ROOT)),checkpoint_sha256=sha256(best),
            best_validation_metrics=numeric_metrics(metrics.results_dict),
            validation_speed_ms=numeric_metrics(metrics.speed),
            class_precision=[float(x) for x in metrics.box.p],class_recall=[float(x) for x in metrics.box.r],
            class_ap50=[float(x) for x in metrics.box.ap50],class_ap=[float(x) for x in metrics.box.ap])
    except BaseException as error:
        provenance.update(status='failed',finished_at=utc_now(),error_type=type(error).__name__,error=str(error),
            elapsed_seconds=round(time.monotonic()-started,2),resumable_checkpoint=(output/'weights/last.pt').exists())
        raise
    finally:write_json(provenance_path,provenance)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();run(args.job,args.resume)
