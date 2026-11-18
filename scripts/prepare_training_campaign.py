"""Prepare immutable, labeled-only development data for the v9 campaign.

Original archives are read-only. User videos are exclusion references only.
The older PPE archives are exclusion references, not silently pooled despite
their sampled missing annotations and inconsistent head/vest definitions.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import av
import numpy as np
import yaml
from PIL import Image

from app.common import ROOT, read_json, sha256, utc_now, write_json
from app.orientation import decode_bgr
from scripts.prepare_archive_states import dhash
from scripts.prepare_data import UnionFind


def distances(values, value):
    lookup = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    return lookup[(values ^ np.uint64(value)).view(np.uint8).reshape(-1, 8)].sum(axis=1)


def prepare(campaign):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', campaign):
        raise ValueError('invalid_campaign_id')
    target = ROOT / 'data/derived' / campaign
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    inventory = list(csv.DictReader((ROOT/'data/manifests/dataset_inventory.csv').open(encoding='utf-8-sig')))
    prior_sha = {r['sha256'] for r in inventory if r.get('sha256')}
    prior_dhash = {int(r['dhash'], 16) for r in inventory if r.get('dhash')}
    for kind in ('head', 'vest'):
        for split in ('train', 'validation', 'test'):
            rows = read_json(ROOT/f'data/manifests/{kind}-state-yolo26s-v8/{split}.json')['records']
            prior_sha.update(r['image_sha256'] for r in rows)
            prior_dhash.update(int(r['dhash'], 16) for r in rows)
    prior_dhash = np.array(sorted(prior_dhash), dtype=np.uint64)
    video_dhash = set()
    video_references = []
    for name in ('크레인 재해.mp4', '크레인 점검.mp4'):
        path = ROOT.parent.parent / name
        video_references.append({'name': name, 'sha256': sha256(path)})
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                bgr, _ = decode_bgr(frame)
                video_dhash.add(dhash(Image.fromarray(bgr[:, :, ::-1])))
    video_dhash = np.array(sorted(video_dhash), dtype=np.uint64)
    source = Path('D:/vigent_private_data/public_ppe/roboflow/ppe_construction_v2.zip')
    source_hash = sha256(source)
    excluded = Counter()
    candidates = []
    with zipfile.ZipFile(source) as archive:
        documentation = {n: archive.read(n).decode('utf-8') for n in archive.namelist() if n.startswith('README')}
        assert 'License: CC BY 4.0' in documentation['README.dataset.txt']
        for member in sorted(n for n in archive.namelist() if n.endswith('_annotations.coco.json')):
            data = json.loads(archive.read(member))
            categories = {c['id']: c['name'] for c in data['categories']}
            assert categories == {0:'construction-safety-persona-KtTw',1:'helmet',2:'no_helmet',3:'no_vest',4:'person',5:'vest'}
            annotation_path = target / (member.split('/')[0] + '-source-coco.json')
            annotation_path.write_bytes(archive.read(member))
            annotations = defaultdict(list)
            for annotation in data['annotations']:
                annotations[annotation['image_id']].append(annotation)
            for row in data['images']:
                payload = archive.read(member.split('/')[0] + '/' + row['file_name'])
                digest = hashlib.sha256(payload).hexdigest()
                im = Image.open(BytesIO(payload)).convert('RGB')
                w, h = im.size
                assert (w, h) == (row['width'], row['height'])
                fingerprint = dhash(im)
                if digest in prior_sha or np.any(distances(prior_dhash, fingerprint) <= 4):
                    excluded['duplicate_of_prior_dataset_any_split'] += 1
                    continue
                if np.any(distances(video_dhash, fingerprint) <= 8):
                    excluded['user_video_exclusion_reference'] += 1
                    continue
                objects = []
                invalid = False
                for ann in annotations[row['id']]:
                    x, y, bw, bh = map(float, ann['bbox'])
                    if ann['category_id'] not in range(1, 6) or not np.isfinite([x,y,bw,bh]).all() or bw<=0 or bh<=0 or x<-.5 or y<-.5 or x+bw>w+.5 or y+bh>h+.5:
                        invalid = True
                        break
                    x1,y1,x2,y2=max(0,x),max(0,y),min(w,x+bw),min(h,y+bh)
                    objects.append([ann['category_id'],(x1+x2)/(2*w),(y1+y2)/(2*h),(x2-x1)/w,(y2-y1)/h])
                if invalid or not objects:
                    excluded['invalid_or_empty_annotation'] += 1
                    continue
                # Require a person annotation for state training. Objects such as
                # product-only vests must not define the wearing-state domain.
                if not any(o[0] == 4 for o in objects):
                    excluded['no_person_annotation_product_or_incomplete'] += 1
                    continue
                candidates.append(dict(image_sha256=digest,dhash=f'{fingerprint:016x}',objects=objects,
                    member=member.split('/')[0]+'/'+row['file_name'],stem=Path(row['file_name']).stem.split('.rf.')[0],
                    original_split={'train':'train','valid':'validation','test':'test'}[member.split('/')[0]],
                    source_annotation=str(annotation_path.relative_to(ROOT)),width=w,height=h,
                    source_archive_sha256=source_hash,user_video_used=False))
        print('archive candidates after prior/video exclusions:',len(candidates),dict(excluded),flush=True)
        union = UnionFind(len(candidates));owners={}
        hashes = np.array([int(r['dhash'],16) for r in candidates],dtype=np.uint64)
        for i,r in enumerate(candidates):
            for key in (('sha',r['image_sha256']),('stem',r['stem'])):
                if key in owners:union.union(i,owners[key])
                else:owners[key]=i
            for offset in np.flatnonzero(distances(hashes[i+1:],hashes[i]) <= 6):
                union.union(i,i+1+int(offset))
        groups=defaultdict(list)
        for i,r in enumerate(candidates):groups[union.find(i)].append(r)
        kept=[]
        for rows in groups.values():
            if len({r['original_split'] for r in rows}) > 1:
                excluded['cross_split_near_duplicate_component'] += len(rows)
                continue
            group=hashlib.sha256('\n'.join(sorted(r['image_sha256'] for r in rows)).encode()).hexdigest()[:24]
            # Fixed additional development holdout, chosen before any model run.
            split=rows[0]['original_split']
            if split=='train' and int(group[:8],16)%10==0:split='validation'
            seen=set()
            for r in sorted(rows,key=lambda r:r['image_sha256']):
                if r['image_sha256'] in seen:excluded['within_split_exact_duplicate']+=1;continue
                seen.add(r['image_sha256']);kept.append({**r,'group':group,'split':split})
        manifests={};statistics={}
        task_specs={'person':({4:0},['person']),'head':({1:0,2:1},['helmet','bare_head']),
                    'vest':({5:0,3:1},['safety_vest','bare_torso'])}
        for task,(mapping,names) in task_specs.items():
            folder=target/task;folder.mkdir()
            records={s:[] for s in ('train','validation','test')}
            for r in kept:
                mapped=[[mapping[o[0]],*o[1:]] for o in r['objects'] if o[0] in mapping]
                if not mapped:continue
                split=r['split'];images=folder/split/'images';labels=folder/split/'labels'
                images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
                image=images/(r['image_sha256'][:24]+'.jpg');label=labels/(image.stem+'.txt')
                image.write_bytes(archive.read(r['member']))
                label.write_text(''.join(str(o[0])+' '+' '.join(f'{v:.8f}' for v in o[1:])+'\n' for o in mapped),encoding='utf-8')
                records[split].append({**r,'image':str(image.relative_to(ROOT)),'label':str(label.relative_to(ROOT)),
                    'derived_label_sha256':sha256(label),'objects':mapped,'license':'CC BY 4.0; attribution in campaign source.json'})
            if task=='person':
                # Retain the established, source-balanced people domain; no new
                # pseudo labels and no old validation/test samples in train.
                for split in records:
                    old=read_json(ROOT/f'data/manifests/ppe-yolo26s-balanced-v3/{split}.json')['records']
                    for r in old:
                        original=ROOT/r['image'];source_label=ROOT/r['label']
                        assert sha256(original)==r['image_sha256'] and sha256(source_label)==r['derived_label_sha256']
                        images=folder/split/'images';labels=folder/split/'labels';images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
                        image=images/('prior-'+original.name);label=labels/(image.stem+'.txt');shutil.copyfile(original,image)
                        lines=[line for line in source_label.read_text().splitlines() if line.split()[0]=='0']
                        label.write_text('\n'.join(lines)+'\n',encoding='utf-8')
                        records[split].append({**r,'image':str(image.relative_to(ROOT)),'label':str(label.relative_to(ROOT)),
                            'derived_label_sha256':sha256(label),'user_video_used':False,'prior_manifest':f'data/manifests/ppe-yolo26s-balanced-v3/{split}.json'})
            stats={};paths={}
            for split,rows in records.items():
                if split in ('train','validation') and len(rows)<32:raise RuntimeError(f'insufficient_{task}_{split}:{len(rows)}')
                path=folder/(split+'.json');write_json(path,{'records':rows,'split':split,'task':task})
                paths[split]=str(path.relative_to(ROOT))
                (folder/(split+'.txt')).write_text(''.join(str(ROOT/r['image'])+'\n' for r in rows),encoding='utf-8')
                counts=Counter()
                for r in rows:
                    counts.update(int(line.split()[0]) for line in (ROOT/r['label']).read_text().splitlines() if line.strip())
                stats[split]={'images':len(rows),'groups':len({r['group'] for r in rows}),'objects':dict(counts),'manifest_sha256':sha256(path)}
            for a,b in (('train','validation'),('train','test'),('validation','test')):
                assert {r['image_sha256'] for r in records[a]}.isdisjoint({r['image_sha256'] for r in records[b]})
                assert {r['group'] for r in records[a]}.isdisjoint({r['group'] for r in records[b]})
            spec={'path':str(folder),'train':'train.txt','val':'validation.txt','names':names}
            if records['test']:spec['test']='test.txt'
            (folder/'data.yaml').write_text(yaml.safe_dump(spec,allow_unicode=True),encoding='utf-8')
            # A bounded smoke subset tests execution only; all pilots use the
            # full frozen lists and identical split/seed for fair comparison.
            for split,limit in (('train',256),('validation',96)):
                subset=random.Random(101).sample(records[split],min(limit,len(records[split])))
                (folder/('smoke-'+split+'.txt')).write_text(''.join(str(ROOT/r['image'])+'\n' for r in subset),encoding='utf-8')
            (folder/'smoke.yaml').write_text(yaml.safe_dump({**spec,'train':'smoke-train.txt','val':'smoke-validation.txt'},allow_unicode=True),encoding='utf-8')
            manifests[task]=paths;statistics[task]=stats
        write_json(target/'source.json',{'archive':source.name,'sha256':source_hash,'documentation':documentation})
        report={'campaign':campaign,'created_at':utc_now(),'statistics':statistics,'manifest_paths':manifests,'excluded':dict(excluded),
                'user_video_used_for_training':False,'video_exclusion_references':video_references,'new_external_dataset_download':False,
                'source_archive_sha256':source_hash,'review':'36 seeded full-frame construction samples reviewed; torso versus full-person annotations distinguished. Previous HHW and safety_vests excluded from v9 training due to observed class errors/partial annotations/product/collage scenes; kept as legacy evaluation references.',
                'split_policy':'Keep source validation; remove SHA/stem/dHash<=6 cross-split components. Reserve 10% train components for added development validation. Exclude SHA/dHash<=4 against all prior inventory and curated PPE splits. User videos dHash<=8 exclusion only.',
                'limitations':['Sampled annotation audit is not exhaustive ground truth certification.','Source site/video IDs unavailable for many photographs; near-hash grouping cannot guarantee site independence.','No new archive test split was provided; independent old tests remain locked.','Native archive images were already stretched to 640 square; upscaling cannot restore missing detail.']}
        write_json(target/'audit.json',report)
        print(json.dumps({'statistics':statistics,'excluded':dict(excluded)},ensure_ascii=True),flush=True)
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    prepare(parser.parse_args().campaign)
