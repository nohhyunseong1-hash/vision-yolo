import copy
from pathlib import Path

import pytest
import yaml

from app.common import sha256, write_json
from scripts import train_campaign_job as worker
from scripts.training_campaign import select_candidate


def candidate(name,ap,recall,speed):
    return {'job_id':name,'status':'completed','best_validation_metrics':{'metrics/mAP50-95(B)':ap},
            'class_recall':recall,'validation_speed_ms':{'inference':speed}}


def test_fast_candidate_cannot_hide_negative_class_recall_loss():
    accurate=candidate('accurate',.6,[.95,.85],8)
    unsafe=candidate('unsafe',.599,[.99,.6],1)
    equivalent=candidate('equivalent',.595,[.94,.84],5)
    assert select_candidate([accurate,unsafe,equivalent])['job_id']=='equivalent'
    unsafe['status']='failed'
    with pytest.raises(ValueError,match='incomplete'):select_candidate([accurate,unsafe])


@pytest.fixture
def dataset(tmp_path,monkeypatch):
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    folder=tmp_path/'data/derived/example';folder.mkdir(parents=True)
    manifests={}
    for split in ('train','validation','test'):
        image=folder/(split+'.jpg');image.write_bytes(split.encode())
        label=folder/(split+'.txt');label.write_text('0 .5 .5 .2 .2\n')
        row={'image':str(image.relative_to(tmp_path)),'label':str(label.relative_to(tmp_path)),
             'image_sha256':sha256(image),'derived_label_sha256':sha256(label),'group':split,'user_video_used':False}
        path=folder/(split+'.json');write_json(path,{'records':[row]})
        manifests[split]={'path':str(path.relative_to(tmp_path)),'sha256':sha256(path)}
        (folder/(split+'-list.txt')).write_text(str(image)+'\n')
    data=folder/'data.yaml';data.write_text(yaml.safe_dump({'path':str(folder),'train':'train-list.txt','val':'validation-list.txt','names':['person']}))
    spec={'data':str(data.relative_to(tmp_path)),'data_sha256':sha256(data),'manifests':manifests,
          'list_sha256':{s:sha256(folder/(s+'-list.txt')) for s in ('train','validation')}}
    return folder,spec


def test_dataset_refuses_changed_labels(dataset):
    folder,spec=dataset
    worker.verify_dataset(spec)
    (folder/'train.txt').write_text('0 .5 .5 .9 .9\n')
    with pytest.raises(ValueError,match='dataset_file_changed'):worker.verify_dataset(spec)


def test_even_rehashed_train_list_cannot_include_test(dataset):
    folder,spec=dataset
    (folder/'train-list.txt').write_text(str(folder/'test.jpg')+'\n')
    spec['list_sha256']['train']=sha256(folder/'train-list.txt')
    with pytest.raises(ValueError,match='outside_approved_split'):worker.verify_dataset(spec)
