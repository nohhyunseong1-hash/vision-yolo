# v9 학습 실행 — 2026-11-18

사용자가 기존 학습 계획의 실행을 승인했다. 기존 로컬 라벨 데이터만 사용하며, 크레인 재해/점검 영상은 중복 제외 기준으로만 읽었다. 영상 프레임·배경·의사 라벨은 학습/검증 목록에 없다. 운영 모델 registry와 기존 `.pt`는 교체하지 않는다.

## 현재 인계 상태 — 2026-11-19

사용자 요청으로 **11월 19일 00:50 KST에 일시 중지**했다. 다시 실행하기 전에 커밋하라는 지시를 받았으며, 이번 저장 작업에서는 학습을 재개하지 않았다.

- 총 20개 작업 중 13개 완료: 시험 3개, 후보 비교 8개, 사람 본 학습 seed 101/202 두 개.
- 중지한 작업: `main-person-s-960-seed303`, 92번째 epoch 진행 중. 재개 checkpoint의 0-based epoch는 91이다. epoch 중간 중지일 수 있으므로 92개 epoch를 모두 완주했다고 해석하지 않는다.
- 이후 대기: head/vest 각 seed 101/202/303, 총 6개 본 학습.
- 재개 파일: `outputs/training/campaign-v9-20261118/runs/main-person-s-960-seed303/weights/resume.pt`.
- SHA256: `c9edf2e31674b39a61643f9a7a9718874f076a03638ac50203c3bc1dba660869`.
- optimizer 상태와 EMA 포함을 CPU에서 확인했고, 학습 큐/작업 프로세스의 종료를 확인했다. `STOP` 표시를 유지해 자동 재시작을 막는다.
- 코드 기준 커밋: `1d5cd4fe936b436f1552a7198883a77704086eb0`. 이후 문서만 추가하는 커밋은 캠페인이 검증하는 학습 스크립트 hash를 바꾸지 않는다.
- 기계 판독용 인계 기록: `docs/training-pause-20261119.json`. 지표는 개발 validation 결과이며 크레인 영상/현장 정확도나 배포 승인 결과가 아니다.

**Git에는 코드·설명·검증 기록을 보관한다.** 학습 checkpoint, optimizer, 데이터와 전체 실행 상태는 Git에서 제외되어 이 PC의 `outputs/training/campaign-v9-20261118/`, `data/derived/campaign-v9-20261118/`, `models/`에 남아 있다. 이 폴더를 지우거나 Git clone만으로 재개할 수 있다고 가정하지 않는다.

재개를 명시적으로 요청받으면 먼저 `user-pause-20261119.json`과 `resume.pt` hash, optimizer, 코드/데이터 hash, 기존 실행 유무를 확인한다. 해당 캠페인의 `STOP`만 보관용 이름으로 옮기고 `scripts.training_campaign run --campaign campaign-v9-20261118 --resume`를 숨김 백그라운드로 실행한다. 지금은 실행하지 않는다.

현재 구현의 72시간 상한은 11월 18일 18:45:24 KST부터의 벽시계 시간이며 일시 중지 시간도 포함한다. 재개할 때 기한(11월 21일 18:45:24 KST)과 남은 예산을 먼저 확인한다. 장기간 중지 후에는 시작 시간이나 job JSON을 임의로 바꾸지 말고, 기존 job/config hash와 실행 이력을 보존하는 명시적 예산 갱신 절차가 필요하다.

## 데이터 검수 결과와 계획 조정

기존 HHW 자료 표본에 맨머리/일반 모자의 helmet 라벨, safety_vests 자료에 라벨이 없는 사람·제품 사진·합성 collage를 확인했다. 그 두 자료를 그대로 증량하지 않고 이번 캠페인 학습에서는 제외했다. 기존 test는 후속 호환성/회귀 평가 자료로 보존한다. 22,068장 자료도 의미·출처 검수가 끝나기 전에는 섞지 않는다.

보유 `ppe_construction_v2.zip`(CC BY 4.0)의 고정 seed 전경 표본 36장을 검토했다. person 전신, 머리 상태, vest/no_vest 몸통 라벨을 분리해서 세 작업으로 준비했다. 이 샘플 검수는 모든 원본 라벨의 완전성을 보장하지 않는다. 원본 archive에는 이미 640×640 stretch가 적용되어 있어 1280 입력으로 키워도 사라진 세부 정보를 복구하지는 못한다.

원본 2,926장 중 기존 전체 자료와 SHA/dHash 중복 1,306장, 사람이 라벨되지 않은 제품/불완전 자료 27장, train/validation 사이 근접 중복 연결 집합 91장, split 내부 완전 중복 30장을 제외했다. 남은 train 연결 집합의 10%를 추가 development validation으로 고정했다. 새 자료는 기존 라벨 전체/정제 PPE 전체 split과 먼저 비교해 중복을 제외했다. 사람 모델에는 기존 source-balanced train 403장과 기존 validation/test를 유지한다.

| 작업 | train 이미지 | full validation 이미지 | 학습 중 후보 선택용 validation | 기존 test |
|---|---:|---:|---:|---:|
| person | 1,491 | 1,314 | 한 source group당 최대 180장, 총 564장 | 685장, 이번 학습에서 평가/선택하지 않음 |
| head | 1,084 | 384 | 384장 | 기존 archive test는 후속 평가에서 별도 사용 |
| vest | 1,069 | 380 | 380장 | 기존 archive test는 후속 평가에서 별도 사용 |

사람/머리/조끼의 숫자는 서로 다른 task용 목록이므로 고유 이미지 수로 합산하지 않는다. 많은 원본의 영상/현장 ID가 없어 사이트 단위 독립성은 증명되지 않았다. 근접 해시·원본 stem·기존 source group 분리는 개발 검증이며 독립 현장 성능 평가가 아니다.

## 실행 큐

1. 최대 크기의 s 모델로 head 640, vest 640, person 1280을 각각 3 epoch 시험한다. 이 단계는 train 256 / val 최대 96장의 고정 부분집합이며 정확도 보고용이 아니다. 전체 특징층을 학습해 실제 메모리/수치 안정성과 체크포인트를 확인한다.
2. head n/s 640, vest n/s 640, person n/s × 960/1280의 **8개 후보**를 각각 전체 train으로 15 epoch 학습한다. 같은 task의 입력 목록과 seed 101은 동일하다.
3. development AP50–95가 최고값에서 0.01 이내, 최저 클래스 recall이 최고 AP 후보보다 0.03 이상 낮지 않은 후보 중 validation 추론 시간이 짧은 모델을 task별로 선정한다. 이는 본 학습에 자원을 배분하는 기준이지 운영 채택 기준이 아니다.
4. 선정한 구조를 공식 초기 가중치에서 다시 시작해 seed **101/202/303**, 각각 최대 **150 epoch / patience 25**로 학습한다. 처음 3 epoch는 앞쪽 6개 층을 잠시 고정한 뒤 전체 특징층을 학습한다. 세 seed 결과를 모두 남긴다.

총 최대 20개 작업(시험 3 + 탐색 8 + 본 학습 9)을 GPU 한 개에서 순차 실행한다. 전체 캠페인은 첫 실행부터 **72시간 상한**으로 제한한다. 이것은 GPU utilization을 적분한 사용량이 아니라 검증·준비를 포함한 보수적인 경과 시간 상한이다. 최대 메모리 할당은 13.5GiB를 감시한다. NaN, 수정된 입력, 실패한 작업이 있으면 다음 작업으로 넘어가지 않고 이유를 기록한다. Ultralytics의 첫 epoch OOM batch 축소가 발생하면 실제 batch와 peak VRAM을 기록한다.

각 epoch의 loss, validation 지표, 실제 batch, GPU 메모리, 모델/입력/코드 hash와 checkpoint를 남긴다. `best.pt`는 validation으로만 선택한다. STOP 파일로 요청하면 체크포인트를 남기고 중지한다. 최종화가 optimizer를 제거하기 전에 별도 `resume.pt`를 보존한다. 중단 복구에는 같은 코드·설정·데이터와 `--resume`가 필요하며, 재부팅 후 자동 시작 기능은 설치하지 않는다.

실행 큐는 프로젝트의 단일 실행 잠금을 보유한다. 학습 중 대시보드 추론 서버를 함께 실행하지 않는다. 컴퓨터가 꺼지거나 절전되면 학습도 중단/대기한다. 후보를 자동으로 운영 registry에 등록하거나 공개 Release에 올리지는 않는다.

## 확인 및 중지

프로젝트 폴더에서:

```powershell
.\.venv\Scripts\python.exe -m scripts.training_campaign status --campaign campaign-v9-20261118
.\.venv\Scripts\python.exe -m scripts.training_campaign stop --campaign campaign-v9-20261118
```

- 진행: `outputs/training/campaign-v9-20261118/status.json`
- 설정: 같은 폴더의 `config.json`
- 로그: 같은 폴더의 `logs/`
- 작업별 가중치/곡선/기록: 같은 폴더의 `runs/<job>/`
- 원본/파생 이력과 고정 split: `data/derived/campaign-v9-20261118/audit.json`

학습 큐 완료 후에는 크기별/클래스별 평가, 기존 test, 두 크레인 영상 비교, 최종 노트북 TensorRT 및 4채널 시험을 수행해야 한다. `training_completed_pending_evaluation`은 그 평가와 현장 목표를 통과했다는 뜻이 아니다.
