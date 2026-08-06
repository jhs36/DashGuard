# Data and Model Provenance

## 1. 목적과 증거 경계

이 문서는 DashGuard가 로컬 검증에 사용한 실제 도로 영상, BDD100K sample, pretrained model, 파생 ONNX artifact, synthetic telemetry의 출처와 사용 범위를 분리한다. 저장소 파일이나 동결된 project state에서 확인할 수 없는 URL·license 조건·배포 권한은 추측하지 않는다. 현재 저장소 root에는 project-level `LICENSE` 파일이 확인되지 않는다.

`data/raw/`, `data/processed/`, `models/`, `outputs/`의 실제 데이터·weight·산출물은 local workspace이고 `.gitignore` 대상이다. Git에는 converter, schema, tests, 문서만 남긴다.

## 2. 실제 도로 영상

| 항목 | 기록 값 |
|---|---|
| title | Calm City Drive at Daytime |
| creator | Jabriel |
| platform | Pexels |
| recorded license | Pexels License |
| provenance status | `documented_in_project_state_but_source_url_not_archived` |
| pilot use | local Day 9 evaluation only |

Day 9 `segment_manifest.json`에는 원본 SHA-256, metadata와 위 attribution이 기록돼 있다. 각 평가 segment 파일의 SHA-256은 별도 reviewed-video scope CSV에도 저장되어 positive가 0개인 negative ground-truth 범위를 prediction 입력과 독립적으로 확인한다. 그러나 원본 URL과 license snapshot은 저장소에 보존되지 않았다. 그러므로 이 문서는 URL을 추측하거나 새로 만들지 않으며, 기록된 license 이름만으로 현재 배포 권한이나 상업적 사용 조건을 단정하지 않는다.

원본 영상과 세 파생 segment는 Git에 포함하지 않는다. 현재 portfolio 배포물에도 영상을 자동 포함하지 않고 local pilot evaluation에만 사용한다. 외부 배포 전에는 원본 URL, creator attribution, 당시와 현재의 원 license 조건, 필요한 고지 방식을 사용자가 다시 확인해야 한다.

## 3. BDD100K Detection 2020 sample

### 3.1 로컬에서 확인한 구조

로컬 label JSON은 BDD100K Detection 2020 형태의 frame record 배열이며 각 record에 `name`, `attributes`, `timestamp`, `labels`가 있고 bbox label은 `category`와 `box2d`를 포함한다. Converter는 이 구조에서 대상 category를 읽어 YOLO normalized bbox text로 바꾼다.

검증 당시 local workspace에는 다음이 존재했다.

- train sample image 10장과 대응 label 10개
- val sample image 10장과 대응 label 10개
- 합계 image 20장
- 변환된 bbox 합계 199개: train 73개, val 126개
- YOLO sample `data.yaml`과 네 장의 visual comparison artifact

이 20장 sample은 converter와 bbox 변환을 확인하기 위한 것이다. 정식 training은 수행하지 않았고, 모델 성능 평가 데이터나 Day 9 event 평가 데이터로 사용하지 않았다.

### 3.2 공식 원본과 제3자 archive 구분

로컬 raw layout은 `data/raw/bdd100k/soumik/`이라는 제3자 재배포 경로 이름을 포함한다. 이 경로를 BDD100K 공식 배포 endpoint로 표현하지 않는다. 공식 BDD100K 데이터셋의 명칭·Detection 2020 schema와 현재 local archive의 취득 경로·무결성·재배포 조건은 서로 다른 provenance 항목이다.

현재 저장소에는 전체 BDD100K data 또는 제3자 archive를 commit하지 않는다. Quarantine zip, full label JSON, sample image, 변환 image·label 모두 Git ignore 범위다. 외부 공개나 재사용 전에는 공식 원출처의 이용 조건과 실제 취득 archive의 출처·license를 각각 재확인해야 한다.

## 4. Detection and tracking model

| 항목 | 값 |
|---|---|
| model | `yolo11n.pt` |
| use | pretrained detection and tracking pipeline |
| training in this project | not performed |
| repository inclusion | weight file not committed |
| current full-video backend | PyTorch CPU |

Project code는 Ultralytics YOLO model을 로드하고 ByteTrack wrapper에 연속 frame을 전달한다. 이 프로젝트는 `yolo11n.pt`를 학습하거나 fine-tune하지 않았으며, BDD100K sample 20장으로 weight를 변경하지 않았다.

저장소 안에서 확인 가능한 근거만으로 model weight의 모든 license 조건이나 상업적 사용 가능 여부를 단정하지 않는다. 배포 전에는 사용자가 model provider의 원 license, weight 배포 조건, 적용되는 dependency license를 현재 버전 기준으로 재확인해야 한다.

## 5. ONNX derived artifact

| 항목 | 값 |
|---|---|
| derived artifact | `yolo11n_fp32_opset20_640.onnx` |
| source | local FP32 export from `yolo11n.pt` |
| shape / opset | fixed `1×3×640×640`, opset 20 |
| use | CPU consistency and benchmark |
| repository inclusion | ONNX file not committed |
| full-video demo backend | not currently used |

ONNX artifact는 Day 7에 local export하고 두 BDD100K sample image에서 PyTorch raw output 및 post-NMS detection과 비교했다. Benchmark도 single-image CPU 범위다. Video decode, ByteTrack state, overlay, risk, telemetry, event·clip 생성이 포함된 full-video ONNX tracking backend는 구현되지 않았다.

파생 ONNX artifact는 source weight의 license·배포 조건과 독립적이라고 가정하지 않는다. Weight와 파생 artifact를 외부에 배포하기 전 원 조건을 다시 확인해야 한다.

## 6. Telemetry provenance

Day 6–9 검증에는 실제 GPS·IMU·OBD 데이터가 없다. 실제 CSV가 제공되지 않은 경우 pipeline은 deterministic synthetic telemetry를 생성한다.

**Synthetic telemetry**는 고정 규칙으로 만든 예시 speed·acceleration series이며 integration, schema, timestamp synchronization, plotting, event context 검증에만 사용한다. 실제 차량 센서 정확도, 운전 상태 또는 안전성의 근거가 아니다. Risk proxy score도 telemetry 값으로 보정되지 않는다.

## 7. 배포 전 체크리스트

- 실제 도로 영상의 원본 URL과 license snapshot을 확보하고 attribution·재배포 조건을 확인한다.
- 원본 및 segment 영상을 배포 bundle에 자동 포함하지 않는다.
- BDD100K 공식 조건과 local 제3자 archive의 provenance를 각각 확인한다.
- Dataset 전체, 제3자 archive, sample image를 Git에 포함하지 않는다.
- `yolo11n.pt`와 ONNX 파생 artifact의 현재 license·weight 배포 조건을 확인한다.
- Synthetic telemetry를 실제 센서 데이터로 소개하지 않는다.
- Local `outputs/` artifact를 Git 대상이나 공개 benchmark 결과로 취급하지 않는다.
