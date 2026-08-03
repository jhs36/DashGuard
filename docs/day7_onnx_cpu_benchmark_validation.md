# DashGuard Day 7 ONNX CPU export·consistency·benchmark 검증

## 1. 목적과 범위

Day 7의 목적은 검증 가능한 PyTorch 모델을 ONNX FP32 고정 shape 모델로 export하고, 동일 입력에서 PyTorch CPU와 ONNX Runtime(ORT) CPU의 출력 일관성과 latency를 분리해 확인하는 것이다.

검증 시점에 프로젝트 학습 산출물인 `best.pt`가 존재하지 않았으므로, Ultralytics가 제공하고 실제 consistency 검증을 통과한 `yolo11n.pt`를 기준 모델로 사용했다. 따라서 이 결과는 향후 학습된 DashGuard 전용 가중치의 정확도나 성능을 대신하지 않는다.

핵심 범위는 다음과 같다.

- `yolo11n.pt`를 batch 1, 640×640, FP32, 고정 shape ONNX로 export
- 동일한 두 BDD100K sample 이미지에서 raw output과 최종 detection consistency 검증
- PyTorch CPU와 ORT `CPUExecutionProvider`의 startup, preprocess, inference, postprocess, image-to-detections latency 분리
- 이미지와 backend 실행 위치를 독립적으로 균형화한 최종 benchmark 산출

TensorRT, INT8, DirectML은 이번 핵심 범위가 아니다. 이들은 provider, 정밀도, 하드웨어 또는 실행 계약을 바꾸므로 별도의 정확도 기준과 benchmark 설계가 필요하다.

## 2. 환경과 의존성

| 항목 | 검증 값 |
|---|---|
| 운영체제 | Windows (`10.0.26200`, AMD64) |
| Python | 3.11.5 |
| CPU | AMD Ryzen 7 5700X3D 8-Core Processor |
| Logical CPU | 16 |
| PyTorch | 2.13.0+cpu |
| Ultralytics | 8.4.107 |
| ONNX | 1.22.0 |
| ONNX Runtime | 1.28.0 |
| ONNX Slim | 0.1.95 |
| NumPy | 2.4.4 |
| OpenCV | 5.0.0 |

직접 의존성은 `onnx==1.22.0`, `onnxruntime==1.28.0`, `onnxslim==0.1.95`로 선언했다. `protobuf==7.35.1`, `ml_dtypes==0.5.4`, `flatbuffers==25.12.19`는 현재 검증 환경의 transitive dependency이며 직접 의존성으로 선언하지 않는다. 재현 환경의 CPU 전용 Torch wheel 정책은 `requirements.txt`의 PyTorch CPU extra index와 `+cpu` pin을 따른다.

Benchmark thread 설정은 다음과 같다.

- PyTorch CPU intra-op 8, inter-op 1, eval mode, inference mode
- ORT intra-op 8, inter-op 1
- ORT execution mode `ORT_SEQUENTIAL`
- ORT graph optimization `ORT_ENABLE_ALL`
- ORT provider는 정확히 `CPUExecutionProvider`
- OpenCV threads 1
- CPU affinity와 process priority는 변경하지 않음

## 3. Export 계약과 결과

### 3.1 Export 계약

| 옵션 | 값 |
|---|---|
| 입력 모델 | `yolo11n.pt` |
| format | ONNX |
| dtype | FP32 |
| batch | 1 |
| imgsz | 640 |
| dynamic | false |
| simplify | true |
| opset | 20 |
| embedded NMS | false |
| device | CPU |

Ultralytics export 계약은 다음 옵션과 동등하다.

```powershell
& ".\.venv\Scripts\yolo.exe" export model=".\yolo11n.pt" format=onnx imgsz=640 batch=1 dynamic=False simplify=True opset=20 nms=False device=cpu
```

Export 결과는 audit 가능한 이름인 `outputs/day7/models/yolo11n_fp32_opset20_640.onnx`로 보존했다.

### 3.2 모델 검증 결과

| 항목 | PyTorch | ONNX |
|---|---:|---:|
| 경로 | `yolo11n.pt` | `outputs/day7/models/yolo11n_fp32_opset20_640.onnx` |
| 크기 | 5,613,764 bytes | 10,741,397 bytes |
| SHA-256 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` | `9b78ee7df552d324f72a3a12acac8a3ab23fc7b5bc52aadb21d05045a50c7227` |

ONNX checker는 PASS했다. ONNX 구조는 다음과 같다.

- IR version: 9
- default-domain opset: 20
- graph node count: 320
- input: `images`, `[1,3,640,640]`, `tensor(float)`
- output: `output0`, `[1,84,8400]`, `tensor(float)`
- 실제 provider: `CPUExecutionProvider`

`.gitignore`는 `*.pt`, `*.onnx`, `outputs/*`를 제외한다. 따라서 PyTorch/ONNX binary와 아래 JSON·CSV는 재현 과정에서 생성하거나 로컬로 보관하는 산출물이며 Git stage 대상이 아니다.

## 4. 전처리와 출력 비교

### 4.1 공유 입력 계약

두 backend는 이미지별로 동일한 최종 FP32 NumPy 배열을 입력으로 사용했다.

1. OpenCV BGR decode
2. 640×640 고정 centered Ultralytics LetterBox
3. BGR → RGB
4. HWC → CHW
5. batch dimension 추가
6. C-contiguous FP32 변환
7. `/255.0` 정규화

LetterBox는 `auto=false`, `scale_fill=false`, `scaleup=true`, `center=true`, padding value 114, `cv2.INTER_LINEAR`로 고정했다. Consistency 검증에서는 같은 NumPy array object를 PyTorch와 ORT backend에 전달했다.

입력 이미지는 다음 두 BDD100K sample이다.

- `data/raw/bdd100k/soumik/sample/images/train/0000f77c-6257be58.jpg`
- `data/raw/bdd100k/soumik/sample/images/train/0000f77c-62c2a288.jpg`

### 4.2 Raw output 비교

두 backend의 raw output shape는 모두 `[1,84,8400]`이다. 두 이미지, 총 1,411,200개 element의 aggregate 차이는 다음과 같다.

| 통계 | 값 |
|---|---:|
| max absolute difference | 0.00787353515625 |
| mean absolute difference | 2.3537877257717864e-06 |
| RMSE | 3.2843825582829846e-05 |
| P50 absolute difference | 2.1252517079517474e-08 |
| P95 absolute difference | 7.8302357486848e-08 |
| P99 absolute difference | 6.103515625e-05 |
| raw `np.allclose(rtol=1e-4, atol=1e-4)` | 2/2 passed |

Raw allclose는 진단 정보다. 프로젝트 acceptance는 실제 공유 postprocess 뒤의 detection consistency로 결정한다.

### 4.3 공유 postprocess와 matching

양 backend output에 동일한 `ultralytics.utils.nms.non_max_suppression`과 `ultralytics.utils.ops.scale_boxes`를 적용했다.

- confidence threshold: 0.25
- IoU threshold: 0.7
- max detections: 300
- agnostic NMS: false
- multi-label: false
- target class IDs: `[0,1,2,3,5,7]`
- target classes: person, bicycle, car, motorcycle, bus, truck

NMS가 입력 tensor를 변경할 수 있으므로 raw output을 복사한 뒤 처리했다. LetterBox ratio/padding을 사용해 원본 이미지 좌표로 복원하고 최종 `Detection`을 생성했다.

Detection matching은 같은 class ID 후보를 bbox IoU 내림차순으로 정렬한 뒤, 이미 사용된 detection을 제외하는 결정적 greedy one-to-one 방식이다. 동일 IoU에서는 PyTorch index와 ONNX index가 tie-breaker다.

프로젝트 acceptance 기준은 다음과 같다.

- detection 개수가 같음
- unmatched detection 0
- class ID와 class name exact match
- matched bbox IoU ≥ 0.995
- bbox 좌표 최대 절대 차이 ≤ 0.25 px
- confidence 절대 차이 ≤ 0.001

이 기준은 현재 모델, 현재 전처리, FP32 CPU backend를 위한 프로젝트 기준이며 범용 ONNX 표준이 아니다.

### 4.4 Consistency 결과

Evidence: `outputs/day7/validation/day7_backend_consistency.json`

- schema: `dashguard.onnx_backend_consistency.v2`
- evidence SHA-256: `005d02eced8173f2d3483ef2217bb99f34bd94dcbc89c465dfb05d2ff458b2d6`
- structural validation: passed
- numerical acceptance: passed
- PyTorch detections: 4
- ONNX detections: 4
- matched: 4
- unmatched PyTorch/ONNX: 0/0
- minimum matched IoU: 0.999998916443024
- maximum bbox coordinate difference: 0.0001220703125 px
- maximum confidence difference: 9.5367431640625e-07

## 5. Benchmark 방법

Evidence:

- JSON: `outputs/day7/benchmarks/day7_cpu_backend_benchmark_balanced.json`
- CSV: `outputs/day7/benchmarks/day7_cpu_backend_benchmark_balanced.csv`
- schema: `dashguard.cpu_backend_benchmark.v2`
- balanced JSON SHA-256: `98d6d87c0f80e5fa3f47f09157d8408a85188e9b44b055ebbefe5e2d478eea61`
- balanced CSV SHA-256: `401aba9700f48a5fa23ef59cc048024c64e37460fa744d3e0f8c97767c237b58`

측정 clock은 `time.perf_counter_ns()`이며 내부 sample은 nanosecond로 보존한 뒤 millisecond로 출력했다. 각 timed 영역에서는 garbage collection을 비활성화하고 원래 상태로 복구했다. Warmup sample은 결과 sample에 포함하지 않았다. 모든 timed sample은 finite, nonnegative 검증을 거쳤으며 선택적으로 제거하지 않았다.

| 경계 | 포함 | 주요 제외 항목 |
|---|---|---|
| startup load | PyTorch application model 준비 또는 ORT SessionOptions/session 생성 | warmup, 반복 inference |
| preprocess | 메모리 BGR에서 최종 `[1,3,640,640]` FP32 배열 생성 | disk read/decode, inference |
| inference-only | 미리 생성한 tensor/array의 direct model forward 또는 `session.run()` | tensor 생성, NumPy adapter, postprocess, 통계 |
| postprocess | raw output 복사, 공유 NMS, 좌표 복원, `Detection` 생성 | inference |
| image-to-detections | 메모리 BGR preprocess, inference, adapter, NMS, 좌표 복원, `Detection` 생성 | disk read/decode, load, consistency 검증, 통계/파일 출력 |

Startup load는 backend별 단일 측정이다. `measurement_type`은 `single_measurement`이며 percentile 필드는 유일한 sample과 같을 뿐 latency 분포를 나타내지 않는다.

### 5.1 Warmup과 timed sample

- stage별 warmup: 20
- stage/backend별 timed iterations: 100
- inference-only, postprocess, image-to-detections backend별 sample: 100
- percentile 방식: NumPy linear interpolation
- standard deviation: population standard deviation (`ddof=0`)

### 5.2 4-step balanced schedule

초기 benchmark는 짝수 iteration에서 image 0/PT-first, 홀수 iteration에서 image 1/ORT-first를 사용해 이미지와 실행 위치가 결합됐다. 이 결과는 공정성 문제를 발견한 audit 과정으로만 보존하며 primary 성능 결과로 사용하지 않는다.

최종 benchmark는 다음 주기를 사용했다.

| `iteration % 4` | 이미지 | backend 순서 |
|---:|---:|---|
| 0 | 0 | PyTorch → ORT |
| 1 | 0 | ORT → PyTorch |
| 2 | 1 | PyTorch → ORT |
| 3 | 1 | ORT → PyTorch |

Warmup 20은 조합별 5회, timed 100은 조합별 25회다. 각 이미지에서 각 backend가 first와 second를 같은 횟수 수행한다. 4의 배수가 아닌 warmup/iteration은 오류로 거부한다.

## 6. 최종 balanced 결과

아래 balanced 결과가 Day 7의 primary 성능 결과다. 단위는 millisecond다.

| 단계 | Backend | Count | P50 | P95 |
|---|---|---:|---:|---:|
| shared preprocess | shared | 100 | 2.57785 | 3.067275 |
| inference-only | PyTorch CPU | 100 | 59.3304 | 94.36384 |
| inference-only | ORT CPU | 100 | 29.74725 | 33.43403 |
| postprocess | PyTorch output | 100 | 1.96985 | 2.528725 |
| postprocess | ONNX output | 100 | 1.9313 | 2.4362 |
| image-to-detections | PyTorch CPU | 100 | 61.90475 | 90.24781 |
| image-to-detections | ORT CPU | 100 | 30.50155 | 36.621685 |

| 파생값 | P50 | P95 |
|---|---:|---:|
| inference-only ONNX speedup over PyTorch | 1.99448× | 2.82239× |
| image-to-detections ONNX speedup over PyTorch | 2.02956× | 2.46433× |

Startup single measurements:

- PyTorch: 64.0313 ms
- ORT CPU: 90.8563 ms

Sequential FPS는 `1000 / latency_ms`로 계산한 latency 역수다. 동시 요청을 처리해 측정한 throughput이 아니다.

## 7. 중요한 관측

- ORT CPU의 P50은 inference-only와 image-to-detections 모두 PyTorch CPU보다 약 2배 빠르다.
- PyTorch inference-only P50은 first 54.8865 ms, second 80.76485 ms였다.
- PyTorch image-to-detections P50은 first 54.39035 ms, second 70.7941 ms였다.
- ORT inference-only P50도 first 26.13455 ms, second 31.01565 ms로 위치 효과가 관측됐다.
- Balanced schedule은 first/second 위치 효과를 두 이미지와 두 backend에 동일하게 배분한다. 위치 효과 자체를 제거하는 설계는 아니다.
- P95에는 OS scheduling, paired backend order, CPU frequency와 thermal 상태, background load의 영향이 포함될 수 있다.
- 이상치 sample은 제거하지 않았다.

## 8. Day 3 속도와의 구분

Day 3의 약 18.66 FPS는 video decode, detection, tracking, rendering 등을 포함한 전체 pipeline 수치다.

Day 7의 image-to-detections는 이미 메모리에 decode된 단일 BGR 이미지에서 preprocess, detection inference, NMS, 좌표 복원, `Detection` 생성만 포함한다. 파일 decode, tracking, rendering은 포함하지 않는다.

따라서 Day 3 FPS와 Day 7의 파생 sequential FPS를 같은 benchmark처럼 직접 비교하지 않는다.

## 9. 한계와 해석 금지 사항

- 입력은 BDD100K sample 이미지 2장뿐이다.
- 모델은 학습된 `best.pt`가 아니라 현재 검증된 `yolo11n.pt`다.
- FP32, batch 1, 고정 640×640만 검증했다.
- ORT provider는 CPU 하나뿐이다.
- 한 대의 PC에서만 측정했다.
- CPU affinity와 process priority를 통제하지 않았다.
- background load, CPU frequency, thermal 상태를 통제하지 않았다.
- paired 실행의 first/second 위치 효과가 남아 있다.
- 파생 FPS는 concurrent throughput이 아니다.
- 모델 정확도, false positive/false negative, 시스템 안전성을 평가하지 않았다.
- Backend export consistency가 통과해도 false positive/false negative 문제가 해결됐다는 뜻이 아니다.
- 이 결과는 전체 DashGuard pipeline이 30 FPS로 동작한다는 증거가 아니다.
- 현재 모델이나 파이프라인이 production-ready라는 근거가 아니다.

## 10. 재현 명령

아래 명령은 저장소 root에서 Windows PowerShell로 실행한다. `.venv` 경로는 저장소 상대 경로를 사용하며 특정 사용자 홈 경로에 고정하지 않는다. Output은 기본적으로 overwrite하지 않으므로, 동일 이름이 이미 존재하면 보존할지 `--overwrite`로 교체할지 명시적으로 결정해야 한다.

### 10.1 환경 설치와 검사

CPU 전용 Torch index 정책을 포함한 requirements를 먼저 사용한다. pip 자체는 업그레이드하지 않는다.

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r ".\requirements.txt"
& ".\.venv\Scripts\python.exe" -m pip install -e "."
& ".\.venv\Scripts\python.exe" -m pip check
```

### 10.2 ONNX consistency

```powershell
& ".\.venv\Scripts\python.exe" ".\scripts\day7_validate_onnx_outputs.py" `
  --image ".\data\raw\bdd100k\soumik\sample\images\train\0000f77c-6257be58.jpg" `
  --image ".\data\raw\bdd100k\soumik\sample\images\train\0000f77c-62c2a288.jpg" `
  --pytorch-model ".\yolo11n.pt" `
  --onnx-model ".\outputs\day7\models\yolo11n_fp32_opset20_640.onnx" `
  --imgsz 640 `
  --confidence 0.25 `
  --iou-threshold 0.7 `
  --max-det 300 `
  --output-json ".\outputs\day7\validation\day7_backend_consistency.json"
```

### 10.3 Balanced CPU benchmark

Consistency가 passed이고 model/image hash가 일치해야 benchmark가 시작된다.

```powershell
& ".\.venv\Scripts\python.exe" ".\scripts\day7_benchmark_backends.py" `
  --image ".\data\raw\bdd100k\soumik\sample\images\train\0000f77c-6257be58.jpg" `
  --image ".\data\raw\bdd100k\soumik\sample\images\train\0000f77c-62c2a288.jpg" `
  --pytorch-model ".\yolo11n.pt" `
  --onnx-model ".\outputs\day7\models\yolo11n_fp32_opset20_640.onnx" `
  --consistency-json ".\outputs\day7\validation\day7_backend_consistency.json" `
  --imgsz 640 `
  --threads 8 `
  --opencv-threads 1 `
  --warmup 20 `
  --iterations 100 `
  --output-json ".\outputs\day7\benchmarks\day7_cpu_backend_benchmark_balanced.json" `
  --output-csv ".\outputs\day7\benchmarks\day7_cpu_backend_benchmark_balanced.csv"
```

### 10.4 Tests

```powershell
& ".\.venv\Scripts\python.exe" -m pytest ".\tests\test_onnx_validation.py" -q
& ".\.venv\Scripts\python.exe" -m pytest ".\tests\test_backend_benchmark.py" -q
& ".\.venv\Scripts\python.exe" -m pytest ".\tests\test_onnx_validation.py" ".\tests\test_backend_benchmark.py" -q
& ".\.venv\Scripts\python.exe" -m pytest -q
```

### 10.5 로컬 산출물 경로

- ONNX model: `outputs/day7/models/yolo11n_fp32_opset20_640.onnx`
- consistency evidence: `outputs/day7/validation/day7_backend_consistency.json`
- 초기 공정성 audit benchmark: `outputs/day7/benchmarks/day7_cpu_backend_benchmark.json`, `.csv`
- primary balanced benchmark: `outputs/day7/benchmarks/day7_cpu_backend_benchmark_balanced.json`, `.csv`

초기 benchmark는 삭제하지 않지만 primary 결과로 선택하지 않는다. 모델 binary와 모든 `outputs/day7` 산출물은 `.gitignore`에 의해 Git에서 제외된다.
