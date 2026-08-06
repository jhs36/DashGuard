# DashGuard Day 8 Gradio 통합 데모 검증

## 1. 목표와 완료 범위

Day 8의 목표는 기존 Day 2·3·5·6의 검증된 처리 라이브러리를 하나의 로컬 Gradio 화면에서 실행하고, 한 run의 영상·CSV·그래프·event·clip을 서로 일관된 workspace로 제공하는 것이다.

완료 범위는 다음과 같다.

- 최대 30초 영상의 입력 검사와 H.264 MP4 표준화
- `yolo11n.pt`를 사용하는 PyTorch CPU 탐지·ByteTrack 추적
- 기존 Day 5 risk proxy 계산과 그래프 생성
- 선택 telemetry CSV 또는 deterministic synthetic telemetry 처리
- 기존 Day 6 시간 동기화, event 후보 생성 및 clip 추출
- primary 산출물과 브라우저 제공용 published 산출물의 격리
- Gradio 진행 상태, 결과 영상, 두 PNG, event 표, 대표 clip 및 전체 clip 제공
- no-server, localhost HTTP, 실제 브라우저 end-to-end smoke
- run summary와 SHA-256을 이용한 산출물 무결성 감사

Day 8은 새 detector, tracker, risk 수식 또는 event 수식을 추가하지 않는다. ONNX full-video tracking, DirectML, 외부 배포도 포함하지 않는다.

## 2. 아키텍처와 thin UI 경계

처리 흐름은 다음과 같다.

```text
Gradio upload
→ Day 8 input preflight
→ Day 2 video standardization
→ Day 3 PyTorch CPU detection/tracking/render
→ Day 5 risk CSV/plot
→ Day 6 telemetry generation or copy/sync
→ Day 6 event timeline/JSON
→ Day 6 FFmpeg clips
→ published copy + SHA-256 verification
→ Gradio presentation
```

`src/dashguard/demo/workflow.py`가 Gradio와 독립적으로 전체 workflow를 조정한다. `src/dashguard/demo/gradio_app.py`는 다음 thin UI 책임만 가진다.

- 입력 파일 경로와 고정 설정을 workflow에 전달
- coarse progress stage를 한국어 설명으로 표시
- 현재 callback이 만든 run만 결과로 수용
- primary가 아닌 `published/` 파일만 브라우저 component에 반환
- `events.json`의 실제 키로 event 표를 구성
- 사용자에게 안전한 오류 문구와 run ID·실패 단계를 표시

UI에는 backend 선택기와 threshold slider가 없다. backend는 `pytorch_cpu`, 장치는 CPU, 기본 모델은 `yolo11n.pt`, confidence는 0.25, image size는 640으로 고정된다. `scripts/day8_gradio_demo.py`는 app 생성, queue concurrency 1, localhost launch만 담당한다.

## 3. 입력 계약

### 영상

- Gradio 영상 component는 upload 전용이다.
- 지원 확장자는 `.avi`, `.m4v`, `.mkv`, `.mov`, `.mp4`, `.webm`이다.
- 파일 존재, 확장자, 양의 width·height·FPS·duration을 workspace 생성 전에 검사한다.
- 최대 길이는 30초이며 호출자가 이 hard limit보다 큰 값을 지정할 수 없다.
- 입력 metadata가 30초를 초과하면 workspace를 만들지 않고 거부한다.
- 모든 허용 입력은 run 안의 `input/standardized_input.mp4`로 표준화한 뒤 분석한다.

실제 browser run 입력은 H.264, 1280×720, 30 FPS, 30.0초, 900 frames였다. Gradio API smoke 중 인플레이스 변환된 `day2_synthetic_input.mp4`는 원본 복구 근거가 없으므로 Day 8 검증 입력에서 제외했다. 실제 browser run에는 Gradio 임시 upload에 보존된 `day3_baseline_input.mp4`를 사용했다.

### 모델과 telemetry

- 모델 파일이 없으면 workspace 생성 전에 거부한다.
- launcher 기본 모델은 저장소 root의 `yolo11n.pt`다.
- telemetry는 선택 입력이며 `.csv` 파일만 허용한다.
- 제공된 CSV는 run의 `input/telemetry_input.csv`로 exclusive copy하고 원본을 수정하지 않는다.
- telemetry를 제공하지 않으면 표준화 영상 길이에 맞춰 10 Hz synthetic CSV를 생성한다.

## 4. run workspace 구조

각 callback은 UTC timestamp와 UUID를 포함하는 고유 run ID를 만들고 `outputs/day8/runs/<run_id>/`에 독립 workspace를 생성한다.

```text
<run_id>/
├── input/
│   ├── standardized_input.mp4
│   └── synthetic_telemetry.csv 또는 telemetry_input.csv
├── analysis/
│   ├── annotated_video.mp4
│   ├── tracks.csv
│   └── metrics.json
├── risk/
│   ├── risk_scores.csv
│   └── risk_proxy.png
├── telemetry/
│   ├── risk_telemetry_sync.csv
│   └── telemetry_plot.png
├── events/
│   ├── frame_timeline.csv
│   └── events.json
├── clips/
│   ├── clip_manifest.json       # event가 있을 때만
│   └── event_*.mp4              # event가 있을 때만
├── published/
│   ├── annotated_video.mp4
│   ├── risk_proxy.png
│   ├── telemetry_plot.png
│   └── event_*.mp4
└── run_summary.json
```

기존 run ID는 덮어쓰지 않는다. preflight 뒤 workspace가 만들어진 후 실패하면 가능한 경우 `status=failed`, 실패 stage, 오류 종류·메시지·traceback과 이미 생성된 파일 목록을 `run_summary.json`에 기록한다.

최초 실제 workflow smoke를 강제 종료하면서 생긴 `outputs/day8/runs/day8_pytorch_smoke_20260804`는 `input/standardized_input.mp4`만 남은 불완전 workspace다. 삭제하거나 성공 근거로 사용하지 않았다. 성공한 직접 workflow smoke는 별도 workspace인 `outputs/day8/runs/day8_pytorch_smoke_20260804_01`이다.

## 5. PyTorch CPU pipeline

실제 browser run은 다음 고정 처리 계약을 사용했다.

| 항목 | 값 |
|---|---:|
| backend | `pytorch_cpu` |
| model | `yolo11n.pt` |
| device | CPU |
| confidence | 0.25 |
| image size | 640 |
| trajectory length | 30 |
| processed frames | 900 |
| frames with tracks | 900 |
| tracking records | 5,308 |
| unique track IDs | 107 |
| model load | 2.2098572초 |
| tracking/render | 44.9636935초 |
| transcode | 6.7894204초 |
| tracking end-to-end | 51.7531139초 |
| tracking/render FPS | 20.0161493 |
| end-to-end FPS | 17.3902579 |

`tracking end-to-end`는 model load를 제외하고 tracking/render와 transcode를 합친 해당 라이브러리 결과다. 전체 workflow 시간은 표준화, risk, telemetry, event, clip, publishing 등을 추가로 포함한다.

## 6. telemetry 제공/미제공 계약

### 제공한 경우

- `.csv` 확장자와 파일 존재를 preflight에서 확인한다.
- 원본을 수정하지 않고 현재 run에 복사한다.
- 기존 Day 6 schema validation과 nearest timestamp sync를 적용한다.
- offset은 0.0초, tolerance는 0.05초다.
- 일치 행이 0이어도 workflow는 성공할 수 있지만 warning을 남긴다.

### 제공하지 않은 경우

- deterministic synthetic telemetry를 10 Hz로 생성한다.
- 실제 browser run에서는 0.0~30.0초 범위의 296행을 생성했다.
- 12.0~12.4초 missing interval과 28.2~29.1초 hard-brake evidence 구간을 포함하는 기존 Day 6 sample 계약이다.
- 생성 source와 UI·그래프 제목에 `synthetic`임을 명시한다.

Synthetic telemetry는 UI와 pipeline 통합을 설명하기 위한 값이며 실제 GPS·IMU·OBD 센서의 정확도, offset, drift, jitter 또는 운전자 행동 정확도의 근거가 아니다. telemetry 값은 video-only risk proxy 수식에 합산되지 않는다.

## 7. event 0개 처리

`events.json`의 event 배열이 비어 있으면 다음 계약을 따른다.

- clip 추출을 호출하지 않는다.
- 가짜 `clip_manifest.json`을 만들지 않는다.
- primary/published clip 목록은 빈 목록이다.
- UI event 표는 고정 열을 가진 빈 DataFrame이다.
- 첫 event clip은 `None`, 전체 clip은 빈 목록으로 반환한다.
- 상태에는 `event candidate 없음`을 표시한다.

이 경계는 자동 테스트로 검증했다. 실제 browser run은 event 2개가 있어 두 clip을 생성했다.

## 8. published 산출물 격리

브라우저에는 분석 단계의 primary 파일을 직접 반환하지 않는다. 다음 파일을 `published/`로 exclusive copy한 뒤 primary-before, primary-after, published SHA-256이 모두 같은지 확인한다.

- 주석 영상
- risk plot
- telemetry plot
- 각 event clip

현재 callback의 `run_dir` 바로 아래 `published/`에 있는 파일만 UI 결과로 허용한다. 이전 run, output root 밖 경로, primary와 같은 경로, event/clip 개수가 다른 결과는 presentation validation 오류로 거부한다. 이 분리는 Gradio가 결과 파일을 다루는 동안 primary 산출물을 보호하기 위한 파일 경계이며 보안 sandbox나 완전한 파일시스템 트랜잭션을 의미하지 않는다.

## 9. Gradio UI 구성

Gradio 6.22.0 `Blocks` app은 다음 component를 제공한다.

- upload 전용 입력 영상
- 선택 telemetry CSV
- 분석 실행 버튼
- backend, model, telemetry source, event count, 전체 처리시간, warning과 limitation을 포함한 상태 요약
- 분석 영상
- risk proxy PNG
- telemetry PNG
- event 후보 표
- events 순서의 첫 번째 대표 clip
- 모든 event clip 파일 목록

callback과 queue의 동시 실행 한도는 1이다. 진행 상태는 입력 확인, 영상 표준화, 객체 탐지·추적, risk proxy, telemetry, event, clip, publishing, 완료 순서로 표시한다.

## 10. 오류 처리

영상 미입력은 workflow 호출 전에 `gr.Error`로 거부한다. 잘못된 경로·확장자·metadata·30초 초과·model·telemetry는 preflight에서 거부하며 이 경우 새 workspace를 만들지 않는다.

Workspace 생성 뒤 오류가 발생하면 후속 stage를 중단하고 실패 summary를 best effort로 기록한 뒤 원래 예외를 다시 발생시킨다. UI는 내부 경로나 traceback을 성공 출력으로 노출하지 않고, 분류된 한국어 오류와 현재 run ID·stage를 표시한다. 예상하지 못한 예외의 상세 정보는 server log에 남긴다.

실제 브라우저에서 영상을 선택하지 않고 분석을 눌렀을 때 `분석할 영상을 업로드하세요.` 오류가 표시됐고, 앱과 callback은 종료되지 않았다. 그 뒤 페이지를 계속 사용할 수 있어 영상 미입력 오류 복구를 PASS로 판정했다.

## 11. 자동 테스트

Day 8 targeted 범위는 다음 세 파일이다.

- `tests/test_day8_workflow.py`
- `tests/test_day8_presentation.py`
- `tests/test_day8_gradio_app.py`

Python 계약은 `pyproject.toml`의 `requires-python = ">=3.11"`이며 Gradio 6.22.0을 `pyproject.toml`, `requirements.txt`, `requirements-lock.txt`에 고정했다.

검증 범위에는 preflight와 안전한 run ID, 고유 workspace, public library 호출 순서, 고정 PyTorch CPU 설정, telemetry 두 분기, 0 event, failure summary, atomic summary 설치, published 격리·hash, telemetry plot, UI component와 progress, 현재 run 검증, 안전한 오류, launcher의 localhost·share 정책이 포함된다.

| 실행 | 결과 |
|---|---:|
| Day 8 targeted | 62 passed in 9.16s |
| 전체 pytest | 586 passed in 17.61s |

추가 환경 감사에서 `pip check`는 `No broken requirements found.`, 현재 environment의 `pip freeze`와 `requirements-lock.txt`는 줄바꿈 정규화 후 82행 exact match였다. `compileall`, Day 8 package/app import, launcher `--help`, README와 본 문서의 UTF-8 strict decode도 PASS했다.

테스트는 mock/fake pipeline을 사용하는 단위·통합 계약 검증이며 실제 30초 모델 추론이나 사람 브라우저 재생 검증을 대신하지 않는다.

## 12. no-server와 localhost HTTP smoke

자동 HTTP smoke와 실제 브라우저 end-to-end smoke를 구분한다.

### no-server smoke

`create_gradio_app()`만 호출해 server를 launch하지 않고 `gr.Blocks`, callback dependency 1개와 component config 생성을 확인했다. 확인 환경은 Python 3.11, Gradio 6.22.0이며 PASS했다. 자동 테스트는 queue 설정도 server launch 없이 확인한다.

### localhost HTTP smoke

분석 callback을 실행하지 않은 별도 smoke에서 app을 `127.0.0.1:7876`, `share=False`로 임시 launch하고 `GET /`를 요청했다. HTTP status 200과 `text/html; charset=utf-8`을 확인한 뒤 server를 정상 종료했다. 이 검증은 app이 localhost에서 응답한다는 사실만 확인하며 업로드, 30초 workflow, media playback 또는 사람 브라우저 동작을 검증하지 않는다.

실제 launcher는 `127.0.0.1`, 기본 port 7860, `share=False`, upload 최대 1 GB, 현재 Day 8 output root만 `allowed_paths`로 설정한다.

## 13. 실제 브라우저 end-to-end smoke

실제 사람 브라우저 검증은 다음 run에서 수행했다.

```text
outputs/day8/runs/gradio_20260804T150620846851Z_02c79fa334854596a450a5e8acdd1177
```

| 검증 항목 | 결과 |
|---|---|
| 페이지 로드 | PASS |
| PyTorch CPU 표시 | PASS |
| 단계별 진행 상태 | PASS |
| 분석 영상 재생 | PASS |
| risk proxy PNG 표시 | PASS |
| telemetry PNG 표시 | PASS |
| event 표 2행 표시 | PASS |
| 첫 event clip 재생 | PASS |
| 전체 clip 2개 접근 | PASS |
| 처리 정보와 한계 표시 | PASS |
| 영상 미입력 오류 표시 후 앱 유지 | PASS |

이 표는 사람 브라우저에서 페이지와 media를 직접 확인한 결과다. Gradio API 또는 localhost `GET /`만으로 판정한 자동 HTTP smoke가 아니다.

브라우저 연결 종료 과정에서 Windows console에 `WinError 10054`가 2회 출력됐다. callback과 server는 유지됐고 run은 `succeeded`였으며 영상·PNG·clip 재생 장애도 없었다. 현재는 비차단 관찰 사항이다. 반복적으로 실행 실패나 재생 장애를 일으킬 경우에만 재검토한다.

## 14. 실제 처리 결과

Primary evidence는 해당 run의 `run_summary.json`, `events/events.json`, `clips/clip_manifest.json`이다.

### run과 실행 시각

| 항목 | 값 |
|---|---|
| run ID | `gradio_20260804T150620846851Z_02c79fa334854596a450a5e8acdd1177` |
| 시작 시각 UTC | 2026-08-04 15:06:20.846851 UTC |
| 시작 시각 KST | 2026-08-05 00:06:20.846851 KST |
| 종료 시각 UTC | 2026-08-04 15:07:30.211201 UTC |
| 종료 시각 KST | 2026-08-05 00:07:30.211201 KST |
| status | `succeeded` |
| backend | `pytorch_cpu` |
| model | `yolo11n.pt` |
| 입력 | H.264, 1280×720, 30 FPS, 30.0초, 900 frames |
| telemetry | synthetic |
| workflow total | 69.3644212초 |

### pipeline 결과

| 항목 | 결과 |
|---|---:|
| tracking records | 5,308 |
| unique track IDs | 107 |
| tracking model load | 2.2098572초 |
| tracking/render | 44.9636935초 |
| transcode | 6.7894204초 |
| tracking end-to-end | 51.7531139초 |
| end-to-end FPS | 17.3902579 |
| risk max | 0.9372351669 |
| synthetic telemetry rows | 296 |
| sync matched / unmatched | 5,198 / 110 |
| candidate frames | 56 |
| strict runs | 17 |
| merged runs before duration filter | 14 |
| final events | 2 |
| clips | 2 |

### event 요약

| event | 구간 | max score | primary | hard brake | clip 구간 |
|---|---|---:|---|---|---|
| `event_001` | 28.133333~28.833333초 | 0.9372351669 | car, track 664 | `true` | 26.133333~29.833333초 |
| `event_002` | 29.333333~29.933333초 | 0.9330165100 | car, track 684 | `false` | 27.333333~30.0초 |

Event 구간은 risk proxy가 sample용 threshold 0.70 이상인 frame candidate를 기존 Day 6 temporal 규칙으로 결합한 검토 후보다. 사고·충돌 발생 또는 실제 위험 상태를 입증하지 않는다.

## 15. SHA-256 integrity 결과

`run_summary.json`의 primary-before, primary-after, published hash를 대조했다. 아래 표의 primary SHA-256은 before와 after가 같고 published 값도 같다.

| 산출물 | primary SHA-256 | published SHA-256 | Match |
|---|---|---|---|
| annotated video | `501a2f36b89f5fcb6ad71423d9c2b428d4f1eec5a619288d5f735cdd1fda852f` | `501a2f36b89f5fcb6ad71423d9c2b428d4f1eec5a619288d5f735cdd1fda852f` | `True` |
| risk plot | `f405e1ecf9a830790756bf61d42e6322800ea7d4cf7b0104f76689d231e083ac` | `f405e1ecf9a830790756bf61d42e6322800ea7d4cf7b0104f76689d231e083ac` | `True` |
| telemetry plot | `838afefea5e3fa05e9c31ecb0ec95a6a5affcc6629365d07a87205307803df46` | `838afefea5e3fa05e9c31ecb0ec95a6a5affcc6629365d07a87205307803df46` | `True` |
| event 001 clip | `237c8d74445ac1fae8cf6478aaf623db0bb72f0a940c1e9c66a3e188e0d65496` | `237c8d74445ac1fae8cf6478aaf623db0bb72f0a940c1e9c66a3e188e0d65496` | `True` |
| event 002 clip | `1df114bf7ccb1fcacdeecc97966350807ebc8f97486b038c643a0b379e54a530` | `1df114bf7ccb1fcacdeecc97966350807ebc8f97486b038c643a0b379e54a530` | `True` |

다섯 쌍 모두 Match `True`다. 이는 같은 Day 8 run에서 published copy가 primary와 byte-identical했다는 증거이며 모델 정확도, 안전성 또는 전원 손실 시 트랜잭션 안전성을 의미하지 않는다.

## 16. 기존 Day 3 기준선과 Day 8 재실행 결과의 구분

기존 Day 3 baseline은 5,360 tracking rows와 108 unique track IDs다. Day 8 browser run의 5,308 rows와 107 IDs는 표준화 영상을 새로 만들고 `yolo11n.pt` PyTorch CPU 추론·tracking을 새로 수행한 결과다.

따라서 Day 8 결과에 Day 3 CSV와 행 단위 또는 track ID 단위 exact 동일성을 요구하지 않는다. 그 차이를 회귀 실패로 숨기거나 5,360행으로 바꾸지 않는다. Day 8의 검증 계약은 같은 Day 8 run 내부에서 다음 관계가 일치하는지 확인하는 것이다.

- standardized video metadata와 900-frame tracking 결과
- 5,308-row tracks, risk 및 synchronized CSV
- risk·telemetry graph의 source
- 900-row frame timeline과 2개 event
- `events.json`의 2개 event와 manifest의 2개 clip
- primary와 published 다섯 쌍의 SHA-256

성공한 직접 workflow smoke `day8_pytorch_smoke_20260804_01`과 실제 browser run은 서로 다른 run이지만, 동일한 새 Day 8 추론 결과 계약인 5,308 rows, 107 IDs, risk max, event 2개와 산출물 hash가 일치했다. 처리 시간은 실행마다 달라질 수 있으므로 browser run의 69.3644212초를 primary로 기록한다.

## 17. 알려진 문제와 미지원 기능

- Synthetic telemetry는 실제 센서 정확도 근거가 아니다.
- score threshold 0.70은 현재 30초 sample용 실험값이며 운영 안전 기준이 아니다.
- risk proxy는 실제 거리, TTC 또는 사고·충돌 확률이 아니다.
- detector false positive·false negative와 track ID fragmentation이 가능하다.
- PyTorch CPU end-to-end 17.3902579 FPS는 입력 30 FPS 실시간 처리에 미달한다.
- Day 7 ONNX detector benchmark는 메모리 이미지의 detection 단계 benchmark다. Day 8의 video decode, tracking, rendering, transcode, risk, telemetry, clip을 포함한 full pipeline 처리시간과 직접 비교하지 않는다.
- ONNX full-video tracking backend는 구현하지 않았다.
- DirectML video backend는 지원하지 않는다.
- 외부 hosting, public share, streaming, 다중 사용자 배포를 검증하지 않았다.
- Queue concurrency는 1이며 부하·장시간 안정성·동시 사용자 성능을 검증하지 않았다.
- 실제 sensor telemetry, event ground truth, Precision/Recall, 다양한 영상·날씨·도로 일반화를 검증하지 않았다.
- Windows browser 연결 종료 시 `WinError 10054` 2회는 현재 비차단 관찰 사항이다.
- 강제 종료로 생긴 불완전 workspace의 자동 정리·resume은 구현하지 않았다.

## 18. 재현 명령

### 로컬 Gradio 실행

저장소 root의 Windows PowerShell에서 실행한다.

```powershell
# DashGuard repository root에서 실행
& ".\.venv\Scripts\python.exe" ".\scripts\day8_gradio_demo.py"
```

기본 접속 주소는 `http://127.0.0.1:7860`이다. 외부 interface에 bind하지 않고 `share=False`로 실행한다. 새 분석은 새 run ID와 workspace를 만들며 기존 run을 덮어쓰지 않는다.

### 검증 명령

```powershell
& ".\.venv\Scripts\python.exe" -m pytest `
  ".\tests\test_day8_workflow.py" `
  ".\tests\test_day8_presentation.py" `
  ".\tests\test_day8_gradio_app.py" -q

& ".\.venv\Scripts\python.exe" -m pytest -q
& ".\.venv\Scripts\python.exe" -m pip check
& ".\.venv\Scripts\python.exe" ".\scripts\day8_gradio_demo.py" --help
```

`requirements-lock.txt` exact 확인은 현재 environment의 `pip freeze --all` 출력을 줄바꿈 정규화 후 lock과 byte-for-byte 비교한다. 실제 browser run 재현에는 FFmpeg/FFprobe와 `libx264`, `yolo11n.pt`, 유효한 최대 30초 영상이 필요하다.

## 19. 적용 범위와 한계

이번 검증은 한 Windows PC, Python 3.11, Gradio 6.22.0, `yolo11n.pt`, PyTorch CPU와 한 개의 30초 H.264 입력에 대한 로컬 통합 데모 검증이다. 자동 테스트, localhost HTTP 응답, 실제 브라우저 재생, 산출물 JSON 계약과 SHA-256 무결성을 각각 분리해 확인했다.

결과는 같은 Day 8 run 내부의 video, CSV, graph, event, clip 연결과 로컬 표시가 동작한다는 근거다. 실제 센서 정확도, 물리 거리·TTC, 사고 확률, 안전성, 실시간 30 FPS, ONNX/DirectML full pipeline, 외부 배포 또는 다중 사용자 운영을 입증하지 않는다. `outputs`의 run, 모델, upload 임시 파일과 불완전 workspace는 로컬 검증 산출물이며 Git commit 대상이 아니다.
