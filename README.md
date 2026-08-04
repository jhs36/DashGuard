# DashGuard

DashGuard는 대시캠 영상에서 도로 객체를 탐지·추적하고, 시간축에 맞춘 차량 주행 정보와 결합하여 주의가 필요한 구간을 시각화하는 프로토타입이다.

## 1. 해결하려는 문제

대시캠 영상을 처음부터 끝까지 사람이 직접 확인하면 시간이 오래 걸린다.

DashGuard는 영상 속 차량, 보행자 등 주요 도로 객체의 움직임과 차량 주행 상태를 분석하여 검토 우선순위가 높은 구간을 찾을 수 있도록 돕는 것을 목표로 한다.

이 프로젝트의 위험 점수는 실제 사고 확률이나 법적 판단을 의미하지 않는다. 영상과 제한된 주행 정보에서 계산한 설명 가능한 휴리스틱 지표이다.

## 2. 주요 입력

- MP4 형식의 대시캠 영상
- 선택적 텔레매틱스 CSV
  - 영상 또는 센서 타임스탬프
  - 차량 속도
  - 종방향 또는 횡방향 가속도
  - 사용 가능한 경우 자이로 또는 방향 정보

실제 텔레매틱스가 없는 영상에서는 데모와 파이프라인 검증을 위해 합성 데이터를 사용할 수 있다.

## 3. 주요 출력

- 객체 탐지·추적 결과가 표시된 주석 영상
- 프레임 또는 시점별 객체·주행 정보
- 규칙 기반 위험 점수 또는 주의 점수
- 검토가 필요한 주요 구간 목록
- 처리 시간과 모델 성능에 대한 간단한 실행 보고서

## 4. 목표 처리 흐름

1. FFmpeg를 이용한 영상 입력 검사 및 표준화
2. YOLO 기반 도로 객체 탐지
3. ByteTrack 기반 프레임 간 객체 추적
4. 영상 타임스탬프와 텔레매틱스 동기화
5. 객체 움직임과 차량 상태를 이용한 휴리스틱 위험 점수 계산
6. 결과 영상과 이벤트 요약 생성
7. Gradio 기반 데모 화면에서 입력과 결과 확인
8. 가능한 경우 ONNX Runtime 기반 추론 성능 비교

## 5. MVP 범위

최소 기능 제품에서는 다음을 구현한다.

- 하나의 짧은 대시캠 샘플 영상 처리
- 차량과 보행자를 포함한 주요 객체 탐지
- 객체별 추적 ID 유지
- 영상과 텔레매틱스 시간축 동기화
- 설명 가능한 규칙 기반 위험 점수
- 결과 영상과 이벤트 요약 출력
- 재현 가능한 설치 및 실행 명령

## 6. 비목표

이 프로젝트는 다음 기능을 목표로 하지 않는다.

- 실제 사고 발생 확률 예측
- 보험, 법률 또는 운전자 과실 판단
- 카메라 보정 없이 정확한 실제 거리 측정
- 센서 보정 없이 정확한 TTC(Time to Collision) 계산
- 자율주행 차량 제어
- 차량용 실시간 임베디드 제품
- 모든 날씨와 도로 환경에서의 일반화 보장
- 안전 인증 또는 상용 수준의 신뢰성 보장

## 7. 기술 스택 초안

- Python 3.11
- OpenCV
- FFmpeg
- Ultralytics YOLO
- ByteTrack
- pandas / NumPy
- Gradio 6.22.0
- ONNX Runtime

기술 스택은 실제 구현 및 호환성 검증 결과에 따라 변경될 수 있다.

## 8. 개발 환경 기준

- 운영체제: Windows
- 셸: Windows PowerShell
- 기준 Python: 3.11
- 최초 추론 기준: CPU
- GPU 가속: 후속 단계에서 선택적으로 검증
- 프로젝트 가상환경: `.venv`

## 9. 현재 상태

Day 8 Gradio 로컬 통합 데모까지 구현을 완료했다. 현재 데모는 기존 영상 표준화, PyTorch CPU 탐지·추적, risk proxy, telemetry 동기화, event 후보와 clip 생성을 하나의 격리된 run workspace에서 실행한다.

검증 범위와 실제 30초 browser run 결과는 `docs/day8_gradio_integration_validation.md`에 기록되어 있다.

## 10. Day 8 Gradio 데모 실행

기준 환경은 Python 3.11과 Gradio 6.22.0이다. 저장소 root의 Windows PowerShell에서 실행한다.

```powershell
Set-Location "E:\Projects\DashGuard"
& "E:\Projects\DashGuard\.venv\Scripts\python.exe" ".\scripts\day8_gradio_demo.py"
```

기본 주소는 `http://127.0.0.1:7860`이며 localhost에만 bind하고 `share=False`로 실행한다.

- 입력 영상은 최대 30초다.
- backend는 PyTorch CPU, model은 `yolo11n.pt`로 고정된다.
- telemetry CSV는 선택 입력이며, 미입력 시 deterministic synthetic telemetry를 사용한다.
- 주요 출력은 주석 영상, risk·telemetry PNG, event 표, 첫 event clip과 전체 clip 목록이다.
- 각 결과는 `outputs/day8/runs/<run_id>/`에 격리되며 브라우저에는 `published/` copy만 제공한다.

Risk proxy는 실제 거리, TTC 또는 사고 확률이 아니다. Synthetic telemetry는 실제 센서 정확도 근거가 아니며, PyTorch CPU full pipeline은 30 FPS 실시간 처리를 보장하지 않는다. ONNX full-video tracking, DirectML, 외부 hosting·streaming 및 다중 사용자 배포는 지원하거나 검증하지 않았다.

## 11. 로컬 개발환경 시작

### 가상환경 생성

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

FFmpeg는 Python 가상환경과 별도로 설치해야 한다.

```powershell
winget install --id Gyan.FFmpeg -e --source winget
ffmpeg -version
ffprobe -version
```

### Day 1 환경 테스트

```powershell
python -m pip check
python -m pytest -q
```

### YOLO 이미지 스모크 테스트 준비

검증용 예제 이미지를 내려받는다.

```powershell
curl.exe -L "https://ultralytics.com/images/bus.jpg" -o ".\data\raw\bus.jpg"
```

YOLO nano 가중치를 내려받아 `models` 폴더로 이동한다.

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); print('model_load: OK')"
Move-Item .\yolo11n.pt .\models\yolo11n.pt
```

CPU 추론을 실행한다.

```powershell
python .\scripts\yolo_smoke.py
```

정상 실행 기준:

- `detections`가 1 이상
- `class_counts` 출력
- `outputs/yolo_smoke/bus_result.jpg` 생성
- `yolo_smoke: OK` 출력

원본 이미지, 모델 가중치, 생성 결과는 Git에 포함하지 않는다.

## 12. Day 1 검증 기준선

- 운영체제: Windows
- 개발 도구: VS Code
- Python: 3.11.5
- PyTorch: CPU 빌드
- FFmpeg: 시스템 설치
- 기준 탐지 모델: YOLO11 nano
- 기준 장치: CPU
- GPU 가속: 후속 단계에서 선택적으로 검증
