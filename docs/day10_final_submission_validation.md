# DashGuard Day 10 최종 제출 검증

## 1. 목적과 범위

이 문서는 DashGuard의 Day 10 최종 제출 준비 상태를 재현 가능한 근거와 함께 동결한다. 검증 범위는 source repository의 설치 가능성, 외부 asset이 없는 자동 검증, localhost Gradio 실제 smoke, output 격리와 무결성, Git 포함 범위, 문서·dependency lock 정리, 공개 전 라이선스 판단이다.

Day 10에서는 새 기능이나 Python 코드를 추가하지 않았다. 제출 정리는 `.gitignore`, `README.md`, 기존 validation 문서의 경로 표현, `requirements-lock.txt`와 이 문서를 포함한 제출 문서에 한정한다. Risk 계산, tracking, telemetry, event 생성과 Day 9 평가의 의미는 변경하지 않는다.

이 문서에서 다음 세 결과는 서로 다른 범위로 읽어야 한다.

- **clean clone 자동 검증**: model, 도로 영상과 기존 output 없이 source·test·package 계약을 확인한 결과
- **실제 Gradio smoke**: 별도로 준비한 local model과 30초 입력을 사용해 한 PC의 PyTorch CPU 경로를 끝까지 실행한 결과
- **Day 7·9 결과**: 각각 두 정적 이미지의 ONNX CPU benchmark와 single-source small-sample pilot의 제한된 평가 결과

## 2. 시작 Git 기준선

Day 10 read-only audit를 시작한 시점의 기준선은 다음과 같다. 이후 제출 정리 변경은 이 기준선 위의 작업 트리 변경으로 관리했다.

| 항목 | 시작 상태 |
|---|---|
| branch | `main` |
| `HEAD` | `f2dabfb` |
| `origin/main` | `f2dabfb` |
| working tree | clean |
| Python | 3.11.5 |
| FFmpeg / ffprobe | 8.1.2 / 8.1.2 |
| detector weight | local `yolo11n.pt`; Git 미포함 |
| project-level `LICENSE` | 없음 |
| Ruff package | 미설치 |
| `pyproject.toml` Ruff 설정 | 없음 |

기준 commit은 검증 출발점을 식별하기 위한 값이다. Day 10 문서와 제출 정리 diff가 아직 commit되지 않은 상태에서는 현재 작업 트리가 clean일 필요는 없으며, 기존 변경을 되돌리거나 기준 commit으로 강제 복원하지 않는다.

## 3. read-only audit 결과

Git index와 repository 구성을 변경하지 않고 먼저 확인한 결과는 다음과 같다.

| 감사 항목 | 결과 | 판정 |
|---|---|---|
| tracked video/model/archive binary | 없음 | PASS |
| `outputs/`의 tracked 파일 | `outputs/.gitkeep`만 존재 | PASS |
| `review.zip`, `review.7z` ignore 여부 | 기존 규칙으로는 제외되지 않음 | 수정 필요 |
| 기존 validation 문서의 사용자 절대경로 | Day 5, Day 6, Day 8에서 확인 | 수정 필요 |
| lock의 DashGuard editable 항목 | Day 7 commit 기반 항목 잔존 | 수정 필요 |
| CPU PyTorch package index | `requirements.txt`에 존재 | lock에도 명시적으로 보존 필요 |
| project-level `LICENSE` | 없음 | 공개 정책 결정 필요 |
| Ruff 사용 근거 | package·설정 모두 없음 | 미도입 상태를 명시 |

추가로 model, video, dataset, 생성 output은 source 제출물과 분리해야 한다는 기존 정책을 확인했다. Audit 자체에서는 파일 삭제, dependency 설치 또는 source 수정 작업을 수행하지 않았다.

## 4. 제출 blocker와 해결 내용

| 제출 blocker 또는 위험 | 해결 내용 | 결과 |
|---|---|---|
| review archive가 실수로 stage될 수 있음 | `.gitignore`에 `*.zip`, `*.7z` 추가 | 해결 |
| tracked 문서가 특정 사용자의 로컬 위치를 노출함 | Day 5, Day 6, Day 8 명령을 repository root 기준 PowerShell 상대경로로 변경 | 해결 |
| `requirements.txt`만 설치하면 `src/` layout package를 찾지 못함 | README 설치 절차에 `python -m pip install -e .`와 이유 추가 | 해결 |
| lock이 과거 Day 7 Git commit의 editable DashGuard를 가리킴 | 해당 editable 항목 제거 | 해결 |
| CPU PyTorch wheel source가 lock에서 불명확함 | `requirements-lock.txt` 첫 줄에 CPU PyTorch extra index 유지 | 해결 |
| fresh clone에 model·입력·기존 output이 있다고 오해할 수 있음 | README와 본 문서에 별도 local asset 준비 조건 명시 | 해결 |
| 배포 권한을 확인하지 않은 영상·파생 frame이 포함될 수 있음 | 외부 배포물에 자동 포함하지 않는 정책 명시 | 해결 |
| project-level license를 임의로 선택할 위험 | 공개 의도와 별도 법적 검토 전까지 추가하지 않음 | 의도적 보류 |

마지막 항목은 source-only private 제출의 기능 blocker가 아니라 공개 배포 전에 해소해야 할 별도 의사결정이다.

## 5. clean clone 전략

검증 중인 작업 트리의 local model, 영상, virtual environment와 과거 `outputs/`가 결과에 섞이지 않도록 별도의 clean clone을 사용했다. Clone은 시작 기준선에서 만들고 fresh virtual environment를 생성했다.

검증 순서는 다음과 같다.

1. clone의 branch와 commit을 확인한다.
2. `yolo11n.pt`, 실제 도로 영상과 과거 output이 없음을 확인한다.
3. Python 3.11.5 fresh virtual environment를 만든다.
4. `requirements.txt`만 설치한 상태의 동작을 확인한다.
5. `python -m pip install -e .` 전후의 package import와 Gradio CLI 도움말을 비교한다.
6. 외부 asset 없이 `pip check`, pytest와 compileall을 실행한다.
7. 별도로 준비한 local model과 local MP4로 localhost Gradio smoke를 실행한다.
8. smoke 후 Git 상태와 output 포함 범위를 다시 확인한다.

Fresh clone의 초기 asset 상태는 다음과 같았다.

- `yolo11n.pt` 없음
- 실제 도로 영상 없음
- `outputs/`에는 `.gitkeep`만 존재
- 기존 run workspace 없음

자동 test와 실제 영상 smoke는 같은 의미가 아니다. 자동 test는 외부 asset 없는 source 재현성을 확인했고, 실제 smoke는 Git 밖에서 준비한 asset으로 전체 local workflow를 확인했다.

## 6. requirements 설치 결과

Python 3.11.5를 사용한 fresh virtual environment 생성에 성공했다. 이어서 다음 설치와 dependency 검사를 수행했다.

```powershell
py -3.11 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r ".\requirements.txt"
& ".\.venv\Scripts\python.exe" -m pip check
```

결과는 다음과 같다.

| 항목 | 결과 |
|---|---|
| fresh virtual environment | 생성 성공 |
| `python -m pip install -r requirements.txt` | 성공 |
| `pip check` | `No broken requirements found.` |
| CPU PyTorch index | `--extra-index-url https://download.pytorch.org/whl/cpu` 유지 |

이 단계는 third-party dependency가 설치된다는 사실을 확인한다. `src/` layout인 DashGuard project 자체의 설치까지 의미하지는 않는다.

## 7. editable install 필요성 재현

`requirements.txt`만 설치한 직후에는 다음 두 동작이 실패했다.

| 확인 | 결과 |
|---|---|
| `import dashguard` | 실패 |
| `day8_gradio_demo.py --help` | 실패 |

원인은 dependency 누락이 아니라 `src/` layout의 project package 자체가 environment에 설치되지 않은 것이었다. 다음 명령을 추가로 실행했다.

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -e "."
```

그 뒤의 결과는 다음과 같다.

| 확인 | 결과 |
|---|---|
| `dashguard` import | 성공 |
| Editable project location | clean clone |
| `day8_gradio_demo.py --help` | 성공 |
| Gradio localhost server launch | 성공 |

따라서 fresh clone 설치 순서는 `requirements.txt` 설치 후 repository root에서 editable install을 수행하는 것으로 문서화했다. `requirements-lock.txt`에 과거 Git commit의 DashGuard editable 항목을 두는 방식으로 이 단계를 대체하지 않는다.

## 8. 외부 asset 없는 pytest·compileall·pip check 결과

Clean clone에서 model, 도로 영상과 기존 output이 없는 상태로 자동 검증을 수행했다.

| 검증 | 결과 |
|---|---:|
| pytest | `641 passed in 17.61s` |
| compileall | exit code `0` |
| pip check | 통과 |
| 외부 model 필요 여부 | 불필요 |
| 실제 도로 영상 필요 여부 | 불필요 |
| 기존 `outputs/` 필요 여부 | 불필요 |
| 검증 후 Git 상태 | clean |

이는 test suite가 local binary나 과거 run에 암묵적으로 의존하지 않는다는 근거다. 반대로 pretrained detector의 실제 영상 품질이나 full pipeline의 외부 asset 없는 실행을 의미하지 않는다.

Day 10 제출 정리 변경이 있는 현재 작업 트리에서도 회귀 검사를 다시 수행했다.

| 검증 | 결과 |
|---|---:|
| pip check | 통과 |
| pytest | `641 passed in 22.12s` |
| compileall | exit code `0` |
| `git diff --check` | exit code `0` |

두 pytest 시간은 서로 다른 실행의 wall time이며 성능 benchmark로 사용하지 않는다.

## 9. Ruff 감사 결과

Ruff package는 검증 environment에 설치되어 있지 않았고 `pyproject.toml`에도 `[tool.ruff]` 설정이 없었다. 그러므로 Day 10에서 Ruff 결과를 임의로 PASS라고 기록하거나 새 lint dependency·설정을 도입하지 않았다.

현재의 정적 확인 범위는 Python compileall과 `git diff --check`다. Ruff를 향후 도입한다면 rule set, 제외 경로, 기존 issue baseline과 CI 동작을 별도 변경으로 합의하고 검증해야 한다.

## 10. 실제 Gradio localhost smoke

별도 clean clone에 local `yolo11n.pt`와 Git 밖의 최대 30초 MP4를 준비해 PyTorch CPU full-video workflow를 localhost에서 실행했다.

| 항목 | 실제 결과 |
|---|---:|
| run status | `succeeded` |
| backend | `pytorch_cpu` |
| 입력 codec / 해상도 | H.264 / 1280×720 |
| frame rate / frames | 30 FPS / 900 |
| 입력 길이 | 30.0초 |
| telemetry source | `synthetic` |
| total processing | 126.5698975초 |
| validation stage | 약 48.92초 |
| tracking stage | 약 58.01초 |
| tracking records | 5,256 |
| unique track IDs | 101 |
| tracking render FPS | 약 18.65 |
| end-to-end analysis FPS | 약 16.35 |
| risk proxy max | 약 0.940389 |
| telemetry matched / unmatched | 5,153 / 103 |
| event candidates | 2 |
| event clips | 2 |
| warnings | 0 |

브라우저에서 다음 presentation 항목을 직접 확인했다.

- annotated video 표시와 재생 성공
- risk proxy plot 표시 성공
- telemetry plot 표시와 `synthetic` 표기 성공
- event table과 clip 목록 표시 성공

30초 입력의 total processing이 약 126.57초였으므로 이 결과는 실시간 동작의 근거가 아니다. FPS 필드는 각 stage가 기록한 제한된 처리 지표이며 전체 wall time과 혼합해 30 FPS 실시간 처리로 해석하지 않는다. 또한 이는 단일 PC, 단일 입력, `yolo11n.pt`, PyTorch CPU 조건의 smoke 결과일 뿐 일반 성능값이 아니다.

## 11. output 격리·published integrity·Git 제외 상태

Smoke run은 기존 결과를 덮어쓰지 않는 고유 run workspace를 생성했다. UI에는 primary artifact와 분리된 `published/` copy만 노출했으며, published artifact와 대응하는 primary artifact의 SHA-256이 일치했다.

| 확인 항목 | 결과 |
|---|---|
| 고유 run workspace | 생성 성공 |
| 기존 run 덮어쓰기 | 없음 |
| primary / published 경로 분리 | 확인 |
| 대응 artifact SHA-256 | 일치 |
| smoke warnings | 0 |
| smoke 후 clean clone Git 상태 | clean |
| tracked `outputs/` 항목 | `outputs/.gitkeep`만 존재 |

Model weight, ONNX artifact, video, dataset, generated CSV·JSON·PNG·MP4와 기타 output binary는 local 검증 산출물이며 Git에 포함하지 않는다. `*.zip`과 `*.7z` ignore 규칙도 추가해 review/export archive의 실수 stage를 방지했다. Ignore 규칙은 배포 권한을 부여하지 않으며, 외부 전달 전에는 별도의 provenance·license 확인이 필요하다.

## 12. README와 lock 갱신 내용

README에는 fresh clone 사용자가 실제로 필요한 설치와 asset 조건을 추가했다.

- `python -m pip install -r requirements.txt` 뒤에 `python -m pip install -e .` 실행
- `src/` layout이므로 editable install 전에는 `dashguard` package를 import할 수 없다는 설명
- fresh clone에는 `yolo11n.pt`, 실제 도로 영상과 기존 output이 없다는 설명
- 실제 분석에는 model과 Git 밖의 최대 30초 local MP4가 필요하다는 설명
- Gradio server 자체는 model 없이 열릴 수 있지만 실제 분석에는 유효한 model이 필요하다는 설명

`requirements-lock.txt`에는 다음을 반영했다.

- 첫 줄에 `--extra-index-url https://download.pytorch.org/whl/cpu` 유지
- 과거 Day 7 commit을 가리키던 DashGuard editable 항목 제거
- 검증 environment의 현재 exact dependency baseline 반영

Lock은 외부 dependency 환경의 기준이며 project package 설치 명령을 대신하지 않는다. Fresh clone에서는 README에 따라 별도로 `python -m pip install -e .`를 실행한다.

## 13. 공개 범위와 라이선스 결정

현재 repository는 **Private**다. Repository root에는 project-level `LICENSE`가 없다. 사용자의 공개 의도, 적용할 source license와 포함 asset의 조건에 대한 별도 법적 검토가 끝나기 전에는 project-level `LICENSE`를 임의로 추가하지 않는다.

공개·외부 전달 범위는 다음 원칙으로 제한한다.

- 실제 도로 영상의 원본 URL과 license snapshot이 repository에 없으므로 해당 영상과 파생 frame을 외부 배포물에 자동 포함하지 않는다.
- `yolo11n.pt`, 파생 ONNX model, 도로 video, dataset, review archive와 `outputs/` binary를 Git에 포함하지 않는다.
- 문서의 attribution 기록만으로 원본 또는 파생물의 재배포 권한을 단정하지 않는다.
- Fresh clone 사용자는 model과 최대 30초 local MP4를 별도로 준비해야 한다.
- Gradio server는 model 없이 열릴 수 있지만 실제 분석 실행에는 local model이 필요하다.
- Portfolio image나 demo video에 제3자 영상 frame을 넣기 전에도 원본 URL, license snapshot, 허용 범위와 필요한 attribution을 다시 확인한다.

현재 Private 상태는 자동으로 모든 외부 asset 사용·배포 문제를 해결하지 않는다. 외부 공개 전 checklist는 [Day 10 demo·portfolio·interview pack](day10_demo_portfolio_interview_pack.md)에 별도로 제공한다.

## 14. 최종 제출 blocker 판정

| 제출 범위 | 판정 | 근거 |
|---|---|---|
| Private repository의 source·문서 제출 | **PASS** | clean clone 설치, 641 tests, compileall, pip check, 실제 localhost smoke와 Git 제외 상태 확인 |
| local demo 재현 | **조건부 PASS** | 사용자가 `yolo11n.pt`와 최대 30초 local MP4를 별도 준비해야 함 |
| model·video·dataset·output binary를 포함한 제출 | **제외** | Git 포함 대상이 아니며 배포 조건도 별도 확인 필요 |
| public source release | **HOLD** | project-level license 선택과 공개 의도 확인 필요 |
| 도로 영상·파생 frame을 포함한 public portfolio | **HOLD** | 원본 URL과 license snapshot이 repository에 없음 |

따라서 현재 정의된 **Private source-only 최종 제출을 막는 기술 blocker는 없다**. 다만 public release와 제3자 media 배포는 완료로 판정하지 않으며, license·provenance 확인 전에는 별도 보류한다.

## 15. 남은 한계와 향후 개선

- Risk proxy는 bbox 기반의 설명 가능한 휴리스틱이다. 실제 거리, TTC, 사고 확률 또는 물리적 안전성을 나타내지 않는다.
- 실제 GPS·IMU·OBD telemetry를 사용하지 않았고 deterministic synthetic telemetry로 schema·sync·표시를 검증했다.
- Day 9은 독립 원본 1개에서 만든 비중복 segment 3개, 각 27.5초인 총 82.5초의 single-source small-sample pilot이다.
- Day 9 ground-truth positive가 0개이므로 Recall은 `null`이며 positive detection ability를 평가하지 못했다.
- Day 9 prediction 7개는 사람 검토에서 모두 false positive였다. 원인 분류는 risk feature limitation 3개, detection artifact 2개, normal motion / ego-motion 2개다.
- 검토한 5개 ID 후보에서는 확정 ID switch가 관찰되지 않았지만, 3개 후보에서 총 21개의 duplicate/fragmentation episode가 관찰됐다. 공식 IDF1 또는 HOTA 평가가 아니다.
- BDD100K는 20장의 annotation 변환과 시각 검증만 수행했으며 fine-tuning이나 model 성능 평가를 수행하지 않았다.
- Day 7 ONNX Runtime CPU의 image-to-detections P50 speedup은 약 2.03배였지만, `yolo11n.pt`, FP32, fixed 640×640, 이미지 2장, Ryzen 7 5700X3D, sequential single-stream latency 조건에 한정된다.
- ONNX는 detector consistency와 제한된 정적 이미지 CPU benchmark에만 사용했고 full-video Gradio backend에는 연결하지 않았다.
- 실제 Gradio smoke는 한 PC·한 input·PyTorch CPU 조건이며 30초 입력에 약 126.57초가 걸려 실시간이 아니다.
- 다중 사용자, 외부 network 배포, 보안 hardening, 운영 모니터링과 production 운영은 검증하지 않았다.

우선 개선 순서는 positive·negative를 함께 포함한 다중 독립 원본 평가, false-positive 원인별 ablation, ego-motion·lateral motion 처리 검토, 실제 telemetry calibration, 공식 MOT dataset 평가, ONNX full-video 통합의 별도 설계·검증이다.

## 16. 최종 재검증 명령

다음 명령은 repository root의 Windows PowerShell에서 실행한다. 모든 경로는 repository-relative이며 model, video 또는 기존 output을 stage하지 않는다.

### 16.1 Git 기준선과 포함 범위

```powershell
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main

git ls-files | Where-Object { $_ -match '\.(pt|onnx|mp4|mov|avi|mkv|zip|7z)$' }
git ls-files outputs
Test-Path ".\LICENSE"
```

기대 결과는 binary 검색 출력이 없고, `git ls-files outputs`에는 `outputs/.gitkeep`만 있으며, 현재 기준에서는 `Test-Path`가 `False`인 것이다. Day 10 제출 변경이 commit되기 전에는 `git status`에 의도한 문서·설정 diff가 표시될 수 있다.

### 16.2 Fresh environment 설치와 자동 검증

```powershell
py -3.11 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r ".\requirements.txt"
& ".\.venv\Scripts\python.exe" -m pip install -e "."
& ".\.venv\Scripts\python.exe" -m pip check
& ".\.venv\Scripts\python.exe" -m pytest -q
& ".\.venv\Scripts\python.exe" -m compileall -q ".\src" ".\scripts" ".\tests"
git diff --check
```

### 16.3 Package와 Gradio entry point

```powershell
& ".\.venv\Scripts\python.exe" -c "import dashguard; print('dashguard import: OK')"
& ".\.venv\Scripts\python.exe" ".\scripts\day8_gradio_demo.py" --help
```

### 16.4 Ruff 설치·설정 감사

```powershell
Get-Command ruff -ErrorAction SilentlyContinue
Select-String -Path ".\pyproject.toml" -Pattern '^\[tool\.ruff'
```

현재 동결 상태에서는 두 명령 모두 Ruff가 도입됐다는 근거를 출력하지 않는 것이 기대 결과다.

### 16.5 Local Gradio 실행

Git 밖에서 적법하게 준비한 최대 30초 MP4와 유효한 local model을 사용한다. Model을 repository root의 기본 위치에 준비했다면 다음과 같이 server를 연다.

```powershell
& ".\.venv\Scripts\python.exe" ".\scripts\day8_gradio_demo.py"
```

브라우저에서는 실행 시작 장면과 실제 완료 결과를 구분한다. 처리 시간을 기다리지 않고 결과로 편집 전환한 demo를 real-time 실행처럼 표현하지 않는다.

## 17. 완료 판정

Day 10에서 요구한 source-only 제출 감사와 문서화는 완료됐다.

- 시작 기준선, clean clone과 current-tree 회귀 결과 기록 완료
- editable install 필요성 재현과 README 반영 완료
- tracked binary·output 범위 확인 및 archive ignore 보완 완료
- 사용자 절대경로를 repository-relative 명령으로 정리 완료
- 실제 localhost Gradio smoke, output 격리와 SHA-256 integrity 기록 완료
- Day 7·9의 수치와 적용 한계를 분리해 기록 완료
- Private repository와 public release의 license 판단을 분리 완료
- model, video, dataset과 output binary를 제출 대상에서 제외 완료

최종 판정은 **Private source-only 제출 PASS, public media·license 공개 결정 HOLD**다.
