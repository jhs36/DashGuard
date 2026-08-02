# Day 6 Telemetry Sync and Event Clip Validation

## 1. 목적

이 문서는 Day 5의 **영상 기반 접근 위험 점수**(video-based approach risk proxy)를 변경하지 않고, **synthetic telemetry**를 별도의 **telemetry evidence**로 시간 동기화한 뒤 frame timeline에서 **고위험 후보 이벤트**(event candidate)를 생성하고 FFmpeg clip으로 추출한 Day 6 통합 검증 결과를 재현 가능한 형태로 기록한다.

Day 6 파이프라인에서 `total_score`는 계속 video-only **risk proxy**이며, telemetry 값은 점수에 포함되지 않는다.

## 2. 범위와 비목표

### 포함 범위

- deterministic synthetic telemetry
- schema validation
- explicit offset
- nearest timestamp sync
- tolerance miss
- frame representative timeline
- experimental event candidate
- temporal merge
- event JSON
- FFmpeg H.264 clip
- unit/integration test
- 실제 30초 영상 smoke와 육안 재생

### 비목표

- 실제 GPS·IMU·OBD 정확도 검증
- 실제 운전자 행동 모델 검증
- 물리 거리 및 TTC 산출
- 사고·충돌 가능성의 확률 모델
- Precision/Recall
- event ground truth 평가
- Re-ID
- detector 또는 tracker 개선
- score overlay 추가

## 3. 입력

검증에 사용한 입력 파일은 다음과 같다.

- `outputs/day5/csv/day5_risk_scores.csv`
- `outputs/day3/videos/day3_baseline.mp4`
- `outputs/day6/telemetry/day6_synthetic_telemetry_usercheck_20260801.csv`

### 기준 영상

- 해상도: 1280×720
- FPS: 30
- frames: 900
- duration: 30초
- bbox, class, confidence, track ID, trajectory overlay 포함
- risk score 숫자 overlay 없음
- 무음

### Day 5 입력

- rows: 5,360
- columns: 30
- tracks: 108
- `total_score`는 video-only risk proxy

## 4. 처리 파이프라인

```text
Day 5 risk CSV
→ synthetic telemetry
→ schema validation
→ explicit offset
→ nearest sync
→ tolerance miss preservation
→ frame representative timeline
→ threshold candidate
→ temporal merge
→ minimum duration
→ event JSON
→ FFmpeg clips
```

telemetry는 별도의 telemetry evidence로 보존된다. `speed_kmh`, `accel_longitudinal`, `accel_lateral`, `brake`, `hard_brake`를 포함한 telemetry 필드는 `total_score`에 합산되지 않으며, Day 5의 점수 수식도 변경하지 않는다.

## 5. Synthetic telemetry 계약과 실제 결과

| 항목 | 설정 또는 결과 |
|---|---:|
| `frequency` | 10Hz |
| `duration` | 0.0~30.0초 |
| rows | 296 |
| `source` | `synthetic` |
| missing interval | 12.0~12.4초 |
| hard-brake evidence interval | 28.2~29.1초 |
| hard-brake telemetry rows | 10 |

hard-brake 조건은 다음과 같다.

```text
brake == 1 and accel_longitudinal <= -3.0
```

이 조건은 Day 6 통합 테스트용 telemetry evidence를 결정적으로 생성하기 위한 조건이다. 실제 센서나 운행 상태에 대한 안전 판단 기준이 아니다.

## 6. 동기화 계약과 실제 결과

### 동기화 계약

- `offset_sec`: 0.0
- `tolerance_sec`: 0.05
- nearest timestamp 사용
- 절대 시간 차이가 동률이면 더 이른 telemetry timestamp 선택
- unlimited forward fill/backfill 미사용
- tolerance 초과 시 telemetry 필드를 결측으로 보존하고 `telemetry_matched=False` 기록

### 실제 결과

| 항목 | 결과 |
|---|---:|
| input risk rows | 5,360 |
| output rows | 5,360 |
| output columns | 40 |
| matched rows | 5,253 |
| unmatched rows | 107 |
| maximum matched absolute delta | 0.0333330000000025초 |
| hard-brake matched rows | 94 |
| `telemetry_source` | `synthetic` |
| infinity | 0 |
| 기존 columns | 기존 30열 순서 exact 보존 |
| 기존 values | 기존 30열 값 exact CSV round-trip 보존 |
| `total_score` | exact 보존 |

CSV float round-trip 확인 중 `bbox_area` 62행에서 기본 CSV 직렬화에 따른 1 ULP 수준의 차이가 발견되었다. 최대 절대 차이는 `5.820766091346741e-11`이었다. 기존 CSV token prefix를 보존하고 새 float 필드에는 17자리 출력 정책을 적용해 해결했으며, 점수 또는 동기화 수식은 변경하지 않았다.

## 7. Event 설정

다음 값은 현재 30초 sample의 통합 검증을 위한 실험 설정이다. 산업 표준 또는 안전 기준이 아니다.

| argument | 값 |
|---|---:|
| `score_threshold` | 0.70 |
| `minimum_event_duration_sec` | 0.15 |
| `merge_gap_sec` | 0.10 |
| `pre_roll_sec` | 2.0 |
| `post_roll_sec` | 1.0 |

frame representative timeline은 frame마다 한 행을 선택하며 tie-break 순서는 다음과 같다.

1. `total_score` descending
2. `timestamp_sec` ascending
3. `track_id` ascending
4. source row ascending

track ID를 Re-ID하거나 서로 병합하지 않았다. timeline의 대표 행 선택과 시간 구간 병합은 객체 ID 병합과 별개의 처리다.

## 8. Timeline과 Event 실제 결과

| 항목 | 결과 |
|---|---:|
| input rows | 5,360 |
| timeline rows | 900 |
| timeline columns | 42 |
| candidate frames | 60 |
| strict runs | 18 |
| merged runs before duration filter | 17 |
| final events | 2 |

### `event_001`

- frame: 844~864
- event time: 28.133333~28.833333초
- duration: 약 0.700000초
- candidate frames: 20
- `max_total_score`: 0.9403815803991128
- `primary_track_id`: 697
- `primary_class`: `car`
- `hard_brake`: `true`
- `telemetry_source`: `synthetic`
- clip: 26.133333~29.833333초

### `event_002`

- frame: 881~897
- event time: 29.366667~29.933333초
- duration: 약 0.566666초
- candidate frames: 17
- `max_total_score`: 0.9329814675758142
- `primary_track_id`: 722
- `primary_class`: `car`
- `hard_brake`: `false`
- `telemetry_source`: `synthetic`
- clip: 27.366667~30.0초

`hard_brake`는 event trigger가 아니라 event 구간에 함께 기록된 별도 telemetry evidence다. trigger reason은 다음 한 가지뿐이다.

```text
total_score_gte_threshold
```

## 9. 알려진 실패 사례 처리

### Track 88

- 촬영 차량 보닛 false positive다.
- ID 예외 또는 mask로 제거하지 않았다.
- 0.70 이상 spike가 있었지만 `minimum_event_duration_sec` 적용 후 최종 event에는 남지 않았다.
- 재추론하면 ID 번호가 바뀔 수 있으므로 `88`이라는 번호를 일반 규칙으로 사용하면 안 된다.

### Track 669

- frame 725의 단일 high-score spike다.
- `minimum_event_duration_sec`을 통과하지 못했다.

### Track 29

- 작은 bbox의 growth spike다.
- 여러 짧은 candidate가 있었지만 최종 event가 되지 않았다.
- 임의의 작은 bbox 감점 규칙을 추가하지 않았다.

### Track 697과 887

- 같은 시간대에 높은 IoU를 보인 fragmentation 또는 duplicate 후보다.
- 객체별 별도 event를 만들지 않고 frame timeline에서 하나의 `event_001`로 처리했다.
- Re-ID 또는 track ID 병합은 수행하지 않았다.
- 최고 score row의 track 697을 `primary_track_id`로 기록했다.

## 10. FFmpeg clip 결과

### 재인코딩 정책

- codec: `libx264`
- preset: `medium`
- CRF: 23
- pixel format: `yuv420p`
- `faststart` 적용
- audio 제거
- stream copy 미사용

### clip별 실제 결과

| clip | requested | actual | codec | 해상도 | FPS | frames | size |
|---|---:|---:|---|---:|---:|---:|---:|
| `event_001.mp4` | 3.700000초 | 3.700000초 | H.264 | 1280×720 | 30 | 111 | 1,661,513 bytes |
| `event_002.mp4` | 2.633333초 | 2.633333초 | H.264 | 1280×720 | 30 | 79 | 1,141,218 bytes |

### manifest

| 항목 | 결과 |
|---|---:|
| `event_count` | 2 |
| `clip_count` | 2 |
| total requested duration | 6.333333초 |
| total actual duration | 6.333333초 |

`event_002`의 `clip_end_sec`는 post-roll을 적용한 값이 원본 영상 범위를 넘지 않도록 영상 duration인 30.0초에서 clamp되었다.

## 11. 자동·실제·사람 검증

### 자동 검증

| 검증 | 결과 |
|---|---:|
| telemetry tests | 34 passed |
| event tests | 53 passed |
| clip tests | 56 passed |
| Day 6 related tests | 143 passed |
| full pytest | 436 passed |

### 실제 데이터 smoke

- synthetic telemetry 생성 성공
- sync CSV 생성 성공
- event JSON 생성 성공
- clip 2개 생성 성공
- ffprobe 검증 성공

### 사람 검증

- `event_001` 정상 재생
- `event_002` 정상 재생
- bbox, ID, trajectory 정상 표시
- 영상 시작과 끝 깨짐 없음
- `event_002`가 원본 영상 끝에서 정상 종료

기준 영상에는 risk score 숫자 overlay가 없으므로 trigger frame을 영상에 표시된 숫자로 직접 확인한 것은 아니다. 검증은 생성된 timeline 및 event JSON의 수치와 clip의 해당 영상 구간을 함께 확인하는 방식으로 수행했다.

마지막 검증일: **2026-08-02 KST**

## 12. 재현 명령

다음 명령은 repository root인 `E:\Projects\DashGuard`에서 PowerShell로 순서대로 실행한다. project Python은 `E:\Projects\DashGuard\.venv\Scripts\python.exe`를 사용한다.

### 1. synthetic telemetry 생성

```powershell
& "E:\Projects\DashGuard\.venv\Scripts\python.exe" ".\scripts\day6_generate_synthetic_telemetry.py" `
  --output-csv ".\outputs\day6\telemetry\day6_synthetic_telemetry_usercheck_20260801.csv" `
  --duration-sec 30.0 `
  --frequency-hz 10.0
```

### 2. risk CSV와 telemetry 동기화

```powershell
& "E:\Projects\DashGuard\.venv\Scripts\python.exe" ".\scripts\day6_sync_telemetry.py" `
  --risk-csv ".\outputs\day5\csv\day5_risk_scores.csv" `
  --telemetry-csv ".\outputs\day6\telemetry\day6_synthetic_telemetry_usercheck_20260801.csv" `
  --output-csv ".\outputs\day6\csv\day6_risk_telemetry_sync_roundtrip_check.csv" `
  --offset-sec 0.0 `
  --tolerance-sec 0.05
```

### 3. event timeline과 JSON 생성

```powershell
& "E:\Projects\DashGuard\.venv\Scripts\python.exe" ".\scripts\day6_detect_events.py" `
  --input-csv ".\outputs\day6\csv\day6_risk_telemetry_sync_roundtrip_check.csv" `
  --input-video ".\outputs\day3\videos\day3_baseline.mp4" `
  --output-timeline-csv ".\outputs\day6\timeline\day6_frame_risk_timeline.csv" `
  --output-events-json ".\outputs\day6\events\day6_event_candidates.json" `
  --score-threshold 0.70 `
  --minimum-event-duration-sec 0.15 `
  --merge-gap-sec 0.10 `
  --pre-roll-sec 2.0 `
  --post-roll-sec 1.0
```

### 4. event clip 생성

```powershell
& "E:\Projects\DashGuard\.venv\Scripts\python.exe" ".\scripts\day6_extract_event_clips.py" `
  --events-json ".\outputs\day6\events\day6_event_candidates.json" `
  --input-video ".\outputs\day3\videos\day3_baseline.mp4" `
  --output-dir ".\outputs\day6\clips" `
  --output-manifest-json ".\outputs\day6\clips\day6_clip_manifest.json" `
  --ffmpeg-bin "ffmpeg" `
  --ffprobe-bin "ffprobe"
```

각 CLI는 기본적으로 기존 결과를 덮어쓰지 않는다. 지정한 output이 이미 있으면 `FileExistsError`가 발생하는 것이 정상 동작이다.

- 기존 결과를 보존하며 재실행하려면 새로운 output 이름 또는 output directory를 사용한다.
- 기존 결과를 의도적으로 교체할 때만 해당 명령에 `--overwrite`를 추가한다.

## 13. 생성된 로컬 산출물

| 종류 | 로컬 경로 |
|---|---|
| synthetic telemetry CSV | `outputs/day6/telemetry/day6_synthetic_telemetry_usercheck_20260801.csv` |
| synchronized CSV | `outputs/day6/csv/day6_risk_telemetry_sync_roundtrip_check.csv` |
| frame timeline CSV | `outputs/day6/timeline/day6_frame_risk_timeline.csv` |
| event candidate JSON | `outputs/day6/events/day6_event_candidates.json` |
| event clip 1 | `outputs/day6/clips/event_001.mp4` |
| event clip 2 | `outputs/day6/clips/event_002.mp4` |
| clip manifest JSON | `outputs/day6/clips/day6_clip_manifest.json` |

`outputs` 아래 파일은 로컬 검증 산출물이며 Git commit 대상이 아니다.

## 14. 한계와 다음 단계

### 한계

- synthetic telemetry만 검증했다.
- 실제 센서의 offset, drift, jitter를 검증하지 않았다.
- threshold와 duration은 현재 sample을 위한 실험 설정이다.
- event ground truth가 없다.
- Precision/Recall을 산출하지 않았다.
- 현재 한 영상에 대해서만 통합 검증했다.
- detector 오탐·미탐과 track fragmentation이 존재한다.
- risk proxy는 실제 위험도나 사고 가능성을 나타내지 않는다.
- score overlay가 없다.
- clip에는 audio가 없다.
- 다중 파일 교체 rollback은 프로세스 오류에 대비하지만 전원 손실까지 보장하는 파일시스템 트랜잭션은 아니다.
- 실행 환경에 FFmpeg와 `libx264`가 필요하다.

### 다음 단계

다음 작업은 Day 7 UI 또는 전체 10일 계획의 다음 항목에서 진행한다. 이 문서는 Day 6의 telemetry sync, event candidate, clip 파이프라인 검증만 기록하며 Day 7 기능 구현을 포함하지 않는다.

요약하면, 이번 검증은 한 개의 30초 영상과 deterministic synthetic telemetry를 이용한 통합 검증이다. 센서 현실성, event ground truth, 일반화 성능은 아직 확인되지 않았으며, 현재 threshold와 risk proxy를 운영상의 위험 판정으로 해석해서는 안 된다.
