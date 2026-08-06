# Day 9 Single-Source Pilot Evaluation

## 1. 목적

이 문서는 DashGuard Day 9의 실제 pipeline 실행, temporal event 평가, human-reviewed false-positive audit, 5-candidate manual ID stability audit를 하나의 검증 기록으로 묶는다. **Ground truth**는 사람이 정답으로 동결한 event 구간이고, **prediction**은 pipeline이 생성한 검토 후보 구간이다.

DashGuard의 **risk proxy**는 bbox 면적·증가율·화면 중심·ROI 같은 영상 특징을 결합한 설명 가능한 휴리스틱 점수다. 실제 거리, TTC 또는 물리적 안전 확률이 아니다.

## 2. 평가 범위

| 항목 | 실제 범위 |
|---|---:|
| evaluation type | single-source small-sample pilot evaluation |
| independent source count | 1 |
| evaluated segment count | 3 |
| segment duration | 각 27.5초 |
| total evaluated duration | 82.5초 |

세 segment는 동일한 원본에서 추출한 비중복 구간이다. 따라서 독립 원본 수는 1이며, 세 개의 독립 영상으로 해석하지 않는다.

포함된 장면은 맑은 낮의 일반 도심 차량 장면, 신호대기, 좌·우회전, 측면 차량 통과다. 야간, 악천후, 강한 카메라 흔들림, 명확한 보행자 중심 장면, 여러 독립 도로 원본, 실제 충돌 또는 급박한 near-miss positive는 포함되지 않았다. 원래 마스터플랜의 조건별 영상 5개 요구도 충족하지 못했다.

## 3. 평가 데이터

원본은 `Calm City Drive at Daytime`, creator `Jabriel`, platform `Pexels`, 기록된 license `Pexels License`다. 출처 URL과 license snapshot은 저장소에 보존되지 않았으며 원본과 파생 segment는 Git 대상이 아니다. 자세한 provenance 경계는 [data_model_provenance.md](data_model_provenance.md)에 기록한다.

세 segment는 각각 1280×720, CFR 30 fps, 825 frame, 27.5초다. 원본 구간은 `[0.0, 27.5)`, `[28.0, 55.5)`, `[56.0, 83.5)`이고 segment 사이에는 각각 0.5초 간격이 있다.

## 4. Ground-truth 작성과 동결

- event type: `visual_rapid_approach`
- interval semantics: `[start_sec, end_sec)`
- minimum duration: 0.30초
- final positive label count: 0
- label version: v1, 검토 후 변경 없음
- reviewed-video scope: 3개 segment, 독립 원본 1개

사람은 prediction 파일을 숨긴 상태에서 주석 없는 세 segment를 검토했다. 다만 reviewer limitation은 다음과 같다.

> Predictions were hidden during this labeling session, but the reviewer had prior exposure to an earlier DashGuard demo run.

segment 3의 약 `[6.0, 8.0)` 우회전 사례는 positive에서 제외했다. 정상 우회전의 ego-motion 및 측면 정차 차량 통과와 실제 전방 급접근을 영상만으로 재현 가능하게 분리하기 어려웠고, 차량이 비정상 회피나 급제동 없이 정상 통과했기 때문이다.

Positive가 0개인 header-only label CSV만으로는 검토한 negative 영상 집합을 표현할 수 없으므로 `labels/event_label_scope_final.csv`를 별도로 동결했다. 이 scope는 세 segment의 `video_id`, segment 파일 SHA-256, 27.5초 duration, `day3_road_source`, reviewer와 completed 상태를 보존한다. `labels/label_scope_freeze.json`에는 scope SHA-256, segment manifest SHA-256과 final label SHA-256을 기록했다.

## 5. Temporal event 평가 계약

**Temporal IoU**는 두 시간 구간의 교집합 길이를 합집합 길이로 나눈 값이다. 모든 구간은 끝점을 포함하지 않는 반개구간 `[start_sec, end_sec)`로 계산한다.

- primary threshold: 0.30
- sensitivity threshold: 0.50
- threshold equality: 포함, 즉 `IoU >= threshold`
- matching: deterministic greedy one-to-one
- 정렬 우선순위: 높은 IoU, 긴 교집합, 작은 시작점 차이, 작은 끝점 차이, label ID, prediction ID

**TP**는 label과 일대일로 matching된 prediction, **FP**는 matching label이 없는 prediction, **FN**은 matching prediction이 없는 label이다. **Precision**은 prediction 가운데 TP의 비율이고, **Recall**은 label 가운데 TP의 비율이다. 분모가 비면 JSON은 `null`을 기록한다. label과 prediction이 모두 0개일 때만 `empty_agreement=true`다.

다음은 계약 확인용 수기 예제이며 실제 pilot 결과가 아니다.

| 예제 threshold | TP | FP | FN |
|---:|---:|---:|---:|
| IoU 0.30 | 2 | 1 | 0 |
| IoU 0.50 | 1 | 2 | 1 |

## 6. 실제 pipeline 실행

세 성공 run은 `yolo11n.pt`, PyTorch CPU, confidence 0.25, image size 640, ByteTrack, deterministic synthetic telemetry, event score threshold 0.70을 사용했다. **Synthetic telemetry**는 실제 GPS·IMU·OBD 대신 schema와 시간 동기화 통합을 확인하기 위해 결정적으로 생성한 예시 데이터다.

| video_id | processing_sec | tracking_records | unique_track_ids | risk_max | event_count | clip_count |
|---|---:|---:|---:|---:|---:|---:|
| `day3_source_seg01_000000_027500` | 78.046 | 3,194 | 70 | 0.999118 | 4 | 4 |
| `day3_source_seg02_028000_055500` | 66.287 | 5,079 | 93 | 0.911677 | 1 | 1 |
| `day3_source_seg03_056000_083500` | 68.152 | 2,282 | 80 | 0.949919 | 2 | 2 |

- successful runs: 3
- preserved failed attempt: 1
- successful run processing total: 212.485초
- prediction events: 7
- clips: 7

처리시간은 현재 PC의 PyTorch CPU 환경에서 얻은 run별 단일 표본이다. 처리량 보장이나 다른 장치의 성능으로 일반화하지 않는다.

## 7. Event 평가 결과

최종 평가 산출물은 reviewed-video scope를 검증한 `evaluation/event_evaluation_report_scoped.json`이다. 기존 `evaluation/event_evaluation_report.json`은 scope 계약 추가 전에 생성된 unscoped report로 보존했다. 두 report의 primary·sensitivity metric과 unmatched prediction ID는 모두 동일하며, scoped report만 evaluation scope와 각 segment SHA-256을 완전하게 기록한다.

| 기준 | labels | predictions | TP | FP | FN | Precision | Recall | empty_agreement |
|---|---:|---:|---:|---:|---:|---:|---|---|
| primary 0.30 | 0 | 7 | 0 | 7 | 0 | 0.0 | `null` | `false` |
| sensitivity 0.50 | 0 | 7 | 0 | 7 | 0 | 0.0 | `null` | `false` |

label이 없으므로 이번 결과에는 IoU threshold가 영향을 주지 않는다. Precision 0.0은 prediction 7개가 존재하지만 matching label이 0개이기 때문이다. Recall은 0.0이 아니라 positive label 분모가 없어 정의할 수 없는 `null`이다. Event 단위 true negative가 정의되지 않아 accuracy, specificity, true-negative rate는 계산하지 않았고 positive 탐지 능력도 평가하지 못했다.

## 8. 사람 검토 결과

평가기의 FP는 temporal matching 결과이고, 사람 확인 FP는 각 영상을 다시 보고 positive 정의에 해당하지 않는다고 판정한 결과다. 7개 prediction을 모두 사람 검토한 뒤에도 7개 모두 positive 정의에 해당하지 않았다.

| 항목 | 결과 |
|---|---:|
| reviewed predictions | 7 |
| confirmed false positives | 7 |
| possible label omissions | 0 |
| ambiguous predictions | 0 |
| confidence high / medium / low | 5 / 2 / 0 |

Root-cause 분포는 `risk_feature_limitation` 3, `detection_artifact` 2, `normal_motion_ego_motion` 2, `tracking_artifact` 0, `ambiguous` 0이다. 이 범주는 영상 관찰에 근거한 suspected cause이며 개별 요소의 인과 기여도를 실험적으로 분리한 결과는 아니다.

## 9. False-positive 유형

| 현상 | 실제 관찰 | 추정 원인 | 현재 처리 | 향후 개선 |
|---|---|---|---|---|
| 정상 좌회전 중 bbox 외형·면적 변화 | 앞차가 정상 좌회전 | 방향·외형 변화가 빠른 bbox growth로 표현됨 | 사람 확인 FP로 보존 | bbox growth의 robust smoothing 또는 track age 조건 검토 |
| 화면 경계에서 부분 차량이 전체 차량로 나타남 | 옆 차선 차량이 앞부분부터 전체 모습으로 진입 | 면적과 중심점이 빠르게 변화 | 사람 확인 FP로 보존 | 경계 출현 객체 warm-up 또는 최소 observation 수 검토 |
| 원거리 소형 차량 bbox 불안정 | 먼 차량 bbox가 여러 형태로 검출됐다가 사라짐 | 작은 객체의 검출 변동 | 사람 확인 FP로 보존 | detection confidence·track age 조건 검토 |
| 차량 간 가림에서 불안정 검출 | 차량들이 겹치는 순간 중간 영역 검출 생성 | 가림 구간의 불안정 검출 | 사람 확인 FP로 보존 | 최소 observation 수와 가림-aware 조건 검토 |
| 정상 측면 차량 통과 | 정차한 자차 옆 차선 차량이 정상 통과 | lateral motion을 접근으로 해석 | 사람 확인 FP로 보존 | lateral motion과 forward approach 분리 검토 |
| 자차 우회전과 카메라 ego-motion | 우회전하며 정차 차량 옆을 통과 | 카메라 회전이 전방 접근처럼 표현됨 | 사람 확인 FP로 보존 | optical flow 또는 ego-motion 보정 검토 |
| 하나의 정상 통과가 인접 event 2개로 분리 | 같은 우회전·측면 통과가 두 event로 나뉨 | 인접 score 구간의 분리 | 두 FP 모두 보존 | event refractory period 또는 병합 계약 재검토 |

이 방향들은 구현 완료 항목이 아니다. 이번 결과를 보고 threshold를 사후 조정하지 않았고, threshold 변경 하나로 문제가 해결된다고 단정하지 않는다.

## 10. Tracking 통합 버그

Ultralytics는 detection이 있지만 확정 track ID가 없는 정상 frame에서 `boxes.id=None`을 반환할 수 있다. Workflow wrapper가 이를 처리하지 못해 segment 3의 첫 실행이 tracking 단계에서 실패했다.

수정은 wrapper가 이 경우를 확정 track 0개인 frame으로 처리하는 범위다. Strict converter 계약은 유지되며, bbox·class·confidence와 배열 길이 등 기존 validation 또는 다른 오류를 숨기지 않는다. 회귀 테스트를 추가했고 기록된 tracking test는 `22 passed`, 전체 pytest 기준선은 `627 passed`다. 이는 tracker 성능 개선이 아니라 pipeline 통합 안정성 버그 수정이다.

### Day 9 final validation

Tracking bug 수정 직후의 위 기준선과 별도로, label scope·malformed CSV·report-name 보완까지 포함한 최종 검증은 다음과 같다.

| 검증 | 결과 |
|---|---:|
| temporal evaluation | 54 passed |
| tracking | 22 passed |
| full pytest | 641 passed |
| compileall | passed |

## 11. 수동 ID 안정성 평가

**ID switch**는 사람이 따라간 한 실제 객체의 primary track ID가 다른 번호로 교체되는 현상이다. **Duplicate track**은 같은 실제 객체 위에 둘 이상의 ID가 동시에 표시되는 후보 현상이고, **fragmentation**은 한 객체의 track이 여러 조각 또는 ID로 분리될 가능성을 뜻한다.

사람은 DaVinci Resolve에서 30 fps 프레임 단위로 5개 실제 차량 후보를 검토했다. Reviewer notes의 `SS.FF`는 `초 + frame/30`으로 변환했으며 원문 시간과 tracker visible range를 대체하지 않았다.

| video_id | track | role | confirmed ID switch | duplicate episodes | confidence | automated class changes |
|---|---:|---|---:|---:|---|---:|
| `day3_source_seg01_000000_027500` | 1 | stable_candidate | 0 | 0 | high | 0 |
| `day3_source_seg02_028000_055500` | 105 | stable_candidate | 0 | 1 | high | 0 |
| `day3_source_seg03_056000_083500` | 4 | stable_candidate | 0 | 0 | high | 0 |
| `day3_source_seg02_028000_055500` | 31 | stress_candidate | 0 | 12 | medium | 12 |
| `day3_source_seg03_056000_083500` | 103 | stress_candidate | 0 | 8 | medium | 18 |

| 수동 평가 항목 | 결과 |
|---|---:|
| candidate count | 5 |
| stable / stress candidates | 3 / 2 |
| confirmed ID switches | 0 |
| duplicate/fragmentation episodes | 21 |
| candidates with duplicate/fragmentation | 3 |
| missing intervals | 0 |
| human-observed occlusion | 0 |
| manual class change count | `null` |

Primary ID가 다른 ID로 완전히 교체되지 않아 검토한 다섯 후보의 switch는 0건이다. 동시에 추가 ID가 보인 현상은 switch와 별개인 duplicate/fragmentation 후보로 집계했다. Stable candidate인 track 105에서도 episode 1개가 관찰됐다. Class 변화는 사람이 완전히 계수하지 않았고 자동 class-change 수치를 사람 판정과 혼합하지 않는다. 이 평가는 공식 IDF1 또는 HOTA가 아니다.

## 12. 재현 명령

명령은 저장소 root에서 실행한다. 기존 동결 산출물을 덮어쓰지 않도록 재평가 출력은 별도 경로를 사용한다. 아래의 특정 run ID는 local Day 9 prediction artifact가 그대로 보존된 workspace에서만 작동한다. Fresh clone에는 원본·segment 영상, prediction run 또는 `events.json`이 포함되지 않는다.

```powershell
python scripts/day8_gradio_demo.py
```

```powershell
python scripts/day9_evaluate_events.py `
  --labels-csv outputs/day9/pilot_source01_v1/labels/event_labels_final.csv `
  --label-scope-csv outputs/day9/pilot_source01_v1/labels/event_label_scope_final.csv `
  --prediction day3_source_seg01_000000_027500 outputs/day9/pilot_source01_v1/prediction_runs/day9_day3_source_seg01_000000_027500_20260805T1104403954884Z/events/events.json `
  --prediction day3_source_seg02_028000_055500 outputs/day9/pilot_source01_v1/prediction_runs/day9_day3_source_seg02_028000_055500_20260805T1106195577795Z/events/events.json `
  --prediction day3_source_seg03_056000_083500 outputs/day9/pilot_source01_v1/prediction_runs/day9_day3_source_seg03_056000_083500_20260805T1111066651749Z/events/events.json `
  --output-json outputs/day9/reproduction/event_evaluation_report_scoped.json
```

```powershell
python -m pytest tests/test_tracking.py -q
python -m pytest tests/test_temporal_evaluation.py -q
python -m pytest -q
python -m compileall -q src scripts tests
```

## 13. 확인된 결과

- 독립 원본 1개에서 만든 비중복 27.5초 segment 3개, 총 82.5초를 평가했다.
- Reviewed-video scope의 세 segment SHA와 prediction video set·duration이 일치한다.
- 동결 label v1의 positive는 0개, pipeline prediction과 clip은 각각 7개다.
- Primary와 sensitivity 모두 TP 0, FP 7, FN 0, Precision 0.0, Recall `null`이다.
- 사람 검토 후 7개 prediction 모두 confirmed false positive이며 label omission 후보는 0개다.
- 검토한 5개 ID 후보에서는 확정 ID switch가 관찰되지 않았다.
- 3개 후보에서 총 21개의 duplicate/fragmentation overlap episode가 관찰됐다.

## 14. 추론

- 현재 negative-only 표본에서는 risk proxy와 detection 변동이 정상 회전·측면 통과·ego-motion에 반응하는 반복 패턴이 있다.
- Stable candidate라는 사전 선정 역할만으로 동시 duplicate 현상이 없다고 가정할 수 없다.
- Positive 표본과 더 다양한 독립 원본을 추가하기 전에는 threshold 선택이나 일반화 성능을 판단할 근거가 부족하다.

위 내용은 관찰된 패턴에서 도출한 제한적 해석이며 원인 요소별 ablation이나 독립 재현으로 확정한 인과 결론이 아니다.

## 15. 미검증·제한

- 독립 원본 1개와 segment 3개뿐이며 positive label이 없다.
- Reviewer 1명의 label 및 prediction review다.
- 실제 GPS·IMU·OBD telemetry를 사용하지 않았다.
- 실제 거리, TTC, 물리적 위험도, 운전자 행동 또는 안전 성능을 평가하지 않았다.
- Detector·tracker·risk feature별 오차 기여를 분리하지 않았다.
- ID 평가는 차량 후보 5개의 수동 표본이며 공식 MOT benchmark가 아니다.
- 일부 duplicate bbox의 ID 문자를 읽지 못했고 class 변화도 수동 완전 계수하지 않았다.
- PyTorch CPU 처리시간은 세 run의 단일 로컬 표본이다.

## 16. 사용 가능한 표현

- `single-source small-sample pilot`
- `human-reviewed false-positive audit`
- `5-candidate manual ID stability audit`
- `confirmed ID switches were not observed in the reviewed five candidates`
- `duplicate/fragmentation candidates were observed`
- `risk proxy`
- `experimental event candidates`

## 17. 사용하면 안 되는 표현

다음 문구는 이 평가가 입증하지 않은 성능 주장이다.

```text
collision detection
accident prediction
crash probability
safety validated
production ready
real-time performance
high accuracy
ID stability 100%
tracker accuracy 100%
IDF1 1.0
HOTA 1.0
generalization verified
Recall 0%
three independent videos
ID switch가 전혀 발생하지 않았다
```

대신 “검토한 5개 후보에서는 확정 ID switch가 관찰되지 않았다”처럼 표본과 판정 범위를 함께 쓴다.
