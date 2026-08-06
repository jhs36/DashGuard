# Day 9 event labeling guide

## 1. 평가 질문과 범위

이 평가는 사람이 영상에서 확인한 `visual_rapid_approach` 구간과 DashGuard가
생성한 event 구간이 시간상 얼마나 겹치는지를 묻는다. 범위는
`single-source small-sample pilot evaluation`이다. 같은 원본에서 서로 겹치지
않는 여러 segment를 만들더라도 독립 원본 수는 1개로 보고한다.

이 작업은 정식 사고·충돌·안전 성능 평가가 아니며, 실제 거리 또는 TTC(time to
collision)를 측정하지 않는다.

## 2. positive event 정의

현재 허용하는 event type은 `visual_rapid_approach` 하나뿐이다.

> 전방 객체가 자차 진행 영역에서 지속적으로 가까워지거나 화면상 크기가
> 명확히 증가하여 사람이 검토할 필요가 있다고 판단할 수 있는 영상 구간

사람이 주석 없는 실제 평가 segment를 직접 보고 판단하며, 관찰 가능한 접근
evidence가 최소 0.30초 지속되어야 한다.

포함할 수 있는 예는 다음과 같다.

- 앞차와의 간격이 계속 줄어드는 모습이 분명한 구간
- 자차 진행 영역의 객체가 연속적으로 커져 접근으로 판단할 수 있는 구간
- 잠깐의 검출 흔들림이 아니라 여러 프레임에서 일관되게 접근하는 구간

다음은 positive로 쓰지 않는다.

- DashGuard score가 threshold 이상이라는 사실 자체
- prediction event를 사람 label에 복사한 구간
- 촬영 차량의 보닛
- 크게 보이지만 정지해 있거나 멀어지는 객체
- 횡방향으로 지나가기만 하는 객체
- 한두 프레임의 bbox jitter
- synthetic hard-brake evidence만 존재하는 구간
- 실제 영상 근거 없이 사고 또는 충돌로 해석한 구간

## 3. 시작과 끝 경계 선택

시작은 명확한 전방 접근 또는 화면상 크기 증가가 시작되고, 그 상태가 최소
0.30초 지속되는 첫 시점이다. 0.30초가 지난 시점을 시작으로 잡는 것이 아니라,
나중에 지속 조건이 확인되면 그 증거가 처음 시작된 시점으로 돌아가 표시한다.

끝은 다음 중 가장 이른 시점이다.

- 접근이 멈추거나 약해진 시점
- 객체가 화면 또는 전방 진행 영역에서 이탈한 시점
- 객체가 가려져 더 이상 판단할 수 없는 시점

모든 구간은 반개구간 `[start_sec, end_sec)`이다. 시작 시점은 포함하고 끝
시점은 포함하지 않는다. 따라서 `[10.0, 12.0)`과 `[12.0, 13.0)`은 닿지만
겹치지 않으며 temporal IoU는 0이다. 시간을 임의로 clamp, swap, drop하지
않는다. `0 <= start_sec < end_sec <= video_duration_sec`를 만족하지 않으면
라벨을 고쳐 다시 검증한다.

## 4. prediction을 보지 않고 먼저 label하는 이유

사람 label은 `events.json`, score graph, hard-brake 값, annotated video 또는
기존 event clip을 보지 않고 주석 없는 평가 segment만 보고 먼저 작성한다.
Prediction을 먼저 보면 모델이 선택한 경계를 정답처럼 따라가는 편향이 생기며,
평가가 prediction을 다시 복사해 확인하는 순환 검증이 된다. CSV에도 prediction
값을 미리 채우지 않는다. Annotated video는 prediction 생성이 끝난 뒤 FP 및
track ID 검토에만 사용한다.

## 5. CSV 작성법

열 순서와 이름은 다음과 같이 고정한다.

```text
schema_version
video_id
source_video_sha256
label_id
event_type
start_sec
end_sec
reviewer_id
review_confidence
notes
```

- `schema_version`: 현재 `1.0`
- `video_id`: segment를 안정적으로 식별하는 비어 있지 않은 문자열
- `source_video_sha256`: 실제 평가 segment 파일의 64자리 lowercase hexadecimal SHA-256
- `label_id`: 같은 video 안에서 고유한 사람 label ID
- `event_type`: `visual_rapid_approach`
- `start_sec`, `end_sec`: 영상 시작을 0으로 한 초 단위 `[start, end)` 경계
- `reviewer_id`: 비어 있지 않은 reviewer 식별자
- `review_confidence`: `high`, `medium`, `low` 중 하나
- `notes`: 판단 근거 또는 모호한 점이며 빈 문자열도 허용

같은 `video_id`의 모든 행은 같은 source SHA-256을 사용한다. 열을 추가하거나
이름/순서를 바꾸면 validation error가 발생한다. 행 순서는 결과에 영향을 주지
않는다. Header만 있는 label CSV는 label 0개를 뜻하지만, 어떤 negative 영상을
검토했는지는 별도 reviewed-video scope CSV와 함께 제공해야 정상 평가 범위가
된다. Template에는 실제 정답처럼 보이는 예제 행을 넣지 않는다.

Scope CSV 열 순서는 다음과 같다.

```text
schema_version,video_id,source_video_sha256,video_duration_sec,independent_source_id,reviewer_id,review_status
```

Scope는 검토 완료한 segment마다 한 행을 가지며 `review_status=completed`만
허용한다. `video_id`는 고유해야 하고, SHA-256은 해당 segment 파일, duration은
그 segment의 실제 길이다. Header-only scope는 허용하지 않는다.

`review_confidence`는 경계와 event 판단이 모두 분명하면 `high`를 사용한다.
Event 포함 여부는 명확하지만 정확한 시작·끝 경계만 모호하면 `medium` 또는
`low`로 낮춘다. Confidence는 matching 가중치가 아니며 모든 유효 label은
동일하게 계산한다.

## 6. 모호한 사례 처리

먼저 영상을 다시 보고 느린 재생과 프레임 이동으로 0.30초 지속 여부와 진행 영역
포함 여부를 확인한다. Event 포함은 명확하고 경계만 모호하면 보수적으로 경계를
정하고 `notes`에 선택 이유를 적은 뒤 confidence를 낮춘다. Event 포함 여부 자체를
재현 가능하게 결정할 수 없으면 primary label에서 제외하고 별도 decision record에
관찰과 제외 이유를 남긴다. 영상 evidence가 부족하면 사고·충돌 의미를 추정해
positive로 만들지 않는다. Label 사이에 겹침이 생겼다면 같은 접근 사건을 중복
표시했는지 먼저 확인한다.

## 7. temporal matching과 수기 예제

같은 `video_id`와 같은 `event_type`만 비교한다. Temporal IoU는 겹친 시간 길이를
두 구간의 합집합 시간 길이로 나눈 값이다. threshold 비교는 `iou >= threshold`로
하며, 한 label과 한 prediction은 각각 최대 한 번만 TP가 된다.

Labels:

```text
L1: [10.0, 12.0)
L2: [20.0, 22.0)
```

Predictions:

```text
P1: [10.5, 11.5)
P2: [21.0, 23.0)
P3: [5.0, 6.0)
```

`L1-P1 = 0.5`, `L2-P2 = 1/3`이고 나머지 pair는 0이다. Primary IoU 0.30에서는
`TP=2`, `FP=1`, `FN=0`, `precision=2/3`, `recall=1.0`이다. Sensitivity IoU
0.50에서는 `TP=1`, `FP=2`, `FN=1`, `precision=1/3`, `recall=0.5`이다.

후보 pair는 다음 순서로 고정 정렬한 뒤 이미 사용된 label/prediction을 건너뛴다.

1. temporal IoU 내림차순
2. intersection duration 내림차순
3. `abs(pred.start_sec - label.start_sec)` 오름차순
4. `abs(pred.end_sec - label.end_sec)` 오름차순
5. `label_id` 사전순
6. `prediction_id` 사전순

TP는 선택된 matched pair, FP는 어떤 label과도 선택되지 않은 prediction, FN은
어떤 prediction과도 선택되지 않은 label이다. Precision은 `TP/(TP+FP)`, recall은
`TP/(TP+FN)`이다. 여러 영상의 micro 결과는 TP·FP·FN을 먼저 합산해 계산한다.
Macro metric은 primary 결과로 만들지 않는다.

IoU 0.30은 짧고 경계가 다소 주관적인 pilot label에서 의미 있는 시간 겹침을
보는 primary 기준이다. IoU 0.50은 더 엄격한 경계 일치를 확인하는 sensitivity
기준이다. 두 결과는 섞지 않고 별도로 보고하며, 둘 중 어느 것도 현재 score
threshold 0.70을 최적화하거나 검증하는 기준이 아니다.

## 8. 빈 영상 처리

- labels 0, predictions 0: `TP=FP=FN=0`, precision/recall `null`,
  `empty_agreement=true`
- labels 0, predictions 있음: 모든 prediction이 FP, precision `0.0`, recall
  `null`, `empty_agreement=false`
- labels 있음, predictions 0: 모든 label이 FN, precision `null`, recall `0.0`,
  `empty_agreement=false`

분모가 0인 metric은 0이나 1로 바꾸지 않고 JSON `null`로 기록한다.

## 9. 실행 입력

한 영상은 다음처럼 명시적으로 연결한다.

```powershell
.venv\Scripts\python.exe scripts\day9_evaluate_events.py `
  --labels-csv data\evaluation\event_labels.csv `
  --label-scope-csv data\evaluation\event_label_scope.csv `
  --prediction segment_01 outputs\day8\events\events.json `
  --output-json outputs\day9\event_evaluation.json
```

여러 영상은 `--prediction VIDEO_ID EVENTS_JSON`을 반복하거나 다음 manifest를
`--manifest`로 전달한다. 상대 `events_json` 경로는 manifest 위치를 기준으로
해석한다.

```json
{
  "predictions": [
    {"video_id": "segment_01", "events_json": "segment_01/events.json"},
    {"video_id": "segment_02", "events_json": "segment_02/events.json"}
  ]
}
```

기존 output은 기본적으로 거부하며 명시적인 `--overwrite`에서만 같은 디렉터리의
임시 파일을 완성한 뒤 atomic replace한다. 입력 CSV, manifest, `events.json`은
수정하지 않는다. Prediction interval은 `start_sec`, `end_sec`만 사용한다.
`clip_start_sec`, `clip_end_sec`, `hard_brake`, `max_total_score`는 matching에
사용하지 않는다.

## 10. provenance와 해석 한계

현재 프로젝트 상태에 기록된 Day 3 원본 정보는 다음과 같다.

```text
title: Calm City Drive at Daytime
creator: Jabriel
platform: Pexels
license: Pexels License
provenance_status: documented_in_project_state_but_source_url_not_archived
```

원본 URL과 license snapshot은 현재 저장소에 보관되어 있지 않다. 영상이나 영상
frame은 Git에 추가하지 않는다.

이 결과로 다음을 주장할 수 없다.

- 실제 거리 또는 TTC 성능
- 사고·충돌 확률
- 제품이나 시스템의 안전성 검증
- 여러 독립 영상으로 일반화되는 성능
- 현재 event threshold 0.70의 최적성 또는 검증
- synthetic telemetry 성능
- 객체 identity 성능, IDF1 또는 HOTA

이는 한 독립 원본의 pilot 평가일 뿐이다. 동일 원본의 비중복 segment는 시간상
다른 sample이지만 독립 촬영 자료가 아니므로 독립 표본 수를 늘리지 않는다.
