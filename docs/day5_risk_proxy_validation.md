# Day 5 영상 기반 접근 위험 점수 검증

## 1. 목적과 용어

이 문서는 현재 Day 5 구현과 Day 3 CSV의 실제 실행 결과를 기준으로 **영상 기반 접근 위험 점수**의 데이터 계약, 특징, 점수 수식, 검증 결과 및 알려진 한계를 기록한다.

- **risk proxy**: 영상 bbox의 크기·변화·화면 내 위치를 결합한 해석 가능한 상대 지표다.
- **고위험 후보 구간**: risk proxy가 상대적으로 높아 후속 육안 검토 대상으로 삼을 수 있는 구간을 뜻한다. 현재 점수 cutoff나 구간 정의는 확정되지 않았다.

이 점수는 물리적 위험량이나 확률이 아니다. 카메라 영상만으로 현실의 위험 정도를 확정하거나 특정 사건의 발생 가능성을 입증하는 값으로 해석해서는 안 된다.

## 2. 입력과 출력 계약

### 입력 CSV

원본 열은 다음 9개다.

| 순서 | 열 |
|---:|---|
| 1 | `frame_index` |
| 2 | `timestamp_sec` |
| 3 | `track_id` |
| 4 | `class_name` |
| 5 | `x1` |
| 6 | `y1` |
| 7 | `x2` |
| 8 | `y2` |
| 9 | `confidence` |

참조 영상과 실제 입력의 확인값은 다음과 같다.

| 항목 | 값 |
|---|---:|
| frame width | 1280 |
| frame height | 720 |
| FPS | 30 |
| frames | 900 |
| duration | 30 seconds |
| input rows | 5,360 |
| unique tracks | 108 |

### 출력 CSV

- 출력은 입력과 동일한 5,360 rows다.
- 원본 행 순서를 유지하며 입력 행을 삭제하지 않는다.
- 원본 9개 열과 그 값을 보존한다.
- 총 30개 열이다: 원본 9개 열, feature 12개 열, score/signal/contribution 9개 열.
- feature 열에는 아래 10개 필수 특징과 계산 감사용 `bbox_valid`, `bbox_clipped`가 포함된다.

## 3. bbox와 history 정책

### bbox 처리

- 원본 bbox(`x1`, `y1`, `x2`, `y2`)는 수정하지 않는다.
- 계산용 bbox만 frame 경계인 `[0, frame_width] × [0, frame_height]`로 clip한다.
- 다음 중 하나에 해당하면 bbox는 invalid다.
  - bbox 좌표 중 하나라도 non-finite
  - `x2 <= x1`
  - `y2 <= y1`
  - clip 후 면적이 0
- invalid 행도 삭제하지 않는다. 해당 행의 계산 불가능한 feature는 결측으로 남기고 score 신호와 contribution은 0으로 처리한다.
- `frame_width <= 0` 또는 `frame_height <= 0`이면 오류다.

### history 처리

- history는 `track_id`별로만 관리한다.
- 서로 다른 track ID를 자동 병합하지 않는다.
- 각 `track_id` 내부 관측 순서는 `frame_index`를 기준으로 정렬하고, `timestamp_sec`은 `delta_t` 계산에 사용한다.
- `delta_t <= 0` 또는 `frame_gap <= 0`이면 history를 reset하고 `growth_status=nonpositive_delta`로 기록한다.
- `delta_t > 0.20` seconds 또는 `frame_gap > 6`이면 history를 reset하고 `growth_status=gap_reset`으로 기록한다.
- `0.20` seconds와 6 frames의 gap 값은 현재 프로젝트 baseline이며 일반 표준값이 아니다.
- invalid bbox는 해당 track의 연속 history를 끊는다. 그 다음 유효 observation은 새로운 history로 시작하며 `growth_status=first_observation`이 된다.

## 4. 특징 정의

아래 범위는 유효 bbox에 대한 값이다. “결측”은 CSV에서 빈 값으로 저장되는 수치 feature를 뜻하며, boolean/status 열은 명시적인 값으로 남는다.

| 특징 | 정의 | 단위 | 범위 | 결측 정책 |
|---|---|---|---|---|
| `bbox_area` | clip된 bbox의 width × height | pixel² | `0 < bbox_area <= frame_width * frame_height` | invalid bbox면 결측 |
| `area_ratio` | frame 전체 면적에 대한 `bbox_area` 비율 | 무차원 | `0 < area_ratio <= 1` | invalid bbox면 결측 |
| `smoothed_area_ratio` | 현재 연속 구간의 최근 3 observations에 대한 `area_ratio` 단순 이동평균 | 무차원 | `0 < smoothed_area_ratio <= 1` | invalid bbox, reset 직후 및 첫 2 observations는 결측 |
| `raw_growth_rate_per_sec` | 연속한 두 raw `area_ratio` 사이의 상대 증가율 | 1/sec | 부호 제한 없는 유한 실수 | 첫 observation, reset 행, invalid bbox 또는 정의 불가능한 계산이면 결측 |
| `growth_rate_per_sec` | 연속한 두 `smoothed_area_ratio` 사이의 상대 증가율 | 1/sec | 부호 제한 없는 유한 실수 | smoothing history가 부족하거나 reset/invalid/정의 불가능이면 결측 |
| `history_length` | 현재 연속 구간에서 해당 행까지의 observation 수 | observations | invalid bbox는 0, 유효 bbox는 1 이상의 정수 | 결측 없음 |
| `growth_valid` | `growth_rate_per_sec`을 유효하게 계산했는지 여부 | boolean | `True` 또는 `False` | 결측 없음; 계산 불가능하면 `False` |
| `growth_status` | growth 계산 상태 | category | 아래 6개 값 | 결측 없음 |
| `center_proximity` | clip된 bbox 하단 중앙점이 화면 하단 중앙에 가까운 정도 | 무차원 | 0~1 | invalid bbox면 결측 |
| `in_roi` | clip된 bbox 하단 중앙점이 ROI 내부 또는 경계선 위인지 여부 | boolean | `True` 또는 `False` | 결측 없음; invalid bbox면 `False` |

### 면적과 growth 수식

```text
bbox_area = clipped_bbox_width * clipped_bbox_height

area_ratio = bbox_area / (frame_width * frame_height)
```

Smoothing은 frame 수가 아니라 **현재 연속 구간의 observation 수**를 기준으로 한다. 기본값은 최근 3 observations의 단순 이동평균이다.

```text
smoothed_area_ratio[t]
    = mean(area_ratio[t-2], area_ratio[t-1], area_ratio[t])
```

첫 2 observations의 `smoothed_area_ratio`는 결측이다. 이전과 현재의 smoothed 값이 모두 있어야 하므로, 기본 설정에서는 연속 4번째 observation부터 `growth_rate_per_sec`이 유효해질 수 있다.

```text
raw_growth_rate_per_sec
    = ((current_area_ratio - previous_area_ratio)
       / previous_area_ratio)
      / delta_t

growth_rate_per_sec
    = ((current_smoothed_area_ratio - previous_smoothed_area_ratio)
       / previous_smoothed_area_ratio)
      / delta_t
```

두 growth feature의 단위는 모두 `1/sec`다.

### 화면 중심 근접도와 ROI

`center_proximity`와 `in_roi`는 모두 clip된 bbox의 하단 중앙점을 사용한다. `center_proximity`는 화면 하단 중앙에 가까울수록 1이고 멀수록 0에 가까워진다.

```text
bbox_bottom_center_x = (clipped_x1 + clipped_x2) / 2
bbox_bottom_center_y = clipped_y2

normalized_x = (bbox_bottom_center_x - frame_width / 2) / (frame_width / 2)
normalized_y = (frame_height - bbox_bottom_center_y) / frame_height

center_proximity
    = 1 - clip(sqrt(normalized_x^2 + normalized_y^2) / sqrt(2), 0, 1)
```

`in_roi`는 하단 중앙점이 아래 정규화 사다리꼴 내부 또는 경계선 위에 있으면 `True`다.

```text
(0.35, 0.48)
(0.65, 0.48)
(0.95, 1.00)
(0.05, 1.00)
```

이 ROI는 현재 영상 기준의 고정 baseline이며 카메라 calibration이 아니다.

### `growth_status`

| 값 | 의미 |
|---|---|
| `invalid_bbox` | bbox가 invalid여서 feature/history 계산을 수행할 수 없음 |
| `first_observation` | 해당 `track_id`의 첫 유효 observation 또는 invalid bbox 뒤의 첫 유효 observation |
| `gap_reset` | 허용 gap을 초과해 새 연속 구간을 시작함 |
| `nonpositive_delta` | `delta_t <= 0` 또는 `frame_gap <= 0`이어서 새 연속 구간을 시작함 |
| `insufficient_history` | 연속 구간은 유효하지만 smoothing growth 계산에 필요한 observation이 아직 부족함 |
| `valid` | `growth_rate_per_sec`이 유효하게 계산됨 |

## 5. 점수 수식

### 신호 정규화

```text
area_signal = clip(area_ratio / 0.10, 0, 1)

growth_signal = 0                                      if growth_valid=False
growth_signal = clip(max(growth_rate_per_sec, 0) / 3.0,
                     0,
                     1)                                otherwise

center_signal = center_proximity

roi_signal = 1 if in_roi else 0
```

유효 bbox의 `center_proximity`는 이미 0~1이지만, score 단계에서도 0~1 범위로 clip한다. invalid bbox의 네 신호는 모두 0이다.

### 가중치와 contribution

| 신호 | 가중치 | contribution |
|---|---:|---|
| area | `area_weight = 0.15` | `area_contribution = area_weight * area_signal` |
| growth | `growth_weight = 0.45` | `growth_contribution = growth_weight * growth_signal` |
| center | `center_weight = 0.15` | `center_contribution = center_weight * center_signal` |
| ROI | `roi_weight = 0.25` | `roi_contribution = roi_weight * roi_signal` |
| 합계 | **1.00** |  |

```text
total_score
    = area_contribution
    + growth_contribution
    + center_contribution
    + roi_contribution
```

`total_score`은 네 contribution의 직접 합이며 사후 clamp를 적용하지 않는다. 허용된 결측에 대응하는 contribution은 0이고 남은 가중치로 재분배하지 않는다. 입력의 `confidence`는 출력에 보존하지만 현재 점수 계산에는 사용하지 않는다.

`0.10`, `3.0/sec` 및 네 가중치는 현재 Day 3 분포를 이용해 정한 해석 가능한 baseline이다. 검증된 산업 표준이나 운영상 안전 판단 threshold가 아니다.

## 6. 실제 실행 결과

### 전체 결과: 측정 사실

| 항목 | 결과 |
|---|---:|
| output rows | 5,360 |
| unique tracks | 108 |
| `growth_valid` rows | 5,004 |
| `growth_status=valid` | 5,004 |
| `growth_status=insufficient_history` | 223 |
| `growth_status=first_observation` | 108 |
| `growth_status=gap_reset` | 25 |
| `total_score` minimum | 0.0348025543178328 |
| `total_score` maximum | 0.9403815803991128 |
| contribution sum maximum error after CSV reload | 2.7755575615628914e-16 |
| score NaN | 0 |
| score infinity | 0 |

### 최고 점수 행: 측정 사실

| 필드 | 값 |
|---|---:|
| `track_id` | 697 |
| `class_name` | `car` |
| `frame_index` | 860 |
| `timestamp_sec` | 28.666667 |
| `total_score` | 0.9403815803991128 |
| `area_ratio` | 0.151420 |
| `growth_rate_per_sec` | 3.655448 |
| `center_proximity` | 0.602544 |
| `in_roi` | `True` |
| `area_contribution` | 0.15 |
| `growth_contribution` | 0.45 |
| `center_contribution` | 약 0.090382 |
| `roi_contribution` | 0.25 |

### 최고 점수 행: 해석

이 행은 큰 bbox 면적, 양의 growth, 화면 하단 중앙 근접도 및 ROI 조건이 동시에 활성화된 **고위험 후보**다. 다만 현실의 충돌 위험도나 사고 발생 가능성을 입증하는 결과는 아니다.

### 탐색용 score 분포

아래 값은 분포를 살펴보기 위한 행 수일 뿐이며, threshold 확정이나 성능 측정 결과가 아니다.

| 조건 | rows |
|---|---:|
| `score >= 0.5` | 996 |
| `score >= 0.6` | 153 |
| `score >= 0.7` | 64 |
| `score >= 0.8` | 33 |
| `score >= 0.9` | 10 |

## 7. 알려진 실패 사례

### Track 88 보닛 false positive

#### 측정 사실

| 항목 | 값 |
|---|---:|
| rows | 450 |
| class | `car` |
| `area_ratio` | 약 0.2294~0.3180 |
| `center_proximity` | 약 0.8961~0.9964 |
| ROI 내부 비율 | 1.0 |
| `growth_valid` rows | 412 |
| median `total_score` | 0.5492729315005367 |
| mean `total_score` | 0.5618817390885851 |
| max `total_score` | 0.8495048712818288 |

#### 해석과 처리 원칙

- 촬영 차량 보닛을 `car`로 오탐한 detector 결과다.
- ID 88이라는 번호는 이 고정 입력에서만 해당하며 일반 규칙으로 하드코딩하지 않는다.
- 큰 면적, 하단 중앙 근접도 및 ROI 내부 조건 때문에 growth가 낮아도 점수가 높아진다.
- detector false positive가 risk proxy로 전파되는 대표 한계다.
- 점수를 좋게 보이게 하기 위해 detector, `confidence`, ByteTrack, ROI 또는 mask를 수정하지 않았다.

### Track fragmentation 또는 중복 후보

- `track 697`과 `track 887`은 frame 856~857 부근에서 유사한 큰 bbox와 `car`/`truck` class 흔들림을 보이며 동시에 높은 점수를 기록했다.
- `track 887`은 7 rows이고 max score는 약 0.908472다.
- 자동 병합이나 Re-ID를 수행하지 않았으므로 각 track을 독립적으로 평가한다.
- 동일 물체의 ID fragmentation 또는 중복 추적이라면 후보 수가 늘 수 있다. 현재 수치만으로 동일 물체라고 단정하지 않으며 육안 검증이 필요한 후보로 취급한다.

### 짧은 track

- `track 669`는 5 rows이고 max score는 약 0.796080이다.
- 1~2 observation ghost track은 `growth_valid=True`가 될 수 없다.
- 그러나 4 observations 이상인 짧은 track은 smoothing growth가 유효해질 수 있다.
- 짧은 high-score 후보는 별도 육안 검토가 필요하다.

### 작은 객체 growth spike

- `track 29`의 max `area_ratio`는 약 0.001742다.
- max `growth_rate_per_sec`은 약 5.471651이다.
- max score는 약 0.797030이다.
- ROI ratio는 1.0이다.
- 작은 bbox에서는 작은 픽셀 변화도 큰 상대 증가율이 될 수 있다.
- smoothing과 saturation으로 영향을 완화했지만 제거하지는 못했다.

## 8. 구조적 한계

- monocular bbox만 사용하므로 미터 단위 거리와 속도 정보를 알 수 없다.
- 카메라 calibration, ego-motion, perspective geometry를 사용하지 않는다.
- detector가 놓친 객체는 점수화할 수 없다.
- false positive는 정상 검출 객체와 같은 방식으로 점수화될 수 있다.
- ID fragmentation을 연결하지 않는다.
- class 흔들림을 보정하지 않는다.
- ROI는 고정 사다리꼴이므로 곡선 도로와 카메라 움직임에 취약하다.
- smoothing 후에도 bbox jitter가 growth에 남아 있을 수 있다.
- manual ground truth와 Precision/Recall이 없다.
- score threshold와 고위험 후보 구간 정의는 아직 검증되지 않았다.
- 현재 결과는 Day 3 한 영상에 대한 baseline이며 일반화 성능에 대한 주장이 아니다.

## 9. 재현 명령과 산출물

저장소 루트의 PowerShell에서 다음 명령을 실행한다.

```powershell
& "E:\Projects\DashGuard\.venv\Scripts\python.exe" ".\scripts\day5_score_tracks.py" --input-csv ".\outputs\day3\csv\day3_tracks.csv" --input-video ".\outputs\day3\standardized\day3_baseline_input.mp4" --output-csv ".\outputs\day5\csv\day5_risk_scores.csv" --output-plot ".\outputs\day5\plots\day5_frame_max_risk_proxy.png"
```

산출물은 다음과 같다.

- `outputs/day5/csv/day5_risk_scores.csv`
- `outputs/day5/plots/day5_frame_max_risk_proxy.png`

그래프는 `timestamp_sec`별 `total_score` 최대값을 표시한다. y축 범위는 0~1이고 threshold 선은 없다. 최고 peak는 영상 마지막 약 1~2초 구간에 있다.

## 10. 검증

아래 테스트 숫자는 **현재 commit 전 결과**다.

| 검증 항목 | 결과 |
|---|---:|
| risk feature tests | 62 passed |
| risk score tests | 91 passed |
| risk report tests | 18 passed |
| Day 5 targeted | 171 passed |
| 전체 pytest | 293 passed |

그래프 육안 확인을 완료했으며, 표시된 제목은 다음과 같다.

```text
Video-based approach risk proxy — frame maximum
```

검증 범위는 데이터 계약, 수식 구현, 행·열 보존, contribution 합, 수치 유한성 및 현재 산출물의 표시 상태다. manual ground truth가 없으므로 위험 탐지 성능이나 threshold 적합성을 검증한 결과로 확대 해석하지 않는다.
