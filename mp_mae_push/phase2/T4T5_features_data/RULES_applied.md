# 외부 MP 데이터 처리 규칙 — 적용 기록 (전체 파싱 전에 작성, 2026-11-21)

규칙 원문은 `mp_mae_push/phase2/prereg/PREREG.md` 4절이다. 이 문서는 규칙을 바꾸지 않고, 파일의 실제 필드에 규칙을 어떻게 대응시켰는지만 적는다. 처음 2,000개 기록을 조사한 결과(아래 1절)를 본 뒤, 전체 파싱 전에 작성했다.

## 1. 필드 조사 (melting_point.sdf 처음 2,000개 기록)

| 필드 | 2,000개 중 존재 | 내용 |
|---|---|---|
| Value | 2,000 | 숫자 321, `a to b` 범위 1,639, `>N` 20, `~N to N` 15, `~N` 5 (원문의 about·above·greater than 같은 낱말도 `~`, `>`로 이미 기호화되어 있음) |
| OriginalText | 2,000 | 특허 원문 구절. 단위 표기: `°C`(공백 허용) 1,450, 단위 없이 `°`만 550, °F·K 0 |
| QuantityType | 2,000 | 전부 `MeltingPoint` |
| SuspiciousValue | 2,000 | 전부 `false` |
| FromLiterature | 2,000 | `false` 1,999, `true` 1 |
| SMILES, StdInChIKey, Patent, Paragraph | 2,000 | 구조·출처 |

분해 표지(dec, decomp, zers, pyrolysis)는 처음 2,000개에 없었다(분해점은 별도 파일 `pyrolysis_point.sdf`로 나뉘어 있고, 그 파일은 쓰지 않는다).

## 2. 규칙과 필드의 대응 (기록 단위, 이 순서로 적용하고 규칙별 제외 수를 센다)

| 순서 | PREREG 규칙 | 적용 방법 |
|---|---|---|
| r0 | (필드 가정) | `QuantityType` ≠ `MeltingPoint`이거나 `Value`·`SMILES` 필드가 없으면 제외(처리할 수 없는 필드로 기록) |
| r1 | 분해 표지 제외 | `OriginalText` 또는 `Value`에 정규식 `\bdec(\.|\b|omp)`, `zers`, `pyroly`(대소문자 무시)가 있으면 제외 |
| r2 | 부등호 값 제외 (`>`, `<`, `≥`, `≤`, `~`, `ca.`) | `Value` 또는 `OriginalText`에 이 기호 중 하나가 있으면 제외 |
| r3 | (값 형식) | `Value`가 숫자 또는 `a to b`가 아니면 형식 불명으로 제외 |
| r4 | 범위 값: 폭 ≤ 5 K이면 중앙값, > 5 K이면 제외 | `a to b`에서 b − a(°C 폭 = K 폭)로 판정. b < a도 폭 = /b − a/ |
| r5 | 단위: °C → +273.15, 불명확하면 제외 | `OriginalText`에 섭씨 표기(`°C`, `° C`, `ºC`, `deg C`, `degrees C`, centigrade, Celsius)가 있으면 °C로 보고 +273.15. **`°`만 있고 C가 없는 기록, °F, K, 표기 없음은 '단위 불명확'으로 제외한다.** 처음 2,000개 중 약 27 %(550개)가 `°`만 있는 기록이라 이 규칙으로 빠진다. 미국 특허 관례상 대부분 °C일 가능성이 높지만, PREREG 규칙을 바꾸지 않기 위해 엄격하게 적용하고 그 수를 따로 보고한다 |
| r6 | 값 범위 50–800 K | K 변환 뒤 50 ≤ T ≤ 800 밖이면 제외 |
| r7 | 구조 | `SMILES` 필드를 RDKit(2026.09.1)으로 읽는다. 읽히지 않거나, `.`가 있거나(여러 조각), 무거운 원자 > 100이면 제외 |

`SuspiciousValue`, `FromLiterature`는 PREREG 규칙에 없는 필드라 거르는 데 쓰지 않고 개수만 보고한다.

## 3. 분자 단위 규칙

| 순서 | PREREG 규칙 | 적용 방법 |
|---|---|---|
| m1 | 한 분자의 여러 기록 | RDKit 이성질체 canonical SMILES별로 묶어 중앙값. 기록 최대 − 최소 > 50 K이면 그 분자 제외 |
| m2 | mp validation·test와 겹침 | RDKit InChIKey(이 환경에서 다시 계산) 첫 블록이 `aux_exclusion_keys.csv`(두 분할의 validation·test 첫 블록 7,076개; 분할 파일에서 다시 만들어 일치 확인)에 있으면 제외 |
| m3 | mp train과 겹침 | 첫 블록이 두 분할 중 하나의 mp train에 있으면 제외(mp 라벨 유지). InChIKey 전체가 mp train과 같은 분자는 외부 값 − mp 라벨 차이를 진단용으로 따로 보고한다 |

참고: 두 분할의 train ∪ validation ∪ test = mp 전체이므로 m2 + m3은 'mp 전체와 첫 블록이 겹치는 분자 제외'와 같다.

## 4. 고품질 부분집합 (단계 5 변형)

분자 단위 최종본 중 규칙 통과 기록 수 ≥ 2이고 기록 범위(최대 − 최소) ≤ 5 K인 분자.
