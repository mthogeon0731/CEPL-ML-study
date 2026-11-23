# DATA_CARD — 외부 녹는점 데이터 (T5)

작성 2026-11-21. 처리 규칙은 `mp_mae_push/phase2/prereg/PREREG.md` 4절(파싱 전 고정)을 그대로 따랐고, 필드 대응은 `RULES_applied.md`에 파싱 전에 적었다.

## 1. Tetko 2016 특허 녹는점 세트 (사용)

| 항목 | 값 |
|---|---|
| 제목(figshare) | Melting Point and Pyrolysis Point Data for Tens of Thousands of Chemicals |
| DOI | 10.6084/m9.figshare.2007426.v1 (figshare 게시 2015-12-09T14:07:12Z, version 1) |
| 관련 논문 | Tetko, Lowe, Williams (2016) J. Cheminform. 8:2 — 특허에서 텍스트 마이닝한 녹는점·분해점 |
| 사용 파일 | `melting_point.sdf` (figshare file id 3575841). `pyrolysis_point.sdf`(분해점, 38,729,853 B)는 규칙에 따라 받지 않음 |
| 받은 주소 | figshare API `https://api.figshare.com/v2/articles/2007426` 의 files 목록에서 name == melting_point.sdf 인 항목의 download_url = `https://ndownloader.figshare.com/files/3575841` (실제 전송은 figshare R2 저장소로 리다이렉트). 지시서 초안의 `files/3426968`은 쓰지 않음 |
| 다운로드 완료 | 2026-11-21T00:32:19Z (UTC), 58.5 s |
| 크기 | 637,981,904 B — figshare 표기 크기와 일치 ✓ |
| md5 | 20618014e0fe69ce7dbd1de87413e1f1 — figshare computed_md5와 일치 ✓ |
| sha256 | 4bfef8fb97dc65926cc15596d643f503eaeeea96a6b4ec1b36bda2a5ae5ad478 |
| 라이선스 | figshare 표기 CC0 1.0 (`https://creativecommons.org/publicdomain/zero/1.0/`). 1단계 가용성 조사에 따르면 논문 본문은 CC-BY로 기술 → 불일치이므로 **인용 필수로 간주**(논문과 figshare DOI 모두 인용) |
| 기록 수 | 228,174 (모두 QuantityType = MeltingPoint). 기록 단위 규칙을 통과한 194,865 기록은 모두 서로 다른 분자(같은 canonical SMILES의 기록이 2개 이상인 분자 0). 제외된 33,309 기록의 중복 여부는 확인하지 않음 |
| 필드 | Value, OriginalText, QuantityType, SMILES, StdInChIKey, Patent, Paragraph, SuspiciousValue(전부 false), FromLiterature(true 72) |

정제 단계별 행 수: `tables/patent_cleaning_steps.csv` (아래 3절에 옮김). 정제본: `tables/patent_clean.csv` (smiles, inchikey, value_K, n_rec, range).

## 2. Enamine 세트 (Tetko 2014, OCHEM article 55638) — 생략

| 시도 (2026-11-21 UTC) | 결과 |
|---|---|
| `https://ochem.eu/article/55638` (시스템 CA, 그리고 certifi CA 묶음 지정, 3회) | `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate` — 서버가 중간 인증서를 보내지 않는 것으로 보임 |
| `http://ochem.eu/article/55638` | HTTP 502 Bad Gateway |
| `https://www.ochem.eu/...` | 샌드박스 프록시 403(허용 목록 밖). 지시에 따라 접근 요청을 하지 않음 |

지시서대로 인증서 검증을 끄지 않고 생략했다. 라이선스(OCHEM 약관)도 확인하지 못했다.

## 3. 정제 단계별 행 수 (`tables/patent_cleaning_steps.csv`)

| 단위 | 단계·규칙 | 시작 수 | 제외 수 | 남은 수 |
|---|---|---|---|---|
| 기록 | SDF 전체 기록 | 228174 |  | 228174 |
| 기록 | r0 읽을 수 없는 기록 |  | 0 | 228174 |
| 기록 | r0 필드 가정 불일치 |  | 0 | 228174 |
| 기록 | r1 분해 표지 |  | 0 | 228174 |
| 기록 | r2 부등호·근사(>,<,≥,≤,~,ca.) |  | 6586 | 221588 |
| 기록 | r3 값 형식 불명 |  | 1 | 221587 |
| 기록 | r4 범위 폭 > 5 K |  | 7051 | 214536 |
| 기록 | r5 단위 불명확: ° 만 있음 |  | 19590 | 194946 |
| 기록 | r5 단위 °F |  | 2 | 194944 |
| 기록 | r5 단위 표기 없음 |  | 68 | 194876 |
| 기록 | r6 50–800 K 밖 |  | 1 | 194875 |
| 기록 | r7 여러 조각 |  | 1 | 194874 |
| 기록 | r7 RDKit 읽기 실패 |  | 3 | 194871 |
| 기록 | r7 무거운 원자 > 100 |  | 6 | 194865 |
| 기록 | r7 InChIKey 실패 |  | 0 | 194865 |
| 분자 | m1 전: 고유 분자(canonical SMILES) |  | 0 | 194865 |
| 분자 | m1 기록 범위 ≤ 50 K |  | 0 | 194865 |
| 분자 | m2 mp validation·test 첫 블록 제외 |  | 28 | 194837 |
| 분자 | m3 mp train 첫 블록 제외 (최종) |  | 36 | 194801 |
| 분자 | 고품질 부분집합 (기록 ≥ 2, 범위 ≤ 5 K) |  |  | 0 |

- 고품질 부분집합(기록 ≥ 2, 범위 ≤ 5 K)은 **0분자**다: 규칙 통과 기록 194,865개에 같은 분자의 반복 기록이 없다(제외된 기록의 중복은 미확인). 단계 5에서는 대체 정의(보고 범위 폭 ≤ 1 K, 60,660분자; `tables/patent_clean_width.csv`)를 썼다.
- r5(`°`만 있고 C가 없는 기록 19,590개, 8.6 %)는 규칙을 엄격히 적용한 결과다. 미국 특허 관례상 대부분 °C일 가능성이 높으나 PREREG를 바꾸지 않았다.
- RDKit으로 다시 계산한 InChIKey가 파일의 StdInChIKey와 같은 기록: 194,708 / 194,865 (99.92 %).
- `SuspiciousValue` = true 0건, `FromLiterature` = true 72건(거르지 않음).

## 4. mp와의 일치도 (진단용, 학습에는 안 씀)

InChIKey 전체가 두 분할 모두의 mp train에 있는 37분자: 외부 값 − mp 라벨 평균 −10.3 K, SD 56.9 K, 중앙값 0.0 K, 평균 절대 차 28.3 K, 절대 차 중앙값 4.0 K(n이 작아 잠정치). 특허 세트와 mp(Bradley)의 겹침은 첫 블록 기준 64분자뿐이라 두 데이터는 화학 공간이 거의 겹치지 않는다.

## 5. 라벨 분포 (`tables/label_distribution_mp_vs_patent.csv`)

| 통계 [K] | mp(19,945) | 특허 정제본(194,801) |
|---|---|---|
| q5 | 209.6 | 337.2 |
| q25 | 321.2 | 382.2 |
| q50 | 380.4 | 422.6 |
| q75 | 442.2 | 466.9 |
| q95 | 523.2 | 529.6 |
| mean | 378.5 | 426.4 |
| sd | 92 | 59 |

특허 값은 평균이 48 K 높고 폭이 좁다(특허에 보고되는 물질은 대부분 상온 고체). 이 분포 차이가 단계 5에서 random validation이 나빠진 주된 이유로 보인다(저융점 y < 300 K 부분집합 MAE가 추가량과 함께 26.1 → 34.4 K로 악화).

## 6. 파일

| 파일 | 내용 |
|---|---|
| `tables/patent_clean.csv` | 최종 정제본 194,801분자 (smiles, inchikey, value_K, n_rec, range) |
| `cache/X_ext_desc217.npy` | 정제본 행 순서의 RDKit 기술자 217개(float32, mp-push RDKit 2026.09.1, train_xgb.featurize 'desc') |
| `tables/patent_clean_width.csv` | 정제본 행별 보고 범위 폭(K; 단일 값은 빈칸) |
| `tables/patent_records_kept.csv` | 기록 단위 규칙 통과분(rec_id 포함) |
| `RULES_applied.md` | 파싱 전에 쓴 규칙 대응표 |
