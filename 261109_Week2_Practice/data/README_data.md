# 분자 물성 데이터베이스 (ML 스터디 2026-11)

CEPL 학부연구생 ML 스터디용 정제 데이터. 물성 하나당 CSV 하나이며, 모든 파일의 형식이 같다.

| 열 | 뜻 |
|---|---|
| `smiles` | RDKit canonical SMILES (입체 표기 유지) |
| `value` | 목표값 (float). 단위는 `unit` 열 |
| `unit` | 단위 문자열 |
| `source` | 원 데이터셋. 여러 출처가 합쳐진 분자는 `A+B` |
| `T_K` | 온도 [K]. **viscosity.csv 에만 있다** |

원본은 `0.연구과제/Shipboard_RPB/Amine_RPB_shared_20261015/.../02_gnn_property_model/dat/<prop>/dataset.pkl`
(GNN 물성 모델 인계 패키지, 실제 학습에 쓴 데이터)이다. 단위는 `9.ml_gnn/code/data_pipeline/datamanager.py`
(`load_unit`, `load_<prop>_data`)와 저장된 값으로 확인했다. 다시 만들려면 `python build_db.py`.

## 1. 물성 목록

| 파일 | N (정제 후) | 단위 | 값 범위 | 출처 | 난이도 |
|---|---|---|---|---|---|
| `logP.csv` | 14,031 | log10 (옥탄올/물 분배계수) | −5.1 ~ 11.3 | OPERA [1] | **쉬움**. 데이터 많고 깨끗하며 원자 기여법으로도 잘 맞는 물성 |
| `density.csv` | 8,746 | g/mL | 0.55 ~ 4.01 | ORGANIC [2] | **쉬움**. 분자량·할로겐 수만으로도 기준선이 나온다 |
| `bp.csv` | 2,159 | K | 249 ~ 773 | ColbyDB (출처 미확인, 아래 주의) | **쉬움**. 크기·수소결합으로 대부분 설명된다. 데이터가 작아 과적합 연습에 좋다 |
| `logSolubility.csv` | 11,172 | log10(mol/L) | −13.2 ~ 2.4 | AqSolDB [3] + OPERA [1] | **중간**. 결정 상태(융점)에 좌우되어 구조만으로는 오차가 남는다 |
| `logPvap.csv` | 3,092 | log10(mmHg), 25 °C | −17.7 ~ 9.5 | ORGANIC [2] + OPERA [1] | **중간**. 값 범위가 27 자릿수라 극단값이 오차를 지배한다 |
| `HL.csv` | 1,536 | log10(atm·m³/mol) | −16.0 ~ 3.5 | OPERA [1] | **중간**. 데이터가 작고 할로겐 화합물 위주라 적용범위가 좁다 |
| `mp.csv` | 19,945 | K | 68 ~ 790 | Bradley Open MP [4] | **어려움**. 결정 패킹은 분자 그래프로 잘 안 보이고 측정 잡음이 크다 |
| `viscosity.csv` | 3,582 점 / 957 분자 | cP (+ `T_K`) | 0.10 ~ 26.5 cP, 227 ~ 404 K | J. Cheminform. 2024 SI [5] | **어려움**. 온도 의존, 분자당 여러 점 → 분할을 분자 단위로 해야 한다 |
| `pKa.csv` | 6,857 | pKa (무차원) | 0.99 ~ 12.0 | pkasolver [6] + mlmpka [7] | **어려움**. 이온화 자리마다 값이 다르다 (아래 주의) |
| `G.csv` · `H.csv` | 289,067 | kcal/mol (절대 에너지) | −4.0×10⁵ ~ −3×10² | St. John 2020 [8] | **심화**. 아래 참고 |

정제 과정 (`build_stats.csv`에 물성별 개수 기록): ① 값 NaN 제거 ② RDKit 파싱 실패 SMILES 제거
③ 같은 canonical SMILES(점도는 SMILES+온도)를 **중앙값**으로 합침.

| 물성 | 원본 N | NaN | 잘못된 SMILES | 중복 병합 | 기타 |
|---|---|---|---|---|---|
| logP | 14,031 | 0 | 0 | 0 | |
| logSolubility | 11,174 | 0 | 2 | 0 | |
| logPvap | 3,092 | 0 | 0 | 0 | |
| bp | 2,228 | 0 | 0 | 69 | |
| mp | 27,976 | 0 | 0 | 8,031 | `donotuse` 표시 행 제거 (원본 pkl 에서 이미 0건) |
| density | 8,907 | 0 | 0 | 161 | |
| viscosity | 3,582 | 0 | 0 | 0 | |
| pKa | 7,015 | 0 | 0 | 76 | 다중 이온화 분자 38종(82행) 제거 |
| HL | 1,536 | 0 | 0 | 0 | |
| G / H | 289,639 | 0 | 0 | 572 | |

## 2. 불러오기

```python
import pandas as pd
df = pd.read_csv('logP.csv')
X_smiles, y = df['smiles'], df['value']

visc = pd.read_csv('viscosity.csv')          # smiles, value, unit, source, T_K
```

권장 분할: 무작위 80/20 대신 **분자(또는 scaffold) 단위 분할**. 점도는 반드시 `smiles` 기준
`GroupKFold`로 나눈다. 같은 분자의 다른 온도가 train/test 에 갈라지면 성능이 부풀려진다.

## 3. 팀 배정 제안 (4팀)

| 팀 | 물성 | 이유 |
|---|---|---|
| 1 | `logP` + `density` | 입문. 데이터 많고 깨끗함. 지문(Morgan) + 랜덤포레스트로 시작 |
| 2 | `bp` + `logPvap` | 휘발성 짝. 끓는점과 증기압의 물리적 연결(Clausius–Clapeyron)을 확인해 볼 수 있다 |
| 3 | `logSolubility` + `mp` | 일반 용해도 식(logS ≈ 0.5 − 0.01(mp[°C] − 25) − logP)으로 세 물성을 엮어 볼 수 있다 |
| 4 | `viscosity` + `HL` | 온도 입력을 다루는 모델(분자 특징 + 1/T)과 작은 데이터셋 다루기 |

`pKa` 는 끝까지 간 팀의 심화 과제로, `G`·`H` 는 대규모 데이터(29만 행) 심화 과제로 둔다.

## 4. 주의 사항

- **단위.** `bp`·`mp` 는 **켈빈**이다. 인계 README 표에는 °C 로 적혀 있으나 코드가 +273.15 를 하고,
  메탄올이 337.98 K 로 저장되어 있다. `HL` 은 코드 주석이 atm·m³/mol 이지만 값이 −16 ~ 3.5 의 음수를
  포함하므로 **log10(atm·m³/mol)** 이다. 원본 학습에서 점도는 `_log` 모델이 ln(cP)로 변환해 학습했다
  (`modeling.select_transformation`). CSV 는 cP 그대로 두었으니 필요하면 `np.log(value)`.
- **viscosity** 는 온도 의존이다. 분자 957종에 3,582점(분자당 1 ~ 33점). 온도 없이 학습하면 안 된다.
  0.1 ~ 26.5 cP 범위로 고점도 액체는 없다. 원 문헌은 International Critical Tables, Lange's Handbook,
  PubChem 등이며 원본 pkl 의 `Reference` 열에 있다.
- **pKa** 는 이온화 자리(ion center)별 값이다. 원본의 `ioncenter` 열이 전부 비어 있어 자리를 구분할 수
  없으므로, 한 분자에 값이 여럿이고 차이가 1 이상인 38종(예: 6.3 과 9.3)은 **서로 다른 자리**로 보고
  제거했다. 중앙값을 내면 의미 없는 값이 된다. 남은 값은 분자당 하나의 대표 pKa 로 읽는다.
  산·염기 구분 정보가 없다는 점도 기억할 것.
- **mp** 는 잡음이 크다. 반복 측정이 있는 4,679종 중 410종은 값 차이가 10 K 이상, 86종은 50 K 이상이다.
  중앙값으로 합쳤지만 이 잡음이 오차 하한을 정한다. `source` 는 모두 `Bradley_ONS_MP` 로 적었다
  (세부 출처 PHYSPROP·Alfa Aesar 등은 원본 `source` 열).
- **bp** 는 원본 pkl 에 `colby` 출처만 있다. 폴더에 OPERA BP 파일(`TR_BP_4077.sdf` 등)이 있지만 학습
  데이터에는 들어가지 않았다(인계 README §6: `dat/opera/` 경로 부재). ColbyDB.sdf 의 원 문헌은 확인하지 못했다.
- **logP** 의 인계 README 는 Chen 2021 / Wu 2018 데이터를 언급하지만 실제 pkl 은 OPERA TR/TST 만 쓴다.
  원본의 `LogP`·`logP` 두 열은 같은 값이며(`datamanager.load_logP_data`), 학습 목표는 `logP` 열이다.
- **density** 의 master `density.csv` 는 ORGANIC 저장소 원본과 내용이 다르다(수정본). `duplicate_inventory.md` 참고.
- **G / H (심화)** 는 M06-2X/def2-TZVP 절대 깁스 에너지·엔탈피(Hartree × 627.509)다. 값이 원자 구성으로
  거의 정해지므로 R² 가 높게 나와도 화학을 배운 것이 아니다. 닫힌 껍질 분자 43,271종
  (`source = StJohn2020_molecule`)과 라디칼 조각 245,796종(`StJohn2020_fragment`)이 섞여 있다.
- **넣지 않은 것 (심화 자료, 원본 폴더에만 있음)**: `bde`·`bdfe`(결합 단위 해리 에너지, `bdfe` 는 `bde`
  사본일 가능성 — 인계 README §6), `G_bonds`·`H_bonds`(결합 단위), `sp`(COSMO 시그마 프로파일 배열),
  `price`(Molport 가격), `lowmp`(`mp` 의 500 K 이하 부분집합).

## 5. 출처

1. Mansouri, K., Grulke, C. M., Judson, R. S. & Williams, A. J. OPERA models for predicting physicochemical
   properties and environmental fate endpoints. *J. Cheminform.* **10**, 10 (2018). https://github.com/kmansouri/OPERA
2. Sanchez-Lengeling, B. et al. Optimizing distributions over molecular space: an objective-reinforced generative
   adversarial network for inverse-design chemistry (ORGANIC). *ChemRxiv* (2017).
   https://github.com/aspuru-guzik-group/ORGANIC (data/datasets)
3. Sorkun, M. C., Khetan, A. & Er, S. AqSolDB, a curated reference set of aqueous solubility and 2D descriptors
   for a diverse set of compounds. *Sci. Data* **6**, 143 (2019).
4. Bradley, J.-C., Williams, A. & Lang, A. Jean-Claude Bradley Open Melting Point Dataset. figshare (2014).
5. *J. Cheminform.* (2024) 논문 `13321_2024_820` 의 Additional file 2 (`13321_2024_820_MOESM2_ESM.csv`).
   논문 제목은 원문으로 확인하지 못했다.
6. Mayr, F., Wieder, M., Wieder, O. & Langer, T. Improving small molecule pKa prediction using transfer learning
   with graph neural networks. *Front. Chem.* **10**, 866585 (2022). (`pkasolver_exp`·`_lit`·`_novartis`)
7. Baltruschat, M. & Czodrowski, P. Machine learning meets pKa. *F1000Research* **9**, 113 (2020). (`mlmpka_lit`)
8. St. John, P. C. et al. Quantum chemical calculations for over 200,000 organic radical species and 40,000
   associated closed-shell molecules. *Sci. Data* **7**, 244 (2020).

학생 발표·보고서에 쓸 때는 위 원 문헌을 인용한다. 5번과 ColbyDB 는 원문 확인 후 인용할 것.
