# 외부 데이터·사전학습 모델·패키지 가용성

조회일 2026-11-20. 이 단계에서는 아무것도 내려받지 않았고 메타데이터(figshare API, PyPI API, GitHub API)와 검색 결과만 조회했다. PubChem은 접속 시험으로 아스피린 2건(CID 조회 1건, InChIKey 조회 1건)만 조회했다.

## 2단계에서 승인이 필요한 것

| 종류 | 대상 | 이유 |
|---|---|---|
| 네트워크 허용 | huggingface.co (모델 파일 호스트 포함) | MoLFormer, ChemBERTa, 텍스트 임베딩 모델 다운로드. 현재 차단 |
| 네트워크 허용 | zenodo.org | CheMeleon 가중치(34.9 MB). 현재 차단 |
| 네트워크 허용 | ochem.eu | Tetko 2014 세트(OCHEM·Enamine). 현재 차단. 등록이 필요할 수 있음 |
| 외부 데이터 사용 | Tetko 2016 특허 MP 세트 (figshare, 638 MB) | 1 GB 미만이지만 외부 데이터. figshare는 접속 가능 |
| 외부 데이터 사용 | Tetko 2014 OCHEM·Enamine 세트 | 외부 데이터 |
| 외부 서비스 호출 | PubChem PUG-REST 약 2만 건(50–85분) | T8-N1 이름, 기억 누설 층화. 접속 가능 |
| 대량 LLM 호출 | T8-N3 서술 19,945건, T8-M4 파일럿 | 비용 가정 문서의 추정 참조 |
| 패키지 설치 | lightgbm, catboost, optuna, mordredcommunity, chemprop, thermo, codecarbon, transformers, sentence-transformers | wheel 합계 약 123 MB(의존성 제외). chemprop·transformers 계열은 별도 환경 권장 |

모든 다운로드는 1 GB 미만이다. 가장 큰 것은 특허 MP 세트 638 MB, 그다음이 MoLFormer 187 MB다.

## 데이터와 모델

| 분류 | 항목 | 공개 여부·받는 곳 | 크기 | 라이선스 | CPU 예상 시간(근거) | 주의점 | 확인 수준 |
|---|---|---|---|---|---|---|---|
| 외부 MP 데이터 | Bradley Open MP (반복측정 포함판) | 공개. figshare https://doi.org/10.6084/m9.figshare.1031637 (v2, 2014-05-20) | BradleyMeltingPointDataset.xlsx 2.23 MB | CC0 | — | 우리 mp.csv의 원천. 다른 트랙이 이미 받아 검증(측정 28,645행 → 19,945종) | figshare API 메타데이터 |
| 외부 MP 데이터 | Bradley Double Plus Good (정제판) | 공개. figshare https://doi.org/10.6084/m9.figshare.1031638 (v1, 2014-05-20) | BradleyDoublePlusGoodMeltingPointDataset.xlsx 0.31 MB | CC0 | — | 3,041종(Tetko 2014 기준). 우리 데이터에 대부분 포함 → 외부 학습 데이터로는 가치 없음, 고품질 부분집합 표지로만 사용 | figshare API 메타데이터 |
| 외부 MP 데이터 | Tetko 2016 특허 MP 세트 | 공개. figshare https://doi.org/10.6084/m9.figshare.2007426 (2015-12-09); OCHEM http://ochem.eu/article/99826 에도 있음 | melting_point.sdf 638 MB, pyrolysis_point.sdf 38.7 MB | figshare 표기 CC0 (논문 본문은 CC-BY라고 기술 → 불일치, 인용은 필수로 간주) | SDF 파싱·InChIKey 생성: 미확인. RDKit 기술자 217개는 우리 측정(19,945분자 434 s)에서 비례하면 약 24만 분자에 약 1.5시간(단일 스레드) | 1 GB 미만이지만 외부 데이터 사용이므로 사전 승인 대상. 학습 세트 241,958(비분해 228,173). 실험 재현성 σ 35 °C로 잡음이 큼. Bradley와 분자 중복 가능 → test·val과 InChIKey(전체·첫 블록) 겹침 제거 필수. 분해점 세트는 섞지 말 것 | figshare API 메타데이터 + 원문 |
| 외부 MP 데이터 | Tetko 2014 세트 (OCHEM 21,883 + Enamine 22,404 + Bradley 3,041 + Bergström 277) | 공개라고 명시. http://ochem.eu/article/55638 — OCHEM 문서상 “공개·자유 다운로드” 기록은 누구나 받을 수 있고, 그 밖의 데이터 다운로드는 등록(무료)과 공개 데이터 기여가 조건 | 미확인 | 미확인(OCHEM 약관 미열람) | 4.7만 분자 기술자 계산 약 17분(우리 측정 비례) | ochem.eu는 네트워크 허용 목록 밖이라 이 조사에서 열지 못함. OCHEM 세트의 4대 출처 중 하나가 ONS MP challenge(Bradley 계열)라 Bradley와 중복이 클 가능성이 높음. Enamine 22,404는 단일 프로토콜 측정이라 품질이 고르고 Bradley와 겹침이 적을 가능성(Tetko 2014: Enamine–Bradley 중복 764종) | 원문 + OCHEM 문서 검색 발췌; 다운로드 경로 직접 확인 못 함 |
| 사전학습 분자 모델 | CheMeleon (chemprop D-MPNN, Mordred 기술자 1,613개로 사전학습) | 공개. Zenodo https://doi.org/10.5281/zenodo.15460715 (chemeleon_mp.pt, 가중치 레코드 v2 — Zenodo 레코드 페이지 검색 결과로 DOI와 파일 크기 확인. 코드 릴리스의 DOI 10.5281/zenodo.16821655 는 별개 레코드); chemprop ≥ 2.2에서 --from-foundation CheMeleon; 코드 https://github.com/JacksonBurns/chemeleon | 34.9 MB (매개변수 9.3M) | MIT | README: 미세조정 1회 “1 GPU-시간보다 훨씬 짧음”. CPU 시간은 미확인 → 파일럿 필요(참고: 수업 GIN 5k 표본 1회 242–289 s). 얼린 인코더의 2,048차원 지문 + GBDT는 순전파 1회라 저렴 | zenodo.org는 허용 목록 밖(접근 승인 필요). chemprop·lightning 설치 필요 | 검색 발췌(Zenodo·chemprop 문서) + GitHub API |
| 사전학습 분자 모델 | MoLFormer-XL-both-10pct (IBM) | 공개. Hugging Face ibm-research/MoLFormer-XL-both-10pct | model.safetensors 187 MB (저장소 전체 375 MB), 매개변수 46.8M | Apache-2.0 | CPU 추론 가능(transformers). 속도 보고치 미확인. 임베딩 768차원 | huggingface.co는 허용 목록 밖. trust_remote_code=True(저장소의 사용자 코드 실행)가 필요 → 코드 검토 후 사용. SMILES 최대 202 토큰 | 검색 발췌(모델 카드) |
| 사전학습 분자 모델 | ChemBERTa-2 77M-MTR / 77M-MLM (DeepChem) | 공개. Hugging Face DeepChem/ChemBERTa-77M-MTR, DeepChem/ChemBERTa-77M-MLM | MTR 저장소 14.1 MB; MLM은 미확인 | 미확인 | 작은 모델이라 CPU 추론 가능할 것으로 보이나 속도 미확인 | “77M”은 사전학습 분자 수이지 매개변수 수가 아님. huggingface.co 차단으로 모델 카드 직접 확인 못 함 | 검색 발췌 |
| T8 텍스트 임베딩 | sentence-transformers/all-MiniLM-L6-v2 | 공개. Hugging Face | model.safetensors 90.9 MB, 매개변수 22.7M, 임베딩 384차원, 입력 최대 256 토큰 | Apache-2.0 | 문장당 CPU 2–8 ms라는 2차 보고(출처 간 불일치) → 19,945문장 약 1–3분으로 추정. 긴 템플릿 서술(100–200 토큰)과 15 W급 CPU에서는 더 느릴 수 있어 실측 필요 | 범용 영어 문장 모델. 화학 지식은 없음(Sadeghi 2024, Kristiadi 2024 참조) | 검색 발췌 |
| T8 텍스트 임베딩 | BAAI/bge-small-en-v1.5 | 공개. Hugging Face | 매개변수 33M, 임베딩 384차원(파일 크기 미확인, FP32 기준 약 130 MB로 계산) | MIT | MiniLM-L6보다 다소 느림(수치 미확인) | 검색용 모델이라 질의 접두어 규칙이 있음(문서 임베딩에는 불필요) | 검색 발췌 |
| T8 텍스트 임베딩 | sentence-transformers/all-mpnet-base-v2 | 공개. Hugging Face | 매개변수 약 109M, 임베딩 768차원, 약 438 MB(2차) | 미확인 | MiniLM-L6보다 약 5배 느림(라이브러리 문서의 표현, 2차) | 품질 상향 비교군으로만 | 검색 발췌 |
| T8 텍스트 임베딩 | 화학 텍스트 특화 임베딩 모델 | 미확인. IUPAC 이름 전용 언어모델은 문헌에 있음(IUPAC-GPT, Mao 2025)이나 가중치 공개 여부를 확인하지 못함 | — | — | — | huggingface.co 차단으로 모델 검색을 못 함. SMILES 전용 모델(ChemBERTa, MoLFormer)은 자연어 서술에는 맞지 않음 | 미확인 |
| T8 이름(N1) | PubChem PUG-REST IUPACName 조회 | 공개, 키 불필요. 이 샌드박스에서 접속됨(시험 2건 중 InChIKey 조회 0.33 s). 예: https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/<InChIKey>/property/IUPACName/JSON | 응답 수백 바이트/건 | 미국 정부 공공 데이터(약관 미열람) | 제한: 초당 5회 이하, 분당 400회 이하, 분당 서버 실행 300초 이하. 19,945분자를 1건씩 조회하면 분당 400회 상한에서 약 50분, 초당 4회로 여유를 두면 약 83분 | InChIKey가 PubChem에 없는 분자는 이름이 비게 됨(빈칸으로 두고 표시). 초과 시 HTTP 503. 조회한 이름에 관용명이 섞이지 않도록 IUPACName 속성만 사용. PubChem에 등재됐는지 자체가 “유명도” 표지이므로 기억 누설 층화 변수로 재사용 가능 | 직접 시험 + NAR 2018 논문 발췌(https://doi.org/10.1093/nar/gky294) |
| T8 서술(N3) | GPT-MolBERTa 프롬프트·서술 데이터 | 프롬프트는 공개(GitHub data_collection/data_generation.py). 서술 데이터·사전학습 가중치는 Google Drive 링크로 공개. https://github.com/Suryanarayanan-Balaji/GPT-MolBERTa | 저장소 566 KB; Drive 파일 크기 미확인 | MIT | — | 프롬프트는 SMILES 하나를 주고 설명을 생성하라는 한 문장이고 물성 금지 지시가 없음(gpt-3.5-turbo, temperature 0, 최대 2,048 토큰). 그대로 쓰면 물성 누설 → 우리는 구조만 쓰도록 지시문을 새로 써야 함. 서술 데이터는 MoleculeNet 9종 대상이라 Bradley 분자는 없음 | GitHub API(README·스크립트) |
| T8 서술(N3) | CaR / LLM4Mol 코드 | 공개. https://github.com/ChnQ/LLM4Mol | 저장소 539 KB | 라이선스 표기 없음 | — | 라이선스가 없어 코드 재사용은 피하고 방법만 참고 | GitHub API |

## Bradley 데이터와 외부 세트의 중복

- Tetko 2014의 OCHEM 세트는 출처 네 곳 중 하나가 ONS MP challenge(Bradley가 주도한 공개 수집)다. 따라서 Bradley Open MP와 분자·측정값이 상당수 겹칠 것으로 봐야 한다. 겹치는 분자의 값은 독립 측정이 아닐 수 있으므로 '외부 라벨로 우리 라벨을 검증'하는 용도로 쓰면 순환 검증이 된다.
- Enamine 세트(22,404)는 한 회사가 한 프로토콜로 잰 값이라 Bradley와 출처가 다르다. Tetko 2014는 Enamine과 Bradley 정제 세트의 중복을 764종으로 보고했다.
- 특허 세트는 USPTO 문서에서 추출한 것이라 출처는 독립이지만 같은 분자가 들어 있을 수 있다. 추가 학습 데이터로서의 가치는 1단계 학습 곡선의 기울기(데이터 2배당 MAE 감소)로 판단하고, 참고로 우리 기준선(약 2만 분자)은 27.5 K이고 문헌의 23만 분자 모델은 21–23 K다(MP 문헌 표; 정제 기준이 달라 직접 비교는 아님).
- 어느 세트든 test·val 분자와 InChIKey 전체·첫 블록이 겹치는 행을 지운 뒤 학습에만 넣는다.

## Python 패키지 (현재 미설치)

PyPI 메타데이터 기준이다. conda-forge 버전은 api.anaconda.org가 차단돼 확인하지 못했다. '설치 크기'는 확인하지 못해 wheel 크기만 적었다(설치 후에는 보통 더 커진다).

| 패키지 | PyPI 최신 | PyPI 게시일 | requires_python | Python 3.11 | Windows | 배포 형태 | wheel 크기(MB) | 주요 의존성(PyPI 메타데이터) | conda-forge 버전 | 비고 |
|---|---|---|---|---|---|---|---|---|---|---|
| lightgbm | 4.7.0 | 2026-07-18 | >=3.10 | 예 | 예 | Windows amd64 wheel (cp311/abi3) | 1.4 | narwhals>=1.15, numpy>=1.21.3, scipy | 미확인(api.anaconda.org 차단) |  |
| catboost | 1.2.10 | 2026-02-18 | 표기 없음 | 예 | 예 | Windows amd64 wheel (cp311/abi3) | 100.2 | graphviz, matplotlib, numpy<3.0,>=1.16.0, pandas<4.0,>=0.24, scipy, plotly, six | 미확인(api.anaconda.org 차단) | wheel이 큼(100 MB) |
| optuna | 5.1.0 | 2026-11-16 | >=3.10 | 예 | 예 | 순수 Python wheel | 0.4 | alembic>=1.5.0, colorlog, numpy, packaging>=20.0, sqlalchemy>=1.4.2, tqdm, PyYAML | 미확인(api.anaconda.org 차단) | SQLite storage는 표준 라이브러리로 충분 |
| mordredcommunity | 2.0.7 | 2026-01-22 | >=3.9 | 예 | 예 | 순수 Python wheel | 0.2 | rdkit, six, numpy, networkx, packaging | 미확인(api.anaconda.org 차단) | RDKit 필요. 2D 기술자 약 1,600개, 계산 시간은 CheMeleon README 기준 100만 분자에 약 500 CPU-시간 → 2만 분자 약 10 CPU-시간(비례 추정) |
| chemprop | 2.3.1 | 2026-08-04 | <3.15,>=3.11 | 예 | 예 | 순수 Python wheel | 0.2 | lightning>=2.0, numpy, pandas, rdkit, scikit-learn, scipy, torch>=2.1, astartes[molecules], ConfigArgParse, rich, descriptastorus, cuik_molmaker_pin | 미확인(api.anaconda.org 차단) | torch ≥ 2.1, lightning 등 의존성이 많아 기존 mlstudy 환경의 torch를 바꿀 수 있음 → 별도 환경 권장. cuik_molmaker_pin의 Windows 지원은 미확인 |
| thermo | 0.6.1 | 2026-07-13 | >=3.9 | 예 | 예 | 순수 Python wheel | 6.4 | fluids>=1.3.1, scipy>=1.6.0, pandas, chemicals>=1.4.0 | 미확인(api.anaconda.org 차단) | Joback 등. chemicals·fluids 동반 설치 |
| codecarbon | 3.3.2 | 2026-10-23 | >=3.10 | 예 | 예 | 순수 Python wheel | 0.4 | arrow, authlib>=1.2.1, joserfc>=1.0.0, click, pandas>=2.3.3, pandas, prometheus_client, psutil>=6.0.0, py-cpuinfo, pydantic<3.0.0,>=2.0.0, nvidia-ml-py, rapidfuzz | 미확인(api.anaconda.org 차단) | Windows에서 CPU 전력 실측 가능 여부 미확인(안 되면 상수 전력 추정으로 동작) → P_cpu 상수 방식과 병기 |
| transformers | 5.23.0 | 2026-11-13 | >=3.11.0 | 예 | 예 | 순수 Python wheel | 13.3 | huggingface-hub<3.0,>=1.31.0, numpy>=1.17, packaging>=20.0, pyyaml>=5.1, regex>=2025.10.22, tokenizers<0.24.0,>=0.23.1, typer, safetensors>=0.8.0, tqdm>=4.60 | 미확인(api.anaconda.org 차단) | torch 필요. 모델 파일은 huggingface.co에서 받아야 함(차단 상태) |
| sentence-transformers | 6.3.0 | 2026-11-11 | >=3.10 | 예 | 예 | 순수 Python wheel | 0.8 | transformers<6.0.0,>=5.0.0, tokenizers>=0.19, huggingface-hub<3.0.0,>=1.3.0, torch>=2.2, numpy>=1.24.0, scikit-learn>=1.1.0, scipy>=1.0.0, typing_extensions>=4.10.0, tqdm>=4.0.0 | 미확인(api.anaconda.org 차단) | transformers ≥ 5, torch ≥ 2.2 필요 |
| torch | 2.15.0 | 2026-10-28 | >=3.11 | 예 | 예 | Windows amd64 wheel (cp311/abi3) | 126.7 | filelock, typing-extensions>=4.13.0, setuptools>=77.0.3, sympy>=1.13.3, networkx>=2.5.1, jinja2, fsspec>=0.8.5, cuda-toolkit[cublas,cudart,cufft,cufile,cupti,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.2.2, cuda-bindings<14,>=13.0.3, nvidia-cudnn-cu13==9.26.0.51, nvidia-cusparselt-cu13==0.8.1, nvidia-nccl-cu13==2.30.7 | 미확인(api.anaconda.org 차단) | 참고용(이미 설치됨). PyPI 최신 wheel 크기 |
| rdkit | 2026.9.2 | 2026-11-11 | 표기 없음 | 예 | 예 | Windows amd64 wheel (cp311/abi3) | 25.4 | numpy, Pillow | 미확인(api.anaconda.org 차단) | 참고용(이미 설치됨) |
