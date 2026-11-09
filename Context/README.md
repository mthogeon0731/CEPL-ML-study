# Context — 1주차 참고자료 (슬라이드 40–41쪽)

슬라이드 `261105_1주차_물성예측_슬라이드.pdf` 40쪽(참고 자료 1/2)·41쪽(참고 자료 2/2: ML·GNN 발전사)의 문헌을
공개 PDF(arXiv, OpenReview, 오픈액세스 저널, 대학 리포지터리)에서 내려받은 것.

- 파일명 규칙: `p40-NN_` / `p41-NN_` = 슬라이드 쪽 번호 + 그 쪽 목록에서의 순번 → 이름순 정렬 = 슬라이드 순서
- 유료 저널(ACS, IEEE, Elsevier, Nature 본지)은 공개 PDF가 없어 **미확보** → 아래 표의 DOI 링크로 학교 계정(서강대 도서관 프록시)에서 직접 내려받아 같은 규칙으로 저장
- 작성일 2026-11-06

## 파일 상태 (2026-11-06)

이 폴더에 **PDF 23개** (확보 목록 전부)가 들어 있다 — Chrome으로 내려받아 `Downloads`에서 복사·이름 정리까지 끝남. `Downloads`에 남은 사본(`p40-*`, `p41-*`, `infopapers_3165_1.pdf`, `Xu2019_…arXiv1810.00826.pdf`)은 지워도 된다.
미확보 10편은 아래 표의 DOI 링크로 학교 계정에서 받아 같은 이름 규칙(`p41-19_Wieder2020_….pdf` 등)으로 추가.

## 40쪽 — 참고 자료 1/2

| # | 문헌 | 이 스터디에서의 역할 | 상태 | 파일 / 링크 |
|---|---|---|---|---|
| 1 | 2024 하반기 연구실 파이썬 스터디 자료 (GNN 학습, 모델 학습, 스크리닝, BGIN 발표) | 연구실 코드·그림의 출처 | 연구실 내부 자료 | 연구실 공유 폴더에서 받기 |
| 2 | Park, D. et al., Sparse Bayesian LSTM networks for plasma etch processes, *Comput. Chem. Eng.* 2022, 159, 107696 | 베이지안 층(가중치 분포) 방법, 공정 서로게이트 예 (30, 33쪽) | 미확보 (Elsevier) | https://doi.org/10.1016/j.compchemeng.2022.107696 |
| 3 | Xu, K., Hu, W., Leskovec, J., Jegelka, S., How Powerful are Graph Neural Networks?, ICLR 2019 | **GIN** — 연구실 모델의 뼈대, sum 집계·WL test (21쪽) | 확보 | `p40-03_Xu2019_HowPowerfulAreGNNs_GIN.pdf` (arXiv:1810.00826) |
| 4 | Blundell, C. et al., Weight Uncertainty in Neural Networks, ICML 2015 | **BGIN**의 Bayes by Backprop (22쪽) | 확보 | `p40-04_Blundell2015_WeightUncertaintyInNNs_BayesByBackprop.pdf` (arXiv:1505.05424) |
| 5 | Rogers, D., Hahn, M., Extended-Connectivity Fingerprints, *J. Chem. Inf. Model.* 2010, 50, 742 | **Morgan 지문(ECFP)** (17쪽) | 미확보 (ACS) | https://doi.org/10.1021/ci100050t |
| 6 | Bemis, G. W., Murcko, M. A., The Properties of Known Drugs. 1. Molecular Frameworks, *J. Med. Chem.* 1996, 39, 2887 | **scaffold split**의 Murcko 골격 (14쪽) | 미확보 (ACS) | https://doi.org/10.1021/jm9602928 |
| 7 | Wildman, S. A., Crippen, G. M., *J. Chem. Inf. Comput. Sci.* 1999, 39, 868 | Crippen logP = RDKit `MolLogP` 기술자 (16, 38쪽) | 미확보 (ACS) | https://doi.org/10.1021/ci990307l |
| 8 | Mansouri, K. et al., OPERA models for predicting physicochemical properties, *J. Cheminform.* 2018, 10, 10 | 실습 logP 데이터 출처 (37쪽) — mp 데이터 출처는 `data/README_data.md` 확인 | 확보 (OA) | `p40-08_Mansouri2018_OPERA_JCheminform.pdf` |
| 9 | Leskovec, J., Stanford CS224W, Lecture 2 Node Embeddings / Lecture 3 GNN | 그래프 임베딩·GNN 유도 (21쪽 "자세한 유도") | 확보 (2026 가을 학기 판) | `p40-09a_Leskovec_CS224W_Lecture02_NodeEmbeddings.pdf`, `p40-09b_Leskovec_CS224W_Lecture03_GNN.pdf` |

## 41쪽 — 참고 자료 2/2: ML·GNN 발전사 (27–31쪽)

| # | 문헌 | 계보에서의 위치 | 상태 | 파일 / 링크 |
|---|---|---|---|---|
| 1 | Gori, Monfardini, Scarselli, A new model for learning in graph domains, IJCNN 2005 | GNN 개념의 시작 | 미확보 (IEEE) | https://doi.org/10.1109/IJCNN.2005.1555942 |
| 2 | Scarselli, F. et al., The graph neural network model, *IEEE Trans. Neural Netw.* 2009, 20, 61 | GNN 정식화 (29쪽 Fig. 1) | 확보 (UOW 리포지터리 저자판) | `p41-02_Scarselli2009_TheGraphNeuralNetworkModel_IEEE_TNN.pdf` ← 원본명 `infopapers_3165_1.pdf` |
| 3 | Bruna, J. et al., Spectral networks and locally connected networks on graphs, ICLR 2014 | spectral GNN | 확보 | `p41-03_Bruna2014_SpectralNetworksOnGraphs.pdf` (arXiv:1312.6203) |
| 4 | Duvenaud, D. et al., Convolutional networks on graphs for learning molecular fingerprints, NeurIPS 2015 | 화학 적용 — 신경망 지문 | 확보 | `p41-04_Duvenaud2015_ConvNetsOnGraphs_NeuralFingerprints.pdf` (arXiv:1509.09292) |
| 5 | Defferrard, M. et al., ChebNet, NeurIPS 2016 | spectral → 국소 필터 | 확보 | `p41-05_Defferrard2016_ChebNet_FastLocalizedSpectralFiltering.pdf` (arXiv:1606.09375) |
| 6 | Kipf, T. N., Welling, M., GCN, ICLR 2017 | **GCN** (mean 집계, 21쪽 비교 대상) | 확보 | `p41-06_Kipf2017_GCN_SemiSupervisedClassification.pdf` (arXiv:1609.02907) |
| 7 | Gilmer, J. et al., Neural message passing for quantum chemistry, ICML 2017 | **MPNN** 틀 — 메시지 전달 일반화 | 확보 | `p41-07_Gilmer2017_MPNN_NeuralMessagePassingForQuantumChemistry.pdf` (arXiv:1704.01212) |
| 8 | Hamilton, W. et al., GraphSAGE, NeurIPS 2017 | 집계 함수·귀납 학습 | 확보 | `p41-08_Hamilton2017_GraphSAGE_InductiveRepresentationLearning.pdf` (arXiv:1706.02216) |
| 9 | Schütt, K. T. et al., SchNet, NeurIPS 2017 | 3D 거리 기반 | 확보 | `p41-09_Schutt2017_SchNet.pdf` (arXiv:1706.08566) |
| 10 | Veličković, P. et al., Graph attention networks, ICLR 2018 | attention 집계 | 확보 | `p41-10_Velickovic2018_GAT_GraphAttentionNetworks.pdf` (arXiv:1710.10903) |
| 11 | Xu, K. et al., How powerful are graph neural networks?, ICLR 2019 | GIN (40쪽 #3과 동일) | 확보 | `p40-03_…` 참조 |
| 12 | Yang, K. et al., Analyzing learned molecular representations for property prediction (D-MPNN, Chemprop), *JCIM* 2019, 59, 3370 | **D-MPNN/Chemprop** (29쪽 Fig. 1) — 결합 중심 메시지 | 확보 (arXiv 판) | `p41-12_Yang2019_Chemprop_DMPNN_AnalyzingLearnedMolecularRepresentations.pdf` (arXiv:1904.01561) |
| 13 | Gasteiger (Klicpera) et al., DimeNet, ICLR 2020 | 방향(각도) 메시지 | 확보 | `p41-13_Gasteiger2020_DimeNet_DirectionalMessagePassing.pdf` (arXiv:2003.03123) |
| 14 | Hu, W. et al., Strategies for pre-training graph neural networks, ICLR 2020 | 자기지도 사전학습 (8쪽) | 확보 | `p41-14_Hu2020_StrategiesForPretrainingGNNs.pdf` (arXiv:1905.12265) |
| 15 | Ying, C. et al., Graphormer, NeurIPS 2021 | Transformer 계열 | 확보 | `p41-15_Ying2021_Graphormer_DoTransformersPerformBadlyForGraphs.pdf` (arXiv:2106.05234) |
| 16 | Satorras, V. G. et al., E(n) equivariant GNNs, ICML 2021 | 등변 GNN (3D) | 확보 | `p41-16_Satorras2021_EGNN_EnEquivariantGNNs.pdf` (arXiv:2102.09844) |
| 17 | Zhou, G. et al., Uni-Mol, ICLR 2023 / Lu, S. et al., Uni-Mol+, *Nat. Commun.* 2024, 15, 7104 | 3D 표현 (28쪽) — 이 스터디는 3D 안 다룸 | 확보 (OpenReview / OA) | `p41-17a_Zhou2023_UniMol_ICLR.pdf`, `p41-17b_Lu2024_UniMolPlus_NatCommun.pdf` |
| 18 | Li, L., Zhang, Y., Wang, G., Xia, K., Kolmogorov–Arnold graph neural networks for molecular property prediction, *Nat. Mach. Intell.* 2025, 7 | **KA-GNN** — 연구실의 다음 후보 (30쪽) | 확보 (arXiv 프리프린트 — 출판본과 내용 차이 가능) | `p41-18_Li2025_KAGNN_KolmogorovArnoldGNN_preprint.pdf` (arXiv:2410.11323); 출판본 https://doi.org/10.1038/s42256-025-01087-7 |
| 19 | Wieder, O. et al., A compact review of molecular property prediction with graph neural networks, *Drug Discov. Today Technol.* 2020, 37, 1 | GNN 연표 그림 (29쪽 Fig. 3) | 미확보 (ScienceDirect — 오픈액세스이나 봇 차단; 브라우저로 직접 받기) | https://doi.org/10.1016/j.ddtec.2020.11.009 |
| 20 | Wu, J. et al., Machine learning methods for pKa prediction of small molecules, *Drug Discov. Today* 2022 | 기술자·지문 경로 vs 그래프 경로 그림 (28쪽 Fig. 1) | 미확보 (Elsevier) | https://doi.org/10.1016/j.drudis.2022.103372 |
| 21 | Butler, K. T. et al., Machine learning for molecular and materials science, *Nature* 2018, 559, 547 | 계산화학 1–3세대 흐름 (27쪽 Fig. 1) | 미확보 (Nature) | https://doi.org/10.1038/s41586-018-0337-2 |
| 22 | LeCun, Y., Bengio, Y., Hinton, G., Deep learning, *Nature* 2015, 521, 436 | 다층 신경망·역전파 (27쪽 Fig. 1) | 미확보 (Nature) | https://doi.org/10.1038/nature14539 |
| 23 | Jiménez-Luna, J., Grisoni, F., Schneider, G., Drug discovery with explainable artificial intelligence, *Nat. Mach. Intell.* 2020, 2, 573 | **XAI 개요** (31쪽, 발표 5절 후분석·XAI) | 확보 (arXiv 프리프린트) | `p41-23_JimenezLuna2020_DrugDiscoveryWithXAI_preprint.pdf` (arXiv:2007.00523); 출판본 https://doi.org/10.1038/s42256-020-00236-4 |

## 발표 섹션별로 먼저 볼 것

| 발표 섹션 (`mp_presentation/`) | 먼저 볼 문헌 | 볼 곳 |
|---|---|---|
| 02 전처리·표현 | Rogers & Hahn (Morgan 지문 원리 — 미확보면 17쪽 그림으로 대체), Bemis & Murcko (scaffold) | 지문 반경·비트 충돌, 골격 정의 |
| 03 모델링 | Xu 2019 (GIN: 식 4.1, sum 집계가 mean/max보다 강한 이유 — Fig. 3), Gilmer 2017 (MPNN 틀), Kipf 2017 (GCN과의 차이) | 21쪽 수식과 대조 |
| 03 모델링 (BGIN 언급 시) | Blundell 2015 (가중치 분포, 식 1–3) | 22쪽 |
| 04 성능 | Yang 2019 (Chemprop: scaffold split 결과표 — random보다 나쁜 것이 정상) | 14, 24쪽 |
| 05 후분석·XAI | Jiménez-Luna 2020 (XAI 분류: 특징 기여, 원자 기여, 불확실성) | 31쪽, 4주차 예고 |
| 배경 (27–30쪽) | Butler 2018, Wieder 2020 (연표) | 발표에서는 한 줄 이상 쓰지 않기 |
