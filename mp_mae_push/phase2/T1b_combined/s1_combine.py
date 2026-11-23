"""1단계: 튜닝 설정(lgb_r1, xgb_r1) x 특징 묶음 x seed x 분할 → validation 예측 (work/s1_<cfg>_<fs>_<split>_s<seed>.npy).
나무 수 = T1 재적합 나무 수 그대로 (열이 늘어도 다시 정하지 않음).
사용: python s1_combine.py --cfgs lgb_r1 --fs F0,F0+F3,F0+F5,F0+F3+F5 --splits random,scaffold --seeds 0,1,2 --budget 4.5"""
import argparse, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1blib as L

ap = argparse.ArgumentParser()
ap.add_argument('--cfgs', required=True)
ap.add_argument('--fs', default='F0+F3,F0+F5,F0+F3+F5')
ap.add_argument('--splits', default='random,scaffold')
ap.add_argument('--seeds', default='0,1,2')
ap.add_argument('--budget', type=float, default=99.0, help='트랙 누적 CPU 상한 [core-h]; 넘으면 중단')
a = ap.parse_args()
for cfg in a.cfgs.split(','):
    for fs in a.fs.split(','):
        for how in a.splits.split(','):
            for s in [int(v) for v in a.seeds.split(',')]:
                if L.track_cost() > a.budget:
                    print('예산 상한 도달, 중단', flush=True); sys.exit(0)
                L.fit_val(f"s1_{cfg}_{fs.replace('+', '')}_{how}_s{s}", cfg, fs, how, s, stage='탐색',
                          notes='1단계 특징 묶음 결합; 나무 수 T1 재적합 값')
print('track cost', round(L.track_cost(), 4), flush=True)
