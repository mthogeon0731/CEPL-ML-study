"""3단계 H19: 극단 MP 구간(y < 250 K, y > 550 K) 가중·oversampling.
변형: w2 (극단 행 가중치 2), w4 (가중치 4), os2 (극단 행을 한 번 더 복제해 2배; 가중치 없음).
설정·나무 수는 1단계 최선 결합 설정 그대로.
사용: python s3_h19.py --cfg lgb_r1 --fs F0+F3+F5 --splits random,scaffold --seeds 0 --variants w2,w4,os2"""
import argparse, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import t1blib as L

LO, HI = 250.0, 550.0


def is_ext(y):
    return (y < LO) | (y > HI)


def variant(v):
    if v.startswith('w'):
        k = float(v[1:])
        return dict(weight_fn=lambda yy, k=k: np.where(is_ext(yy), k, 1.0)), f'극단 행 가중치 {k:g}'
    if v == 'os2':
        return dict(oversample=lambda tr: tr[is_ext(L.y_all()[tr])]), '극단 행 1회 복제 (2배)'
    raise ValueError(v)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='lgb_r1')
    ap.add_argument('--fs', default='F0+F3+F5')
    ap.add_argument('--splits', default='random,scaffold')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--variants', default='w2,w4,os2')
    ap.add_argument('--budget', type=float, default=99.0)
    a = ap.parse_args()
    for v in a.variants.split(','):
        kw, desc = variant(v)
        for how in a.splits.split(','):
            for s in [int(x) for x in a.seeds.split(',')]:
                if L.track_cost() > a.budget:
                    print('예산 상한, 중단', flush=True); sys.exit(0)
                L.fit_val(f"h19_{v}_{a.cfg}_{a.fs.replace('+', '')}_{how}_s{s}", a.cfg, a.fs, how, s, stage='탐색',
                          data_variant=f'train, {desc}', notes=f'H19 {desc}', **kw)
    print('track cost', round(L.track_cost(), 4), flush=True)
