"""[추가] train.py 로그에서 validate MAE 최저값(* 표시 줄)과 그 에폭을 뽑는다 (train.py 는 validate 값을 summary.csv 에 쓰지 않는다)."""
import re, sys, glob, json
def best_val(path):
    best = None
    for line in open(path, encoding='utf-8', errors='ignore'):
        m = re.search(r'epoch\s+(\d+)\s+train loss ([\d.]+)\s+val MAE ([\d.]+).*\*\s*$', line)
        if m:
            ep, v = int(m.group(1)), float(m.group(3))
            if best is None or v < best[1] - 1e-9 or True:
                best = (ep, v)          # '*' 줄은 갱신된 최저값이므로 마지막 '*' 가 최저
    return best
if __name__ == '__main__':
    out = {p: best_val(p) for p in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else 'logs/*.log'))}
    print(json.dumps(out, indent=1, ensure_ascii=False))
