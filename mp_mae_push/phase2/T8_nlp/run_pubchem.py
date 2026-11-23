"""run_pubchem.py : InChIKey → (CID, IUPACName) 와 CID → 동의어 수를 PubChem PUG-REST 묶음 POST 로 조회.

요청 간격 ≥ 0.3 s (초당 4회 이하), 503/429/시간초과는 지수 대기 후 재시도, 400은 묶음을 반으로 나눠 재시도.
중간 결과는 T8_nlp/pubchem_raw_*.jsonl 에 쌓아 끊겨도 이어 한다.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

BASE = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
OUT = 'T8_nlp'
_last = [0.0]
stats = dict(requests=0, retries=0, http_errors=0)


def post(path, data, tries=6):
    for k in range(tries):
        wait = 0.3 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        stats['requests'] += 1
        try:
            req = urllib.request.Request(BASE + path, data=urllib.parse.urlencode(data).encode(), method='POST')
            with urllib.request.urlopen(req, timeout=120) as r:
                return 200, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            stats['http_errors'] += 1
            if e.code == 404:
                return 404, None
            if e.code == 400:
                return 400, None
            stats['retries'] += 1
            time.sleep(min(60, 2 ** k * 2))
        except Exception:
            stats['retries'] += 1
            time.sleep(min(60, 2 ** k * 2))
    return -1, None


def prop_batch(keys, fh):
    st, J = post('/compound/inchikey/property/InChIKey,IUPACName/JSON', {'inchikey': ','.join(keys)})
    if st == 200:
        for d in J['PropertyTable']['Properties']:
            fh.write(json.dumps(d) + '\n')
        fh.write(json.dumps({'_done': keys}) + '\n')
    elif st == 404:
        fh.write(json.dumps({'_done': keys}) + '\n')
    elif st == 400 and len(keys) > 1:
        h = len(keys) // 2
        prop_batch(keys[:h], fh)
        prop_batch(keys[h:], fh)
    elif st == 400:
        fh.write(json.dumps({'_done': keys, '_bad': True}) + '\n')
    else:
        print('FAILED batch', keys[0], flush=True)
    fh.flush()


def main():
    s = pd.read_csv('mp_mae_push/splits/random.csv').sort_values('row')
    keys = sorted(set(s['inchikey'].dropna()))
    p1 = os.path.join(OUT, 'pubchem_raw_props.jsonl')
    done = set()
    if os.path.exists(p1):
        for ln in open(p1, encoding='utf-8'):
            d = json.loads(ln)
            if '_done' in d:
                done.update(d['_done'])
    todo = [k for k in keys if k not in done]
    print('keys', len(keys), 'todo', len(todo), flush=True)
    t0 = time.time()
    with open(p1, 'a', encoding='utf-8') as fh:
        for i in range(0, len(todo), 100):
            prop_batch(todo[i:i + 100], fh)
            if i % 2000 == 0:
                print('props', i, round(time.time() - t0), stats, flush=True)
    rows = [json.loads(ln) for ln in open(p1, encoding='utf-8')]
    props = pd.DataFrame([r for r in rows if 'CID' in r])
    cids = sorted(set(props['CID'].astype(int)))
    p2 = os.path.join(OUT, 'pubchem_raw_syn.jsonl')
    done2 = set()
    if os.path.exists(p2):
        for ln in open(p2, encoding='utf-8'):
            d = json.loads(ln)
            if '_done' in d:
                done2.update(d['_done'])
    todo2 = [c for c in cids if c not in done2]
    print('cids', len(cids), 'todo', len(todo2), flush=True)
    with open(p2, 'a', encoding='utf-8') as fh:
        for i in range(0, len(todo2), 100):
            b = todo2[i:i + 100]
            st, J = post('/compound/cid/synonyms/JSON', {'cid': ','.join(map(str, b))})
            if st == 200:
                for d in J['InformationList']['Information']:
                    fh.write(json.dumps({'CID': d['CID'], 'n_syn': len(d.get('Synonym', []))}) + '\n')
                fh.write(json.dumps({'_done': b}) + '\n')
            elif st in (404, 400):
                fh.write(json.dumps({'_done': b, '_status': st}) + '\n')
            else:
                print('FAILED syn batch', b[0], flush=True)
            fh.flush()
            if i % 2000 == 0:
                print('syn', i, round(time.time() - t0), stats, flush=True)
    json.dump(dict(stats, wall_s=round(time.time() - t0, 1)), open(os.path.join(OUT, 'pubchem_stats.txt'), 'w'))
    print('DONE', stats, flush=True)


if __name__ == '__main__':
    main()
