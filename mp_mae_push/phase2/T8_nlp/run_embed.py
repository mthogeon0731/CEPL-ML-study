"""run_embed.py <texts.csv> <text_col> <hf_model> <out.npy> [threads]

sentence-transformers 로 문장 임베딩(float32, L2 정규화)을 만든다. 2,000개 단위로 중간 저장해 이어 할 수 있다.
비용(벽시계·CPU 시간·문장당 ms)은 <out>.cost.json 에 남긴다.
"""
import json
import os
import sys
import time

os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('HF_HOME', os.path.abspath('T8_nlp/hf_cache'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

src, col, model_name, out = sys.argv[1:5]
threads = int(sys.argv[5]) if len(sys.argv) > 5 else 4
torch.set_num_threads(threads)
texts = pd.read_csv(src, keep_default_na=False)[col].astype(str).tolist()
proc = psutil.Process()
c0 = sum(proc.cpu_times()[:2])
t0 = time.time()
model = SentenceTransformer(model_name, device='cpu')
t_load = time.time() - t0
print('loaded', model_name, 'max_seq_length', model.max_seq_length, round(t_load, 1), 's', flush=True)
tok = model.tokenizer
ntok = [len(tok(t, add_special_tokens=True)['input_ids']) for t in texts]
trunc = float(np.mean(np.array(ntok) > model.max_seq_length))
part_dir = out + '.parts'
os.makedirs(part_dir, exist_ok=True)
CH = 2000
t1 = time.time()
for i in range(0, len(texts), CH):
    pf = os.path.join(part_dir, f'{i:06d}.npy')
    if os.path.exists(pf):
        continue
    E = model.encode(texts[i:i + CH], batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=False).astype(np.float32)
    np.save(pf, E)
    print(i + len(E), 'done', round(time.time() - t1), 's', flush=True)
t_enc = time.time() - t1
E = np.concatenate([np.load(os.path.join(part_dir, f)) for f in sorted(os.listdir(part_dir))])
assert len(E) == len(texts), (len(E), len(texts))
np.save(out, E)
cost = dict(model=model_name, n=len(texts), dim=int(E.shape[1]), threads=threads, load_s=round(t_load, 1),
            encode_wall_s=round(t_enc, 1), ms_per_text=round(1000 * t_enc / len(texts), 2),
            cpu_core_h=round((sum(proc.cpu_times()[:2]) - c0) / 3600, 4),
            peak_rss_gb=round(proc.memory_info().rss / 2 ** 30, 3), tokens_mean=float(np.mean(ntok)),
            tokens_max=int(np.max(ntok)), frac_truncated=trunc, max_seq_length=int(model.max_seq_length),
            note='encode_wall_s 는 이번 실행에서 새로 인코딩한 조각만의 시간(이어 하기 시 과소)')
json.dump(cost, open(out + '.cost.json', 'w'), indent=1)
print('DONE', cost, flush=True)
