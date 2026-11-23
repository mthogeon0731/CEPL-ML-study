"""run_queue.py <queue_name> : 무거운 작업을 순서대로 하나씩 실행(동시 무거운 작업 2개 이하 유지). 선행 파일이 생길 때까지 기다린다."""
import os
import subprocess
import sys
import time

PY = sys.executable
PYTXT = os.environ.get('PYTXT', '')
Q = {
    'A': [  # XGBoost/Ridge 평가 (mp-push)
        ('T8_nlp/preds/seeds/N2_minilm_ridge_scaffold_n4000_val_seeds.csv',
         [PY, 'T8_nlp/run_reps.py', 'REF_R217', 'RDKit217', 'mp_mae_push/cache/X_desc217.npy']),
        (None, [PY, 'T8_nlp/run_reps.py', 'M2_R217F5', 'RDKit217+F5(101)', 'mp_mae_push/cache/X_desc217.npy+T8_nlp/features/F5_rules.npy']),
        ('T8_nlp/emb/n1_minilm.npy', [PY, 'T8_nlp/run_reps.py', 'N1_minilm', 'N1 PubChem IUPAC 이름 → all-MiniLM-L6-v2(384)', 'T8_nlp/emb/n1_minilm.npy']),
        ('T8_nlp/emb/n2_bge.npy', [PY, 'T8_nlp/run_reps.py', 'N2_bge', 'N2 템플릿 서술 → bge-small-en-v1.5(384)', 'T8_nlp/emb/n2_bge.npy']),
    ],
    'B': [  # 임베딩 (mp-text)
        ('T8_nlp/emb/n1_minilm.npy', [PYTXT, 'T8_nlp/run_embed.py', 'T8_nlp/n2_template_texts.csv', 'text', 'BAAI/bge-small-en-v1.5', 'T8_nlp/emb/n2_bge.npy', '4']),
    ],
}
for need, cmd in Q[sys.argv[1]]:
    while need and not os.path.exists(need):
        time.sleep(20)
    print('RUN', ' '.join(cmd[1:4]), time.strftime('%H:%M:%S'), flush=True)
    r = subprocess.run(cmd)
    print('END', r.returncode, time.strftime('%H:%M:%S'), flush=True)
print('QUEUE DONE', flush=True)
