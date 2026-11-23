"""n3rep_generate.py : N3 보강 생성(이 세션 커널에서 실행한 코드 사본). 지시문·필터는 앞 실행과 동일.
필터는 앞 README 정의(숫자+°C·K·g/cm³·g/mL·mmHg·kPa·atm, 또는 melting/boiling/density)로 재구현 — 기존 3,600건 플래그와 15/15, 3,585/3,585 일치.
"""
import json, os, re, time
import numpy as np
PROP_RE = re.compile(r'(\d+(?:\.\d+)?\s*(?:°\s*C|K\b|g\s*/\s*cm(?:3|³|\^3)|g\s*/\s*mL|mmHg|kPa|atm\b))|\b(melting|boiling|density)\b', re.I)
RAW = 'T8_nlp_n3rep/n3rep_raw.jsonl'; PROG = 'T8_nlp_n3rep/n3rep_progress.csv'
CAP_CALLS = 3450; CAP_TOK = 1_900_000; CAP_USD = 0.9
PR = dict(inp=0.10e-6, out=0.50e-6, cr=0.01e-6, cw=0.20e-6)   # Haiku USD/token; 캐시 쓰기는 가정
# PROMPT = open(n3_prompt_template.txt).read(); 요청 = PROMPT.replace('{smi}', smiles), max_tokens 400, host.llm 기본 모델, 20건 묶음, max_concurrency 4
def ok(r): return (not r.get('error')) and r.get('stop_reason') == 'end_turn' and bool(r.get('text')) and not PROP_RE.search(r['text'])
# 순서: validation 1차(1,636) -> validation 재시도 1회 -> train k = min(1800, 3450 - 사용 호출 - 16)개(seed 0 순열 앞부분) -> train 재시도 1회.
# 매 20건 묶음마다 jsonl 에 추가 기록(500건마다 저장 조건보다 촘촘). 묶음 전 누적 토큰+20x560 <= 1.9M, 호출 <= 3,450, 비용 <= 0.9 USD 확인.
