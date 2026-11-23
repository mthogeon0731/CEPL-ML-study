import subprocess, sys, os
sys.path.insert(0, 'T1b_combined/scripts'); sys.dont_write_bytecode = True
import t1blib as L
for how in ('random', 'scaffold'):
    L.fit_oof(f'oof_lgb_r1_F0F3F5_{how}_s0', 'lgb_r1', 'F0+F3+F5', how, 0, stage='최종', notes='최종 후보 OOF (T1 fold 규칙)')
for v, how in [('w4','random'),('w4','scaffold'),('os2','random'),('os2','scaffold')]:
    r = subprocess.run([sys.executable, '-u', 'T1b_combined/scripts/s3_h19.py', '--fs', 'F0+F3+F5', '--splits', how, '--seeds', '0', '--variants', v, '--budget', '6.88'])
print('track cost', L.track_cost(), flush=True)
