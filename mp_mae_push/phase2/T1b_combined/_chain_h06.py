import subprocess, sys
for st, extra in [('es', []), ('refit', ['--seeds','0,1,2']), ('oof', ['--seeds','0'])]:
    r = subprocess.run([sys.executable, '-u', 'T1b_combined/scripts/s2_yalk.py', '--stage', st, '--fs', 'F0+F3+F5', '--splits', 'random,scaffold', '--budget', '6.2'] + extra)
    print('stage', st, 'rc', r.returncode, flush=True)
