"""대기열 실행기 (래퍼 로그 파일 이름을 나누려고 따로 둔다): hyb:scaffold:0,hyb:scaffold:1,hyb:scaffold:2,graphonly:scaffold:1"""
import os, runpy, sys
sys.argv = ['train_hybrid.py', '--stage', 'main', '--threads', '1', '--budget_core_h', '8.0', '--runs', 'hyb:scaffold:0,hyb:scaffold:1,hyb:scaffold:2,graphonly:scaffold:1']
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_hybrid.py'), run_name='__main__')
