import os 
import subprocess

SCRIPT_NAME = 'breast_cancer.py'
FOLDER_NAME = 'cancer'

def main():
    # 1510 uses a func where we also divide by norm(x)
    start = 2511
    n_seeds = 100
    end = n_seeds + start
    step = (end - start)//n_seeds

    for seed in range(start, end, step):
        for skew in [False, True]: # False crashes
            subprocess.run([ 
                'python', 
                SCRIPT_NAME,
                'PDEFunc',
                str(skew), 
                str(seed), 
                f'tests/{FOLDER_NAME}_PDEFunc_{skew=}{seed}'
            ])
        skew = True
        subprocess.run([ 
            'python', 
            SCRIPT_NAME,
            'RegularFunc',
            str(skew), 
            str(seed), 
            f'tests/{FOLDER_NAME}_RegularFunc_{skew=}{seed}'
        ])
    
if __name__ == '__main__':
    main()