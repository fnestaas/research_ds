import os 
import subprocess

SCRIPT_NAME = 'breast_cancer.py'
FOLDER_NAME = 'cancer'

def main():
    # seeds 1002 to 1016 have tau = 1/10 and use standard training parameters
    # seeds 1017 to 1020 use a larger batch size (64) and learning rate (1e-2)
    start = 510
    n_seeds = 100
    end = n_seeds + start
    step = (end - start)//n_seeds

    for seed in range(start, end, step):
        for skew in [True]: # False crashes
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