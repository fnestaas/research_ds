import os 
import subprocess

def main():
    start = 1500
    n_seeds = 100
    end = n_seeds + start
    step = (end - start)//n_seeds

    for seed in range(start, end, step):
        for skew in [False, True]:
            subprocess.run([ 
                'python', 
                'air_pollution_test.py',
                'PDEFunc',
                str(skew), 
                str(seed), 
                f'tests/pollution_PDEFunc_{skew=}{seed}'
            ])
        skew = True
        subprocess.run([ 
            'python', 
            'air_pollution_test.py',
            'RegularFunc',
            str(skew), 
            str(seed), 
            f'tests/pollution_RegularFunc_{skew=}{seed}'
        ])
if __name__ == '__main__':
    main()