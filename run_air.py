import os 
import subprocess

def main():
    start = 21
    n_seeds = 100
    end = n_seeds + start
    step = (end - start)//n_seeds

    for seed in range(start, end, step):
        skew = True
        subprocess.run([ 
            'python', 
            'air_pollution_test.py',
            'RegularFunc',
            str(skew), 
            str(seed), 
            f'tests/pollution_RegularFunc_{skew=}{seed}'
        ])
        for skew in [True, False]:
            subprocess.run([ 
                'python', 
                'air_pollution_test.py',
                'PDEFunc',
                str(skew), 
                str(seed), 
                f'tests/pollution_PDEFunc_{skew=}{seed}'
            ])
    
if __name__ == '__main__':
    main()