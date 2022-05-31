import os 
import subprocess

def main():
    start = 2
    n_seeds = 9
    end = n_seeds + start

    skew = True # irrelevant for RegularFunc tests
    for seed in range(start, end, (end - start)//n_seeds): 
        subprocess.run([ 
            'python', 
            'air_pollution_test.py',
            'RegularFunc',
            str(skew), 
            str(seed), 
            f'tests/pollution_RegularFunc_{skew=}{seed}'
        ])

    assert False
    for skew in [True, False]:
        sk = 'skew' if skew else 'any'
        for seed in range(start, end, (end - start)//n_seeds): 
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