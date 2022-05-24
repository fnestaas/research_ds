import os 
import subprocess

def main():
    """
    NB NB NB NB

    When I ran these tests earlier, I had a bug in the code, so the results for skew=False are actually as if I had used skew=True...
    """
    n_seeds = 10

    for skew in [True, False]:
        sk = 'skew' if skew else 'any'
        for seed in range(n_seeds): 
            subprocess.run([ 
                'python', 
                'mnist_test.py',
                'NonRegularFunc',
                str(skew), 
                str(seed), 
                f'tests/new_mnist_run_{sk}{seed}'
            ])
    
    skew = True 
    for seed in range(n_seeds):
        skew = True
        subprocess.run([ 
            'python', 
            'mnist_test.py',
            'RegularFunc',
            str(skew), 
            str(seed), 
            f'test/new_mnist_run_regfunc{seed}'
        ])
    
if __name__ == '__main__':
    main()