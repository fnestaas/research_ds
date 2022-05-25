import os 
import subprocess

def main():
    skew = True
    seed = 5678

    sk = 'skew' if skew else 'any'

    subprocess.run([ 
        'python', 
        'mnist_test.py',
        'NonRegularFunc',
        str(skew), 
        str(seed), 
        f'tests/mnist_run_{sk}{seed}'
    ])
    
if __name__ == '__main__':
    main()