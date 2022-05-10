import os 
import subprocess

def main():
    plot = False 
    track_stats = True 
    do_backward = True
    regularize = False 
    integrate = True
    skew_pde = True
    which_func = 'PDEFunc'

    for use_autodiff in [True]:
        for final_activation in ['sigmoid']: # ['swish', 'identity']:
            if skew_pde: sk = '_skew' 
            else: sk = ''
            # if use_autodiff: ad = '_autodiff'
            # else: ad = '_manual'
            # if integrate: it = '_integrate'
            # else: it = '_no_int'
            dst = which_func + final_activation + sk
            subprocess.run([
                'python', 
                'first_test.py', 
                str(track_stats), 
                which_func, 
                str(do_backward), 
                str(regularize), 
                str(plot), 
                str(use_autodiff), 
                str(skew_pde),
                str(integrate), 
                final_activation,
                dst
                ])

if __name__ == '__main__':
    main()