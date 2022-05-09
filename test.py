import os 
import subprocess

def main():
    plot = False 
    track_stats = True 
    do_backward = True
    regularize = False 
    which_func = 'PDEFunc'

    for use_autodiff in [True]:
        for skew_pde in [False, True]:
            if skew_pde: sk = '_skew' 
            else: sk = ''
            if use_autodiff: ad = '_autodiff'
            else: ad = '_manual'
            dst = which_func + ad + sk
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
                dst
                ])

if __name__ == '__main__':
    main()