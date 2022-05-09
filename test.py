import os 
import subprocess

def main():
    # assert False, 'not working properly, fix bugs!'
    plot = False 
    track_stats = True 
    do_backward = True
    regularize = False 
    which_func = 'PDEFunc'

    for use_autodiff in [True, False]:
        for skew_pde in [True, False]:
            if skew_pde: sk = 'skew' 
            else: sk = ''
            if use_autodiff: ad = 'autodiff'
            else: ad = 'manual'
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

    
    
    """
    which_func = 'ODE2ODEFunc'
    do_backward = True 
    dst = 'odefunc_auto_backward'

    track_stats = True 
    regularize = False 
    plot = False 
    use_autodiff = True 

    subprocess.run([
        'python', 
        'first_test.py', 
        str(track_stats), 
        which_func, 
        str(do_backward), 
        str(regularize), 
        str(plot), 
        str(use_autodiff), 
        dst
        ])

    which_func = 'ODE2ODEFunc'
    do_backward = True 
    dst = 'odefunc_manual_backward'

    track_stats = True 
    regularize = False 
    plot = False 
    use_autodiff = True 

    subprocess.run([
        'python', 
        'first_test.py', 
        str(track_stats), 
        which_func, 
        str(do_backward), 
        str(regularize), 
        str(plot), 
        str(use_autodiff), 
        dst
        ])
    """


if __name__ == '__main__':
    main()