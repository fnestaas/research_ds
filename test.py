import os 
import subprocess

def main():
    plot = False 
    track_stats = True 
    do_backward = True
    regularize = False 
    integrate = True
    skew_pde = True
    final_activation = 'identity'
    which_func = 'PDEFunc'
    use_autodiff = True

    # for integrate in [True, False]: # skew matrices, integrate or no
    #     if integrate: it = '_integrate'
    #     else: it = '_no_integral'
    #     skew_pde = True
    #     sk = 'skew'
    #     for seed in range(10):
    #         dst = f'tests/{sk}{it}{seed}'
    #         subprocess.run([
    #             'python', 
    #             'first_test.py', 
    #             str(track_stats), 
    #             which_func, 
    #             str(do_backward), 
    #             str(regularize), 
    #             str(plot), 
    #             str(use_autodiff), 
    #             str(skew_pde),
    #             str(integrate), 
    #             final_activation,
    #             str(seed),
    #             dst
    #         ])

    for skew_pde in [False]: # integrate, but vary whether we use skew sym; already computed what happens if we have skew and integrate
        if skew_pde: sk = 'skew'
        else: sk = 'any'
        integrate = True
        it = '_integrate'
        for seed in range(5, 10):
            dst = f'tests/{sk}{it}{seed}'
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
                str(seed),
                dst
            ])
    
if __name__ == '__main__':
    main()