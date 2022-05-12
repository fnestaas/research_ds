from cmath import nan
import matplotlib.pyplot as plt 
import joblib 
import numpy as np 

# folders = [ 
#     'PDEFuncidentity_skew', 
#     'PDEFuncswish_skew',
#     'PDEFuncsigmoid_skew', 
#     'PDEFuncabs_skew'
# ]

folders = [
    'mnist_run_skew',
    # 'mnist_run_none'
]

fig, axs = plt.subplots(2, 3, figsize=(10, 15))
y_max = 0

for i, folder in enumerate(folders):
    adjoints = joblib.load('tests/' + folder + '/adjoint_norm.pkl')
    label=folder
    if len(adjoints) > 0:
        mean_var = [np.nanmean(np.quantile(adjoint, q=.79, axis=-1)/np.quantile(adjoint, q=.21, axis=-1)) for adjoint in adjoints]
        nan_mask = np.isnan(np.array([np.quantile(adjoint, q=.79, axis=-1)/np.quantile(adjoint, q=.21, axis=-1) for adjoint in adjoints]))
        num_nans = np.sum(nan_mask)
        print(f'got {num_nans/np.size(nan_mask)} nans\n')
        # mean_var = [np.mean(np.var(adjoint, axis=-1)) for adjoint in adjoints]
        epochs = np.arange(len(adjoints))
        y_max = max([y_max, 1.1*max(mean_var)])
        axs[0, 0].plot(
            epochs,
            mean_var, 
            label=label
        )
        sample = int(.75*len(adjoints))
        sample_adjoint = adjoints[sample][0, :]
        axs[0, 1].plot(
            np.arange(len(sample_adjoint)), 
            sample_adjoint,
            label=label
        )

        nfes = joblib.load('tests/' + folder + '/num_steps.pkl')
        mean_nfe = [np.mean(nfe) for nfe in nfes]
        epochs = np.arange(len(mean_nfe))
        axs[1, 0].plot( 
            epochs, 
            mean_nfe, 
            label=folder 
        )

        state_norm = joblib.load('tests/' + folder + '/state_norm.pkl')
        mean_state_norm_var = [np.mean(np.var(adjoint, axis=-1)) for adjoint in state_norm]
        axs[1, 1].plot( 
            epochs, 
            mean_state_norm_var, 
            label=folder
        )
        
        sample_state_norm = state_norm[sample][0, :]
        axs[1, 2].plot( 
            np.arange(len(sample_state_norm)),
            sample_state_norm,
            label=folder 
        )

axs[0, 0].legend()
axs[0, 1].legend()
axs[1, 0].legend()
axs[1, 1].legend()
axs[1, 2].legend()

axs[0, 0].set_title('adjoint norm score')
axs[0, 1].set_title('sample adjoint norm')
axs[1, 0].set_title('number of function evaluations')
axs[1, 1].set_title('variance of state norm')
axs[1, 2].set_title('sample state norm')

fig.tight_layout()

plt.show()

# save = input('do you want to save? (yes/no)')
# if save == 'yes':
#     fig.savefig('outputs/activation_functions.pdf')
