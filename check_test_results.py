from cmath import nan
import matplotlib.pyplot as plt 
import joblib 
import numpy as np 
import models.NeuralODEClassifier as node
import jax.random as jrandom
import jax.numpy as jnp

# folders = [ 
#     'PDEFuncidentity_skew', 
#     'PDEFuncswish_skew',
#     'PDEFuncsigmoid_skew', 
#     'PDEFuncabs_skew'
# ]

folders = [
    'new_initialization{SKEW_PDE}',
    'new_initializationTrue',
    # 'just_a_test',
    # 'skew_integrate0',
    # 'cifar10_run_skew',
    # 'cifar10_run_none',
    # 'mnist_run_skew0',
    # 'mnist_run_any0',
    # 'mnist_run_regfunc0',
    # 'mnist_run_skew'
    # 'PDEFuncswish_skew',
    # 'mnist_run_none'
]

fig, axs = plt.subplots(2, 3, figsize=(10, 15))
y_max = 0

lower_q = .21
upper_q = 1-lower_q

# Check some model statistics (not all weights are 0...)
# model_key = jrandom.PRNGKey(0)
# d = 5
# func = node.PDEFunc(d=d, width_size=d, depth=2, integrate=False, skew=True) 
# model = node.NeuralODEClassifier(func, in_size=28*28, out_size=10, key=model_key, rtol=1e-2, atol=1e1, use_out=True)

for i, folder in enumerate(folders):
    if folder[-2] == 'c':
        lib = 'test/'
        
    else:
        lib = 'tests/'
    adjoints = joblib.load(lib + folder + '/adjoint_norm.pkl')
    
    label=folder
    if len(adjoints) > 0:
        mean_var = [np.nanmean(np.quantile(adjoint, q=upper_q, axis=-1)/np.quantile(adjoint, q=lower_q, axis=-1)) for adjoint in adjoints]
        # nan_mask = np.isnan(np.array([np.quantile(adjoint, q=.79, axis=-1)/np.quantile(adjoint, q=.21, axis=-1) for adjoint in adjoints]))
        # num_nans = np.sum(nan_mask)
        # print(f'got {num_nans/np.size(nan_mask)*100}% nans\n')
        # mean_var = [np.mean(np.var(adjoint, axis=-1)) for adjoint in adjoints]
        epochs = np.arange(len(adjoints))
        y_max = max([y_max, 1.1*max(mean_var)])
        axs[0, 0].plot(
            epochs,
            mean_var, 
            label=label
        )
        sample = int(len(adjoints)) - 1
        try:
            sample_adjoint = adjoints[sample][0, :]
        except:
            sample_adjoint = adjoints[sample]
        axs[0, 1].plot(
            np.arange(len(sample_adjoint)), 
            sample_adjoint,
            label=label
        )

        # axs[0, 2].plot(

        # )

        nfes = joblib.load(lib + folder + '/num_steps.pkl')
        mean_nfe = [np.mean(nfe) for nfe in nfes]
        epochs = np.arange(len(mean_nfe))
        axs[1, 0].plot( 
            epochs, 
            mean_nfe, 
            label=folder 
        )

        state_norm = joblib.load(lib + folder + '/state_norm.pkl')
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
